#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from matplotlib.mlab import window_none
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from my_robot_msgs.msg import LabwareOBB
import struct

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)  # 放在最前面，避免被覆盖

print("[DEBUG] Added to sys.path:", script_dir)

# ⚠️导入你已有的所有函数，例如：
from obb_utils import (crop_cloud, clean_pointcloud, extract_table_plane, merge_objects_with_table_residual,
                       project_to_xoy, cluster_plane_objects, extract_main_colors_by_cluster,
                       load_color_templates_from_txt, classify_clusters_with_templates_strict,
                       fit_rect_via_pca, visualize_and_fit_min_rect, plot_cluster_with_names, COLOR_TEMPLATE)


class OBBPublisher:
    def __init__(self):
        rospy.init_node("obb_publisher_node")
        self.template_path = rospy.get_param("~template_path", "/color_templates.txt")
        self.publishers = {}  # name -> rospy.Publisher

        self.template = COLOR_TEMPLATE

        self.sub = rospy.Subscriber("/merged_cloud", PointCloud2, self.pointcloud_callback, queue_size=1)

        rospy.loginfo("OBB Publisher Node Started")
        rospy.spin()

    def pointcloud_callback(self, msg):
        rospy.loginfo("Received PointCloud2")
        # PointCloud2 -> Open3D
        cloud = self.pointcloud2_to_open3d(msg)

        if len(cloud.points) == 0:
            rospy.logwarn("Empty point cloud received.")
            return

        try:
            self.process_and_publish_obb(cloud, msg.header)
        except Exception as e:
            rospy.logerr(f"Failed to process point cloud: {e}")

    def pointcloud2_to_open3d(self, ros_cloud):
        points = []
        colors = []
        field_names = [field.name for field in ros_cloud.fields]
        use_rgb = 'rgb' in field_names
        for point in pc2.read_points(ros_cloud, skip_nans=True,
                                     field_names=("x", "y", "z", "rgb") if use_rgb else ("x", "y", "z")):
            if use_rgb:
                x, y, z, rgb = point
                packed = struct.unpack('I', struct.pack('f', rgb))[0]
                r = (packed >> 16) & 0xFF
                g = (packed >> 8) & 0xFF
                b = packed & 0xFF
                colors.append([r / 255.0, g / 255.0, b / 255.0])
            else:
                x, y, z = point
                colors.append([0.5, 0.5, 0.5])  # default gray
            points.append([x, y, z])

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(np.array(points))
        cloud_o3d.colors = o3d.utility.Vector3dVector(np.array(colors))
        return cloud_o3d

    def process_and_publish_obb(self, pcd, header):
        # Step 1: Clean & Segment
        pcd = crop_cloud(pcd)
        o3d.visualization.draw_geometries([pcd], window_name='msg input')
        pcd = clean_pointcloud(pcd)
        o3d.visualization.draw_geometries([pcd], window_name='cleaned pcd')
        _, table, others = extract_table_plane(pcd)
        o3d.visualization.draw_geometries([others], window_name='objects')
        table_pts, table_colors = np.asarray(table.points), np.asarray(table.colors)
        obj_pts, obj_colors = np.asarray(others.points), np.asarray(others.colors)

        pts_filtered, colors_filtered, _ = merge_objects_with_table_residual(
            table_pts, table_colors, obj_pts, obj_colors)

        all_pts = np.vstack([pts_filtered, obj_pts])
        all_colors = np.vstack([colors_filtered, obj_colors])

        rospy.loginfo("Project points to xOy plane.")

        projected = project_to_xoy(all_pts)
        projected_pcd = o3d.geometry.PointCloud()
        projected_pcd.points = o3d.utility.Vector3dVector(projected)
        projected_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        o3d.visualization.draw_geometries([projected_pcd])

        projected_pcd = clean_pointcloud(projected_pcd, voxel_size=0.001)
        points_2d = np.asarray(projected_pcd.points)[:, :2]
        colors_2d = np.asarray(projected_pcd.colors)

        rospy.loginfo("Cluster projected points.")

        # Step 2: Cluster
        labels = cluster_plane_objects(points_2d)

        rospy.loginfo("Classify clustered points.")
        # Step 3: Classify
        cluster_colors_info = extract_main_colors_by_cluster(points_2d, colors_2d, labels)
        classified = classify_clusters_with_templates_strict(cluster_colors_info, self.template)

        rospy.loginfo(f"Classfied into {len(classified) - 1} clusters.")
        all_points = np.asarray(projected_pcd.points)

        # 🧠 准备用于可视化的数据
        name_mapping = {}
        obb_mapping = {}

        for label in set(labels):
            if label == -1 or label not in classified:
                continue

            obj_name = classified[label][0]
            cluster_mask = (labels == label)
            points2d = points_2d[cluster_mask]
            # points3d = all_pts[cluster_mask]

            # Step 4: Fit OBB
            if obj_name == "tip rack":
                rect = visualize_and_fit_min_rect(points2d, alpha=0.0001)
                height = 0.10
            elif obj_name == "pipette":
                rect = fit_rect_via_pca(points2d)
                height = 0.30
            elif obj_name == "pH sensor":
                rect = fit_rect_via_pca(points2d)
                height = 0.20
            elif obj_name == "beaker":
                rect = fit_rect_via_pca(points2d)
                height = 0.08
            else:
                continue

            name_mapping[label] = obj_name
            obb_mapping[label] = rect

            self.publish_obb(label, rect, height, header, obj_name)

        # 🧪 可视化聚类 + OBB
        plot_cluster_with_names(points_2d, labels, name_mapping, obb_mapping)

    def publish_obb(self, label, rect, height, header, obj_name):
        center_xy, (length, width), angle_deg = rect
        angle_rad = np.deg2rad(angle_deg)

        pose = Pose()
        pose.position.x = center_xy[0]
        pose.position.y = center_xy[1]
        pose.position.z = height / 2.0

        quat = R.from_euler('z', angle_rad).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        msg = LabwareOBB()
        msg.header = header
        msg.pose = pose
        msg.x_length = length
        msg.y_width = width
        msg.z_height = height

        topic_name = f"/labware"+f"/{obj_name.replace(' ', '_')}"
        topic_name = topic_name + f'{label}'+f'/obb'
        if topic_name not in self.publishers:
            self.publishers[topic_name] = rospy.Publisher(topic_name, LabwareOBB, queue_size=1, latch=True)

        self.publishers[topic_name].publish(msg)
        rospy.loginfo(f"Published OBB for {obj_name}{label}")


if __name__ == "__main__":
    try:
        OBBPublisher()
    except rospy.ROSInterruptException:
        pass
