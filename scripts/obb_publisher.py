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
                       fit_rect_via_pca, axis_aligned_min_area_rect, visualize_and_fit_min_rect,
                       sort_and_reindex_clusters, plot_cluster_with_names, cluster_plane_objects_DBSCAN,
                       save_3d_pointcloud_with_obb, COLOR_TEMPLATE, create_obb_geometry,
                       create_obb_lineset, plot_clusters_with_legend, remove_near_black_points)


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
        origninal_pcd = pcd
        pcd = crop_cloud(pcd)
        o3d.visualization.draw_geometries([pcd], window_name='msg input')
        pcd = clean_pointcloud(pcd)
        o3d.visualization.draw_geometries([pcd], window_name='Preprocessed Point Cloud')
        _, table, others = extract_table_plane(pcd)

        #table_red = table
        # others_green = others
        # 桌面赋红色（RGB: 1, 0, 0）
        # table_red.paint_uniform_color([1.0, 0.0, 0.0])

        # 物体赋绿色（RGB: 0, 1, 0）
        # others_green.paint_uniform_color([0.0, 1.0, 0.0])

        # 可视化
        # o3d.visualization.draw_geometries([table_red, others_green], window_name="Extract Table and Objects")
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

        # o3d.visualization.draw_geometries([projected_pcd], window_name = "Projected Point Cloud")



        #remove color black
        projected_pcd = remove_near_black_points(projected_pcd, threshold=0.6)
        projected_pcd = clean_pointcloud(projected_pcd, voxel_size=0.001, min_points=50)
        projected_pcd = clean_pointcloud(projected_pcd, voxel_size=0.001, min_points=50)
        o3d.visualization.draw_geometries([projected_pcd], window_name="Projected Point Cloud")
        points_2d = np.asarray(projected_pcd.points)[:, :2]
        colors_2d = np.asarray(projected_pcd.colors)

        rospy.loginfo("Cluster projected points.")

        # Step 2: Cluster
        labels = cluster_plane_objects(points_2d)
        labels_dbscan = cluster_plane_objects_DBSCAN(points_2d)

        plot_clusters_with_legend(points_2d,labels,method='HDBSCAN')
        plot_clusters_with_legend(points_2d,labels_dbscan,method='DBSCAN')

        rospy.loginfo("Classify clustered points.")
        # Step 3: Classify
        cluster_colors_info = extract_main_colors_by_cluster(points_2d, colors_2d, labels)
        classified = classify_clusters_with_templates_strict(cluster_colors_info, self.template)



        rospy.loginfo(f"Classfied into {len(classified) - 1} clusters.")
        all_points = np.asarray(projected_pcd.points)

        # 🧠 准备用于可视化的数据
        name_mapping = {}
        obb_mapping = {}

        # 准备一个映射表：每类物体名称 -> 拟合函数 + 高度
        fit_rect_func_by_name = {
            "tip rack": lambda pts: (axis_aligned_min_area_rect(pts), 0.10),
            "pipette": lambda pts: (axis_aligned_min_area_rect(pts), 0.24),
            "pH sensor": lambda pts: (fit_rect_via_pca(pts), 0.18),
            "beaker": lambda pts: (fit_rect_via_pca(pts), 0.08),
        }

        sorted_clusters = sort_and_reindex_clusters(points_2d, labels, classified, fit_rect_func_by_name)

        for new_label, obj_name, rect, height in sorted_clusters:
            name_mapping[new_label] = obj_name
            obb_mapping[new_label] = rect
            self.publish_obb(new_label, rect, height, header, obj_name)

        # 🧪 可视化聚类 + OBB
        plot_cluster_with_names(points_2d, labels, name_mapping, obb_mapping)
        # 🧪 保存 3D 点云和 OBB 可视化
        geometries = []
        geometries.append(origninal_pcd)
        for label, obj_name, rect, height in sorted_clusters:
            rospy.loginfo(f"drawing OBB of {obj_name}")
            obb = create_obb_lineset(rect[0], rect[1], rect[2], height)
            geometries.append(obb)

        o3d.visualization.draw_geometries(geometries, window_name="Full PointCloud with OBBs")

    def publish_obb(self, label, rect, height, header, obj_name):
        center_xy, (length, width), angle_deg = rect
        angle_rad = np.deg2rad(angle_deg)

        pose = Pose()
        pose.position.x = center_xy[0]
        pose.position.y = center_xy[1]
        pose.position.z = 0.0

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

        topic_name = f"/labware" + f"/{obj_name.replace(' ', '_')}"
        topic_name = topic_name + f'{label}' + f'/obb'
        if topic_name not in self.publishers:
            self.publishers[topic_name] = rospy.Publisher(topic_name, LabwareOBB, queue_size=1, latch=True)

        self.publishers[topic_name].publish(msg)
        rospy.loginfo(f"Published OBB for {obj_name}{label}")


if __name__ == "__main__":
    try:
        OBBPublisher()
    except rospy.ROSInterruptException:
        pass
