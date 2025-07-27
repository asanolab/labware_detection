import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import tf2_ros
import struct
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_matrix
from std_msgs.msg import Bool
from my_robot_msgs.msg import MovePose
from scipy.spatial.transform import Rotation

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import numpy as np
import tf2_ros
import struct
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_matrix
from std_msgs.msg import Bool
from my_robot_msgs.msg import MovePose
import math
import os
import copy


class PointCloudCollector:
    def __init__(self):
        rospy.init_node('pointcloud_collector')

        # Params
        self.base_frame = "link_base"
        self.cloud_topic = "/camera/depth/color/points"
        self.rgb_topic = "/camera/color/image_raw"
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw"
        self.camera_info_topic = "/camera/color/camera_info"
        self.voxel_size = 0.005
        self.pose_file = rospy.get_param("~pose_file", "slam_points.txt")
        self.marker_file = rospy.get_param("~marker_file", "marker_positions.txt")
        self.save_file = rospy.get_param("~save_file", "/home/wsy/merged_cloud.pcd")

        # Load files
        self.poses = self.load_poses_from_file(self.pose_file)
        self.marker_positions = self.load_markers_from_file(self.marker_file)

        # Camera intrinsics
        info_msg = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        self.K = np.array(info_msg.K).reshape(3, 3)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform = None

        # ROS pub/sub
        self.cloud_pub = rospy.Publisher('/merged_cloud', PointCloud2, queue_size=1, latch=True)
        self.pose_pub = rospy.Publisher("/arm_control/move_pose", MovePose, queue_size=1)
        rospy.Subscriber("/arm_control/move_done", Bool, self.move_done_callback)

        self.bridge = CvBridge()

        # State
        self.current_index = 0
        self.collected_clouds = []
        self.move_done = False
        self.pose_sent = False

        self.detected_centers = []
        self.real_centers = []

        rospy.on_shutdown(self.save_and_visualize_cloud)
        rospy.loginfo("PointCloud Collector Node Started")

    def load_poses_from_file(self, filepath):
        poses = []
        if not os.path.isfile(filepath):
            rospy.logerr(f"Pose file not found: {filepath}")
            return poses
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    rospy.logwarn(f"Ignoring malformed line: {line}")
                    continue
                x, y, z, r, p, y_ = map(float, parts)
                r = math.radians(r)
                p = math.radians(p)
                y_ = math.radians(y_)
                poses.append([x, y, z, r, p, y_])
        rospy.loginfo(f"Loaded {len(poses)} target poses.")
        return poses

    def load_markers_from_file(self, filepath):
        markers = []
        if not os.path.isfile(filepath):
            rospy.logerr(f"Marker file not found: {filepath}")
            return markers
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                x, y, z = map(float, parts)
                markers.append([x / 1000.0, y / 1000.0, z / 1000.0])
        rospy.loginfo(f"Loaded {len(markers)} marker positions.")
        return np.array(markers)

    def move_done_callback(self, msg):
        self.move_done = msg.data
        rospy.loginfo(f"🔔 Received /arm_control/move_done = {msg.data}")

    def publish_pose_once(self):
        pose = self.poses[self.current_index]
        msg = MovePose()
        msg.pose = pose

        # 确保机械臂节点已经订阅了
        while self.pose_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logwarn("Waiting for /arm_control/move_pose subscriber...")
            rospy.sleep(0.5)

        self.pose_pub.publish(msg)
        rospy.loginfo(f"📤 Published pose {self.current_index + 1}/{len(self.poses)}: {pose}")

        self.pose_sent = True

    def get_aruco_poses(self):
        rgb_msg = rospy.wait_for_message(self.rgb_topic, Image, timeout=10.0)
        depth_msg = rospy.wait_for_message(self.depth_topic, Image, timeout=10.0)

        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") / 1000.0  # 转为米

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(rgb_img)

        centers_base, Rs_base = [], []

        if ids is None:
            return centers_base, Rs_base

        # 获取 TF: camera_color_optical_frame → base_link
        if self.transform is None:
            transform = self.tf_buffer.lookup_transform(self.base_frame, "camera_color_optical_frame", rospy.Time(0),
                                                        rospy.Duration(1.0))
        else:
            transform = self.transform
        T_cam_to_base = self.transform_to_matrix(transform)

        camera_matrix = self.K
        dist_coeffs = np.zeros((5, 1))

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, camera_matrix, dist_coeffs)

        for idx, id_ in enumerate(ids.flatten()):
            if id_ != 100:
                continue
            pts_2d = corners[idx][0]  # (4, 2)
            pts_3d_cam = []

            for (u, v) in pts_2d:
                z = depth_img[int(round(v)), int(round(u))]
                if z == 0:
                    continue  # 跳过无效深度
                x = (u - self.K[0, 2]) * z / self.K[0, 0]
                y = (v - self.K[1, 2]) * z / self.K[1, 1]
                pts_3d_cam.append([x, y, z])

            if len(pts_3d_cam) < 3:
                continue

            pts_3d_cam = np.array(pts_3d_cam)
            center_cam = pts_3d_cam.mean(axis=0)

            rvec = rvecs[idx]
            R_cam, _ = cv2.Rodrigues(rvec)

            # 转换到 base_link
            center_base = T_cam_to_base[:3, :3] @ center_cam + T_cam_to_base[:3, 3]
            R_base = T_cam_to_base[:3, :3] @ R_cam

            centers_base.append(center_base)
            Rs_base.append(R_base)

            # 可视化调试
            # cv2.aruco.drawDetectedMarkers(rgb_img, [corners[idx]], np.array([id_]))
            # cv2.drawFrameAxes(rgb_img, camera_matrix, dist_coeffs, rvec, tvecs[idx], 0.03)

        # cv2.imshow("ArUco Detection Debug", rgb_img)
        # cv2.waitKey(10)

        return centers_base, Rs_base

    def match_to_known_markers(self, detected_centers):
        detected_centers = np.array(detected_centers)

        best_detected = None
        best_real = None
        best_dist = float('inf')

        for center in detected_centers:
            dists = np.linalg.norm(self.marker_positions - center, axis=1)
            idx = np.argmin(dists)
            real_center = self.marker_positions[idx]
            dist = np.linalg.norm(center - real_center)

            if dist < best_dist:
                best_dist = dist
                best_detected = center
                best_real = real_center

        rospy.loginfo(f"Selected marker matching error: {best_dist:.4f} m")

        # 保存用于标定
        self.detected_centers.append(best_detected)
        self.real_centers.append(best_real)

        return np.array([best_detected]), np.array([best_real])

    def align_plane_to_xy(self, cloud):
        points = np.asarray(cloud.points)
        mask = (points[:, 2] >= -0.05) & (points[:, 2] <= 0.05)
        points = points[mask]

        if len(points) < 50:
            rospy.logwarn("Not enough points to fit plane.")
            return np.eye(4)

        # 拟合平面
        points_mean = points.mean(axis=0)
        centered = points - points_mean
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]

        # 确保 normal 指向 z 正方向
        if normal[2] < 0:
            normal = -normal

        # 旋转矩阵，使 normal 对齐 [0, 0, 1]
        z_axis = np.array([0, 0, 1])
        v = np.cross(normal, z_axis)
        c = np.dot(normal, z_axis)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ points_mean  # 把桌面中心也带到原点附近，便于后续
        return T

    def estimate_rigid_transform(self, source_points, source_rotations, target_points):
        src = np.array(source_points)
        tgt = np.array(target_points)
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt, axis=0)
        src_centered = src - centroid_src
        tgt_centered = tgt - centroid_tgt

        '''
        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src

        # 进一步利用姿态：如果只有一个 marker，直接用检测到的姿态对齐 xoy
        if len(src) == 1:
            R = source_rotations[0]
            R_target = np.eye(3)
            R = R_target @ R.T
            t = tgt[0] - R @ src[0]
        '''

        T = np.eye(4)

        R = np.eye(3)
        t = centroid_tgt - centroid_src

        T[:3, :3] = R
        T[:3, 3] = t

        rospy.loginfo(f'transform matrix: {T}')

        return T

    def estimate_translation_only(self, source_points, target_points):
        src = np.mean(np.array(source_points), axis=0)
        tgt = np.mean(np.array(target_points), axis=0)
        T = np.eye(4)
        T[:3, 3] = tgt - src
        rospy.loginfo(f"Translation only T:\n{T}")
        return T

    def capture_and_store(self):

        def highlight_marker_area(cloud_o3d, marker_center, radius=0.01):
            points = np.asarray(cloud_o3d.points)
            colors = np.asarray(cloud_o3d.colors)

            # 计算所有点与 marker_center 的距离
            dists = np.linalg.norm(points - marker_center, axis=1)

            # 找到半径范围内的点，涂成红色
            colors[dists < radius] = [1.0, 0.0, 0.0]  # 红色

            # 更新点云颜色
            cloud_o3d.colors = o3d.utility.Vector3dVector(colors)

            # 可视化
            # o3d.visualization.draw_geometries([cloud_o3d])

        rospy.loginfo(f"⏳ Start capturing multi-frame at pose {self.current_index + 1}")
        accumulated_cloud = o3d.geometry.PointCloud()
        frame_count = 10  # 采集 10 帧叠加

        for i in range(frame_count):
            try:
                cloud_msg = rospy.wait_for_message(self.cloud_topic, PointCloud2, timeout=5.0)
                rospy.sleep(1.0)
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, cloud_msg.header.frame_id,
                    rospy.Time(0), rospy.Duration(1.0))
                self.transform = transform
            except Exception as e:
                rospy.logwarn(f"TF or Cloud error: {e}")
                continue

            cloud_o3d = self.ros_to_o3d_cloud(cloud_msg)

            # 在变换前筛选z
            points = np.asarray(cloud_o3d.points)
            mask = (points[:, 2] >= 0.2) & (points[:, 2] <= 0.5)
            points = points[mask]
            colors = np.asarray(cloud_o3d.colors)[mask]

            cloud_o3d = o3d.geometry.PointCloud()
            cloud_o3d.points = o3d.utility.Vector3dVector(points)
            cloud_o3d.colors = o3d.utility.Vector3dVector(colors)

            # 变换到 base_link
            cloud_o3d.transform(self.transform_to_matrix(transform))

            # ✅ 第二步筛选 XYZ 在你指定的范围内
            points = np.asarray(cloud_o3d.points)
            mask = (
                    (points[:, 0] >= -0.55) & (points[:, 0] <= 0.10) &
                    (points[:, 1] >= 0.0) & (points[:, 1] <= 0.7) &
                    (points[:, 2] >= -0.1) & (points[:, 2] <= 0.5)
            )
            points = points[mask]
            colors = np.asarray(cloud_o3d.colors)[mask]

            cloud_o3d = o3d.geometry.PointCloud()
            cloud_o3d.points = o3d.utility.Vector3dVector(points)
            cloud_o3d.colors = o3d.utility.Vector3dVector(colors)

            accumulated_cloud += cloud_o3d
            rospy.sleep(0.2)  # 等小半秒防止缓存

        if len(accumulated_cloud.points) == 0:
            return

        centers, Rs = self.get_aruco_poses()
        if len(centers) == 0:
            return

        # 点云预处理
        accumulated_cloud = accumulated_cloud.voxel_down_sample(self.voxel_size)
        accumulated_cloud, _ = accumulated_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # 找到匹配真实位置
        sources, targets = self.match_to_known_markers(centers)

        # 姿态矫正（align z up）旋转变换
        R = self.align_plane_to_xy(accumulated_cloud)
        accumulated_cloud.transform(R)

        centers_rotated = [(R @ np.append(source, 1))[:3] for source in sources]

        T = self.estimate_translation_only(centers_rotated, targets)

        # aruco平移变换
        accumulated_cloud.transform(T)
        # highlight_marker_area(accumulated_cloud, np.array([-0.11, 0.16, 0.0]), radius=0.01)
        self.collected_clouds.append(accumulated_cloud)

    def run(self):
        rate = rospy.Rate(0.1)
        while not rospy.is_shutdown() and self.current_index < len(self.poses):
            if not self.pose_sent:
                self.publish_pose_once()
                rospy.sleep(2.0)

            if self.move_done:
                rospy.loginfo("📥 Robot at target. Start capturing point cloud.")
                rospy.sleep(1.0)
                self.capture_and_store()
                self.current_index += 1
                self.pose_sent = False  # Ready for next pose
                self.move_done = False  # Reset until next confirmation
                # self.move_done_pub.publish(False)  # Ensure downstream knows we reset

            rate.sleep()

        rospy.loginfo("🎉 All poses visited. Finalizing merged cloud.")
        self.save_and_visualize_cloud()
        rospy.spin()

    def save_and_visualize_cloud(self):
        if not self.collected_clouds:
            rospy.loginfo("No clouds to save.")
            return
        merged_cloud = copy.deepcopy(self.collected_clouds[0])
        for new_cloud in self.collected_clouds[1:]:
            merged_cloud += copy.deepcopy(new_cloud)
            merged_cloud = merged_cloud.voxel_down_sample(self.voxel_size)
        merged_cloud, _ = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        o3d.io.write_point_cloud('/home/wsy/' + self.save_file, merged_cloud)
        rospy.loginfo(f"Merged cloud saved as {self.save_file}")
        ros_cloud_msg = self.o3d_to_ros_cloud(merged_cloud)
        self.cloud_pub.publish(ros_cloud_msg)
        rospy.loginfo("📤 Published merged cloud. Node will keep alive for subscribers.")
        rospy.loginfo(f"detected centers: {self.detected_centers}, real centers: {self.real_centers}")

    def ros_to_o3d_cloud(self, ros_cloud):
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

    def transform_to_matrix(self, transform: TransformStamped):
        trans = transform.transform.translation
        rot = transform.transform.rotation
        matrix = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        matrix[0, 3] = trans.x
        matrix[1, 3] = trans.y
        matrix[2, 3] = trans.z
        return matrix

    def o3d_to_ros_cloud(self, cloud_o3d):
        points = np.asarray(cloud_o3d.points)
        colors = np.asarray(cloud_o3d.colors)
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.base_frame
        cloud_data = []
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
            cloud_data.append([x, y, z, rgb])
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1)
        ]
        cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        return cloud_msg


if __name__ == '__main__':
    collector = PointCloudCollector()
    collector.run()
