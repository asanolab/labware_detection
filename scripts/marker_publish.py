#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect ArUco markers, obtain their full 6-DoF poses in camera frame,
then convert to base_link frame via TF2.
"""
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import tf


class MarkerPosePublisher:
    def __init__(self):
        rospy.init_node("aruco_marker_pose_publisher")

        self.bridge = CvBridge()
        self.depth_img = None
        self.K = None
        self.intrinsics_ready = False

        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.base_frame = rospy.get_param("~base_frame", "link_base")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # publishers
        self.pose_pub = rospy.Publisher("/marker_pose_base", PoseStamped, queue_size=10)
        self.image_pub = rospy.Publisher("/aruco_image", Image, queue_size=1)
        self.marker_pubs = dict()

        # aruco
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        self.aruco_params = aruco.DetectorParameters()

        # subs
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.cam_info_cb)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_cb)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb)


        rospy.loginfo("MarkerPosePublisher initialized.")
        rospy.spin()

    def cam_info_cb(self, msg):
        if not self.intrinsics_ready:
            self.K = np.array(msg.K).reshape(3, 3)
            self.intrinsics_ready = True
            rospy.loginfo("Camera intrinsics received.")

    def depth_cb(self, msg):
        # 16UC1 depth image in mm → convert to meters
        raw_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        self.depth_img = raw_depth.astype(np.float32) / 1000.0

    def image_cb(self, msg):
        if not self.intrinsics_ready or self.depth_img is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        corners, ids, _ = aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
            return

        aruco.drawDetectedMarkers(img, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i][0]
            u, v = np.mean(pts, axis=0).astype(int)

            if not (0 <= v < self.depth_img.shape[0] and 0 <= u < self.depth_img.shape[1]):
                continue

            z = self.depth_img[v, u]
            if np.isnan(z) or z <= 0.1:
                continue

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # 姿态估计（旋转）
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.K, None)
            rvec = rvecs[i]
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            quat = tf.transformations.quaternion_from_matrix(np.vstack([
                np.hstack([rotation_matrix, np.array([[0], [0], [0]])]),
                np.array([[0, 0, 0, 1]])
            ]))

            # 构造 pose
            pose_cam = PoseStamped()
            pose_cam.header = msg.header
            pose_cam.header.frame_id = self.camera_frame
            pose_cam.pose.position.x = x
            pose_cam.pose.position.y = y
            pose_cam.pose.position.z = z
            pose_cam.pose.orientation.x = quat[0]
            pose_cam.pose.orientation.y = quat[1]
            pose_cam.pose.orientation.z = quat[2]
            pose_cam.pose.orientation.w = quat[3]

            try:
                pose_cam.header.stamp = rospy.Time(0)
                pose_base = self.tf_buffer.transform(pose_cam, self.base_frame, rospy.Duration(0.1))
                topic_name = f"/marker_pose_base/{marker_id}"
                if marker_id not in self.marker_pubs:
                    self.marker_pubs[marker_id]=rospy.Publisher(topic_name,PoseStamped,queue_size=10)
                self.marker_pubs[marker_id].publish(pose_base)

                rospy.loginfo(f"[ID {marker_id}] base_link pose: "
                           f"x={pose_base.pose.position.x:.3f}, "
                           f"y={pose_base.pose.position.y:.3f}, "
                           f"z={pose_base.pose.position.z:.3f}")

            except Exception as e:
                rospy.logwarn(f"TF failed: {e}")

        # 显示 marker
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))


if __name__ == "__main__":
    try:
        MarkerPosePublisher()
    except rospy.ROSInterruptException:
        pass
