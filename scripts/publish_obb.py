#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from my_robot_msgs.msg import LabwareOBB

def publish_sample_obb():
    rospy.init_node('labware_obb_publisher')

    pub = rospy.Publisher('/labware/beaker/obb', LabwareOBB, queue_size=10)

    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        obb_msg = LabwareOBB()

        # Header
        obb_msg.header = Header()
        obb_msg.header.stamp = rospy.Time.now()
        obb_msg.header.frame_id = "link_base"

        # Pose
        obb_msg.pose = Pose()
        obb_msg.pose.position = Point(-0.0675, 0.275, 0.0)  # 中心点坐标
        obb_msg.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)  # 无旋转，单位四元数

        # Dimensions
        obb_msg.x_length = 0.15  # 长 10cm
        obb_msg.y_width = 0.11  # 宽 8cm
        obb_msg.z_height = 0.10  # 高 12cm

        pub.publish(obb_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_sample_obb()
    except rospy.ROSInterruptException:
        pass