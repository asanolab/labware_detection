# Point-Cloud-Based Labware Detection
Detect the oriented bounding boxs of labware on the table based on point cloud captured in multiple robot poses, with the Cluster Algorithms.

```
$ rosrun robot_control robot_init.py
$ roslaunch pc_detection pc_slam.launch
```
Available args:
```
  <arg name="pose_file" default="slam_points.txt"/>
  <arg name="save_file" default="merged_cloud.pcd"/>
  <arg name="marker_file" default="marker_position"/>
```
Use your own pose\_file and marker\_file.

# Marker publisher
```
$ rosrun pc_detection marker_pulish.py
```
