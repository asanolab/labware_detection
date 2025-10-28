import numpy as np
import open3d as o3d
import cv2
import alphashape
import sklearn.cluster
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import hdbscan
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time


def crop_cloud(pcd):
    points = np.asarray(pcd.points)
    mask = (points[:, 0] >= -0.35) & (points[:, 0] <= 0.06) & \
           (points[:, 1] >= 0.1) & (points[:, 1] <= 0.55) & \
           (points[:, 2] >= -0.02) & (points[:, 2] <= 0.4)
    cropped = pcd.select_by_index(np.where(mask)[0])
    # print("Cropped successfully")
    # o3d.visualization.draw_geometries([cropped], window_name='Cropped')
    return cropped


def clean_pointcloud(pcd, voxel=True, voxel_size=0.001, statistical=True, radius=True, dbscan=True, min_points=10):
    pcd_clean = pcd
    if voxel:
        pcd_clean = pcd_clean.voxel_down_sample(voxel_size)
    if statistical:
        pcd_clean, _ = pcd_clean.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if radius:
        pcd_clean, _ = pcd_clean.remove_radius_outlier(nb_points=10, radius=0.01)
    if dbscan:
        labels = np.array(pcd_clean.cluster_dbscan(eps=0.1, min_points=10))
        pcd_clean = pcd_clean.select_by_index(np.where(labels >= 0)[0])
    return pcd_clean


def extract_table_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    table = pcd.select_by_index(inliers)
    others = pcd.select_by_index(inliers, invert=True)
    return plane_model, table, others


def remove_table_by_colors(points, colors, threshold=30, n_clusters=2):
    colors_rgb = (colors * 255).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(colors_rgb)
    main_colors = kmeans.cluster_centers_
    dists = np.linalg.norm(colors_rgb[:, None, :] - main_colors[None, :, :], axis=2)
    min_dists = dists.min(axis=1)
    mask = min_dists > threshold
    points_filtered = points[mask]
    colors_filtered = colors[mask]
    return points_filtered, colors_filtered, main_colors / 255.0


def merge_objects_with_table_residual(table_points, table_colors, object_points, object_colors, threshold=30):
    table_points_filtered, table_colors_filtered, main_color = remove_table_by_colors(
        table_points, table_colors, threshold
    )
    points_combined = np.vstack([table_points_filtered, object_points])
    colors_combined = np.vstack([table_colors_filtered, object_colors])
    return object_points, object_colors, main_color


def project_to_xoy(points):
    projected_points = points.copy()
    projected_points[:, 2] = 0.0
    return projected_points


def remove_near_black_points(pcd, threshold=0.1):
    """
    移除颜色接近黑色的点（RGB距离小于阈值）
    参数:
        pcd: Open3D 点云
        threshold: 黑色的欧氏距离阈值，推荐值 0.1（范围 [0,1]）
    返回:
        过滤后的点云
    """
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    # 计算颜色到黑色的欧氏距离
    black = np.array([0, 0, 0])
    dist = np.linalg.norm(colors - black, axis=1)

    # 选取距离较远的（不是接近黑色的）
    keep_mask = dist > threshold

    # 创建新点云
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[keep_mask])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[keep_mask])

    return filtered_pcd


def cluster_plane_objects(points_2d, eps=0.01, min_samples=20, min_cluster_points=100):
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_points, min_samples=min_samples).fit_predict(points_2d)
    new_labels = labels.copy()
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            continue
        count = np.sum(labels == k)
        if count < min_cluster_points:
            new_labels[labels == k] = -1
    return new_labels


def cluster_plane_objects_DBSCAN(points_2d, eps=0.01, min_samples=35, min_cluster_points=30):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_2d)
    new_labels = labels.copy()
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            continue
        count = np.sum(labels == k)
        if count < min_cluster_points:
            new_labels[labels == k] = -1

    return new_labels


def plot_clusters_with_legend(points_2d, labels, method):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels[unique_labels != -1])

    # 为聚类分配颜色
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(unique_labels))]

    # 画图
    plt.figure(figsize=(8, 6))
    for idx, k in enumerate(unique_labels):
        if k == -1:
            # 噪声点为灰色
            col = (0.5, 0.5, 0.5, 0.5)
            label = "Noise"
        else:
            col = colors[idx]
            label = f"Cluster {k}"
        class_member_mask = (labels == k)
        xy = points_2d[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=10, color=col, label=label)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"/tmp/cluster_{method}_debug_vis"
    os.makedirs(save_dir, exist_ok=True)
    plt.title(f'{method}: Clustered Points (Total Clusters = {num_clusters})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis("equal")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
    save_path = os.path.join(save_dir, f"obb_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[DEBUG] Saved OBB visualization to {save_path}")


def extract_main_colors_by_cluster(points_2d, colors, labels, k=2):
    unique_labels = set(labels)
    cluster_colors_info = {}
    for idx, label in enumerate(sorted(unique_labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_colors = colors[mask]
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(cluster_colors)
        centers = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_, minlength=k)
        percentages = counts / counts.sum()
        color_info = [(percentages[i], centers[i]) for i in range(k)]
        cluster_colors_info[label] = color_info
    return cluster_colors_info


def classify_clusters_with_templates_strict(cluster_colors_info, templates):
    results = {}
    for label, cluster_colors in cluster_colors_info.items():
        p_cluster = np.array([p for p, _ in cluster_colors])
        c_cluster = np.array([c for _, c in cluster_colors])
        best_obj = "unknown"
        best_score = float('inf')
        for obj_name, template in templates.items():
            p_template = np.array([p for p, _ in template])
            c_template = np.array([c for _, c in template])
            if len(c_cluster) != len(c_template):
                continue
            dist_mat = cdist(c_cluster, c_template)
            row_idx, col_idx = linear_sum_assignment(dist_mat)
            matched_dists = dist_mat[row_idx, col_idx]
            weights = p_cluster[row_idx]
            weighted_score = np.sum(matched_dists * weights)
            if weighted_score < best_score:
                best_score = weighted_score
                best_obj = obj_name
        results[label] = (best_obj, best_score)
    return results


def load_color_templates_from_txt(file_path):
    templates = {}
    current_name = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("object_name:"):
                current_name = line.split(":", 1)[1].strip()
                templates[current_name] = []
            elif line.startswith("color:"):
                parts = line.split(":", 1)[1].strip().split(None, 1)
                percentage = float(parts[0])
                rgb = eval(parts[1])
                templates[current_name].append((percentage, np.array(rgb)))
    return templates


def visualize_and_fit_min_rect(points_2d, alpha=0.02):
    alpha_shape = alphashape.alphashape(points_2d, alpha)
    if alpha_shape.geom_type != 'Polygon':
        return None
    alpha_coords = np.array(alpha_shape.exterior.coords)
    hull = ConvexHull(alpha_coords)
    convex_coords = alpha_coords[hull.vertices]
    convex_coords_cv2 = convex_coords.astype(np.float32)
    rect = cv2.minAreaRect(convex_coords_cv2)
    return rect  # ((cx, cy), (w, h), angle)


def fit_rect_via_pca(points_2d):
    mean = np.mean(points_2d, axis=0)
    centered = points_2d - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    R = eigvecs[:, ::-1]
    rotated = centered @ R
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    extent = max_xy - min_xy
    center_local = (min_xy + max_xy) / 2
    center_global = center_local @ R.T + mean
    angle_rad = np.arctan2(R[1, 0], R[0, 0])
    angle_deg = np.rad2deg(angle_rad)
    length, width = extent
    if width > length:
        length, width = width, length
        angle_deg += 90
    return tuple(center_global), (length, width), angle_deg


def axis_aligned_min_area_rect(points_2d):
    """
    模拟 cv2.minAreaRect 的返回形式，但限制矩形必须平行于坐标轴（angle = 0）

    Args:
        points_2d (np.ndarray): [N, 2] 的二维点坐标

    Returns:
        tuple: ((center_x, center_y), (width, height), angle)
    """
    x_min, y_min = np.min(points_2d, axis=0)
    x_max, y_max = np.max(points_2d, axis=0)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # 为保持一致性，将较长边放在 width 上，angle 为 0°
    if height > width:
        width, height = height, width
        angle = 90.0  # 沿 y 轴方向的长边
    else:
        angle = 0.0  # 沿 x 轴方向的长边

    return ((center_x, center_y), (width, height), angle)


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import time
import cv2


def plot_cluster_with_names(points_2d, labels, name_mapping, obb_mapping):
    """
    可视化聚类 + OBB + 名称标注
    points_2d: 所有投影点的坐标 [N, 2]
    labels: 每个点的聚类标签 [N]
    name_mapping: dict[label -> name]
    obb_mapping: dict[label -> ((cx, cy), (w, h), angle)]
    """
    unique_labels = set(labels)
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        mask = (labels == label)
        pts = points_2d[mask]
        color = cmap(i % 10)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=color, label=name_mapping.get(label, f'cluster {label}'))

        # 使用 OpenCV boxPoints 绘制 OBB
        if label in obb_mapping:
            rect = obb_mapping[label]  # ((cx, cy), (w, h), angle_deg)
            box = cv2.boxPoints(rect).astype(np.float32)
            box = np.vstack([box, box[0]])  # 封闭轮廓
            ax.plot(box[:, 0], box[:, 1], 'k-', linewidth=1.5)
            cx, cy = rect[0]
            ax.text(cx, cy, name_mapping[label], fontsize=8, ha='center', va='center', bbox=dict(fc='white', alpha=0.7))

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Clustered Objects with OBBs and Labels")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "/tmp/obb_debug_vis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"obb_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[DEBUG] Saved OBB visualization to {save_path}")


def sort_and_reindex_clusters(points_2d, labels, classified, fit_rect_func_by_name):
    cluster_info_list = []
    for label in set(labels):
        if label == -1 or label not in classified:
            continue
        obj_name = classified[label][0]
        cluster_mask = (labels == label)
        cluster_points = points_2d[cluster_mask]
        if obj_name not in fit_rect_func_by_name:
            continue
        rect, height = fit_rect_func_by_name[obj_name](cluster_points)
        cx, cy = rect[0]
        cluster_info_list.append((obj_name, cx, cy, label, rect, height))

    # 排序: 先按 obj_name，再按 x, y
    cluster_info_list.sort(key=lambda x: (x[0], x[1], x[2]))

    # 返回结构化结果: new_label, obj_name, rect, height
    sorted_result = []
    for new_label, (obj_name, cx, cy, old_label, rect, height) in enumerate(cluster_info_list):
        sorted_result.append((new_label, obj_name, rect, height))
    return sorted_result


def save_3d_pointcloud_with_obb(clusters):
    """
    将聚类点云和其对应的3D OBB一起保存为一个 Open3D 可视化模型。
    clusters: list of (label, obj_name, points_3d, rect_2d, height)
    save_path: path to save the point cloud with OBB as .ply or .pcd
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "/tmp/obb_debug_vis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"obb_3d_{timestamp}.pcd")
    geometries = []
    for label, obj_name, pts, rect, height in clusters:
        # 点云本身
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(pcd)

        # 构造 OBB
        (cx, cy), (w, l), angle_deg = rect
        angle_rad = np.deg2rad(angle_deg)
        center = np.array([cx, cy, height / 2])
        Rz = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
        obb = o3d.geometry.OrientedBoundingBox(center=center, R=Rz, extent=[w, l, height])
        obb.color = [1, 0, 0]
        geometries.append(obb)

    o3d.io.write_point_cloud(save_path, geometries[0])  # 保存第一类为 pcd
    o3d.visualization.draw_geometries(geometries, window_name="3D OBB with PointCloud")
    print(f"[DEBUG] Saved 3D point cloud with OBBs to visualizer (not as file).")


def create_obb_geometry(center, size, angle_deg, height, color=[1, 0, 0]):
    """
    创建一个 OrientedBoundingBox 几何体用于可视化
    center: (cx, cy)
    size: (w, l)
    angle_deg: 角度
    height: z方向高度
    """
    cx, cy = center
    w, l = size
    cz = height / 2.0
    print(f"x,y,z : {cx}, {cy}, {cz}")
    obb_center = np.array([cx, cy, cz])
    angle_rad = np.deg2rad(angle_deg)
    Rz = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
    extent = np.array([w, l, height])

    obb = o3d.geometry.OrientedBoundingBox(center=obb_center, R=Rz, extent=extent)
    obb.color = color
    return obb


def create_obb_lineset(center_xy, size, angle_deg, height):
    """
    根据 2D OBB + 高度 构造一个用于可视化的 3D OBB LineSet
    """
    cx, cy = center_xy
    w, l = size
    angle_rad = np.deg2rad(angle_deg)

    # 计算底面四个点（OpenCV 顺时针）
    box2d = cv2.boxPoints(((cx, cy), (w, l), angle_deg))
    box3d = []

    # 添加上下两层点
    for z in [0, height]:
        for pt in box2d:
            box3d.append([pt[0], pt[1], z])

    # 构造 LineSet 的连接关系
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7]  # sides
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
    return line_set


# 内置颜色模板
COLOR_TEMPLATE = {
    "beaker": [
        (0.6860, np.array([0.5815, 0.6290, 0.6524])),
        (0.3140, np.array([0.5104, 0.5358, 0.5448]))
    ],
    "tip rack": [
        (0.5695, np.array([0.4137, 0.5549, 0.6628])),
        (0.4305, np.array([0.0986, 0.3801, 0.6203]))
    ],
    "pipette": [
        (0.8984, np.array([0.1492, 0.4708, 0.3247])),
        (0.1016, np.array([0.4547, 0.5407, 0.5154]))
    ],
    "pH sensor": [
        (0.4722, np.array([0.1583, 0.4194, 0.4348])),
        (0.5278, np.array([0.5816, 0.6583, 0.6961]))
    ]
}
