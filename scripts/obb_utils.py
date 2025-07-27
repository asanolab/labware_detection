import numpy as np
import open3d as o3d
import cv2
import alphashape
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import hdbscan
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
    #print("Cropped successfully")
    # o3d.visualization.draw_geometries([cropped], window_name='Cropped')
    return cropped

def clean_pointcloud(pcd, voxel=True, voxel_size=0.001, statistical=True, radius=True, dbscan=True):
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


def cluster_plane_objects(points_2d, eps=0.01, min_samples=20, min_cluster_points=50):
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