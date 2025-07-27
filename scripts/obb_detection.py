import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.flow import minimum_cut
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint
import alphashape
from scipy.spatial import ConvexHull
import hdbscan
import cv2


def load_and_crop_cloud(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    o3d.visualization.draw_geometries([pcd], window_name='File input')
    points = np.asarray(pcd.points)
    mask = (points[:, 0] >= -0.35) & (points[:, 0] <= 0.06) & \
           (points[:, 1] >= 0.1) & (points[:, 1] <= 0.55) & \
           (points[:, 2] >= -0.02) & (points[:, 2] <= 0.4)
    cropped = pcd.select_by_index(np.where(mask)[0])
    print("Cropped successfully")
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
    print("Table extracted successfully")
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


def cluster_plane_objects(points_2d, eps=0.01, min_samples=20, min_cluster_points=30):
    # labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_2d)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_points, min_samples=min_samples).fit_predict(points_2d)
    new_labels = labels.copy()
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            continue
        count = np.sum(labels == k)
        if count < min_cluster_points:
            new_labels[labels == k] = -1

    # 可视化聚类结果并添加图例
    plt.figure(figsize=(10, 8))
    unique_labels = set(new_labels)

    colors = plt.get_cmap('tab10', len(unique_labels))
    for idx, k in enumerate(sorted(unique_labels)):
        class_member_mask = (new_labels == k)
        xy = points_2d[class_member_mask]
        if k == -1:
            color = 'k'
            label = 'Noise'
        else:
            color = colors(idx)
            label = f'Cluster {k}'
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=5, label=label)

    plt.legend()
    plt.axis('equal')
    plt.title(f'DBSCAN Cluster Result: {len(unique_labels) - (1 if -1 in unique_labels else 0)}')
    plt.show()

    return new_labels


def extract_main_colors_by_cluster(points_2d, colors, labels, k=2):
    """
    对每个聚类簇提取主颜色并统计比例。

    Args:
        points_2d: [N, 2]
        colors: [N, 3]
        labels: [N] DBSCAN 聚类标签
        k: 每个簇中提取的主颜色数

    Returns:
        cluster_colors_info: dict[label] = list of (percentage, color)
    """
    unique_labels = set(labels)
    cluster_colors_info = {}

    # plt.figure(figsize=(10, 5 * len(unique_labels)))
    for idx, label in enumerate(sorted(unique_labels)):
        if label == -1:
            continue
        mask = labels == label
        cluster_colors = colors[mask]
        cluster_points = points_2d[mask]

        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(cluster_colors)
        centers = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_, minlength=k)
        percentages = counts / counts.sum()

        # 存储颜色与比例
        color_info = [(percentages[i], centers[i]) for i in range(k)]
        cluster_colors_info[label] = color_info

        '''
        # 可视化饼图
        plt.subplot(len(unique_labels), 2, 2 * idx + 1)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_colors, s=5)
        plt.title(f"Cluster {label} Points")

        plt.subplot(len(unique_labels), 2, 2 * idx + 2)
        colors_rgb = [tuple(c) for _, c in color_info]
        wedges, texts, autotexts = plt.pie(
            [p for p, _ in color_info],
            colors=colors_rgb,
            labels=[f"{i + 1}: {p * 100:.1f}%" for i, (p, _) in enumerate(color_info)],
            autopct='%1.1f%%'
        )
        plt.title(f"Cluster {label} Main Colors")
        '''

    for i in range(len(cluster_colors_info)):
        print(f'cluster id {i}: main color and percent: {cluster_colors_info[i]}')

    # plt.tight_layout()
    # plt.show()

    return cluster_colors_info


def classify_clusters_with_templates(cluster_colors_info, templates):
    results = {}
    for label, cluster_colors in cluster_colors_info.items():
        # cluster_colors: [(p, [r, g, b]), ...]
        p_cluster = np.array([p for p, _ in cluster_colors])
        c_cluster = np.array([c for _, c in cluster_colors])

        best_obj = "unknown"
        best_score = float('inf')

        for obj_name, template_colors in templates.items():
            p_template = np.array([p for p, _ in template_colors])
            c_template = np.array([c for _, c in template_colors])

            # 距离矩阵：[n_cluster, n_template]
            dist_mat = cdist(c_cluster, c_template, metric='euclidean')
            min_dists = dist_mat.min(axis=1)

            # 使用 cluster 的占比作为权重（不是模板的）
            score = np.sum(p_cluster * min_dists)

            print(f'id: {label}, template: {obj_name}, score:{score}')

            if score < best_score:
                best_score = score
                best_obj = obj_name

        results[label] = (best_obj, best_score)
    return results


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

            # 如果两个模板主颜色数 ≠ 簇的主颜色数，先跳过
            if len(c_cluster) != len(c_template):
                continue  # 或者用 padding 来补齐（可拓展）

            # 距离矩阵：[n_cluster, n_template]
            dist_mat = cdist(c_cluster, c_template)

            # 匈牙利算法找最优匹配
            row_idx, col_idx = linear_sum_assignment(dist_mat)

            # 加权匹配代价（使用簇中的占比为权重）
            matched_dists = dist_mat[row_idx, col_idx]
            weights = p_cluster[row_idx]
            weighted_score = np.sum(matched_dists * weights)

            if weighted_score < best_score:
                best_score = weighted_score
                best_obj = obj_name

        results[label] = (best_obj, best_score)
    return results


def load_color_templates_from_txt(file_path):
    """
    加载模板文件，每个 object_name 后面跟若干行 color: p [r, g, b]
    返回：dict[str, List[(p, [r, g, b])]]
    """
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
                rgb = eval(parts[1])  # 安全性可改为 json.loads(re.sub)
                templates[current_name].append((percentage, np.array(rgb)))
    return templates


def plot_cluster_with_names(points_2d, labels, name_mapping):
    """
    labels: 每个点对应的聚类标签
    name_mapping: dict[int -> str]，聚类标签 -> 物体名称
    """


    # plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(unique_labels))]
    """
    for idx, k in enumerate(sorted(unique_labels)):
        class_member_mask = (labels == k)
        xy = points_2d[class_member_mask]
        if k == -1:
            color = 'k'
            label = 'Noise'
        else:
            color = colors[idx]
            label = name_mapping.get(k, f'Cluster {k}')
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=5, label=label)

    plt.legend()
    plt.axis('equal')
    plt.title("Clustered Objects with Predicted Labels")
    plt.show() 
    """



def visualize_and_fit_min_rect(points_2d, alpha=0.02):
    """
    可视化原始点、α-shape 边界、凸包、最小外接矩形
    """
    # 1️⃣ α-shape 提取凹边界
    alpha_shape = alphashape.alphashape(points_2d, alpha)
    if alpha_shape.geom_type != 'Polygon':
        print("Alpha shape is not a polygon. Try different alpha.")
        return

    alpha_coords = np.array(alpha_shape.exterior.coords)

    # 2️⃣ 凸包边界点
    hull = ConvexHull(alpha_coords)
    convex_coords = alpha_coords[hull.vertices]

    # 3️⃣ 拟合最小外接矩形
    convex_coords_cv2 = convex_coords.astype(np.float32)
    rect = cv2.minAreaRect(convex_coords_cv2)  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)  # 获取四个顶点
    box = np.vstack([box, box[0]])  # 封闭矩形

    """
    # 4️⃣ 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=3, color='gray', label="Original Points")

    plt.plot(alpha_coords[:, 0], alpha_coords[:, 1], 'b-', lw=2, label="Alpha Shape")
    closed_convex = np.vstack([convex_coords, convex_coords[0]])
    plt.plot(closed_convex[:, 0], closed_convex[:, 1], 'r--', lw=2, label="Convex Hull")
    plt.plot(box[:, 0], box[:, 1], 'g-', lw=2, label="Min Area Rectangle")

    plt.axis('equal')
    plt.legend()
    plt.title("Alpha Shape + Convex Hull + Min Rectangle")
    plt.show()
    """

    # 可选返回结果
    return rect  # center(x, y), (w, h), angle


def fit_rect_via_pca(points_2d):
    """
    返回与 cv2.minAreaRect 一致的格式：((cx, cy), (length, width), angle)
    """
    mean = np.mean(points_2d, axis=0)
    centered = points_2d - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    R = eigvecs[:, ::-1]  # 第一列是主方向

    rotated = centered @ R
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    extent = max_xy - min_xy
    center_local = (min_xy + max_xy) / 2
    center_global = center_local @ R.T + mean

    # angle: 主方向与 x 轴的夹角
    angle_rad = np.arctan2(R[1, 0], R[0, 0])
    angle_deg = np.rad2deg(angle_rad)

    length, width = extent
    if width > length:
        length, width = width, length
        angle_deg += 90  # 主轴变为次轴

    return tuple(center_global), (length, width), angle_deg


def draw_all_clusters_with_obb(cluster_dict):
    plt.figure(figsize=(10, 8))

    colors = plt.get_cmap('tab10')
    for idx, (label, info) in enumerate(cluster_dict.items()):
        pts = info["points_2d"]
        rect = info["rect"]
        name = info["name"]

        # 点云
        plt.scatter(pts[:, 0], pts[:, 1], s=5, color=colors(idx % 10), label=f"{name} ({label})")

        # 画 OBB 矩形
        box = cv2.boxPoints(rect)  # shape: (4, 2)
        box = np.vstack([box, box[0]])  # 封闭路径
        plt.plot(box[:, 0], box[:, 1], 'k-', lw=2)

        # 标注中心
        # center = rect[0]
        # plt.text(center[0], center[1], name, fontsize=9, ha='center', va='center',bbox=dict(facecolor='white', alpha=0.6, boxstyle='round'))

    plt.axis('equal')
    plt.title("Clustered Objects with OBBs")
    plt.legend()
    plt.show()


def main():
    pcd = load_and_crop_cloud("/home/wsy/merged_cloud.pcd")
    pcd = clean_pointcloud(pcd, statistical=True, radius=True, dbscan=True)
    o3d.visualization.draw_geometries([pcd], window_name='cleaned pcd')
    _, table, others = extract_table_plane(pcd)
    # o3d.visualization.draw_geometries([others], window_name='objects')

    table_points = np.asarray(table.points)
    table_colors = np.asarray(table.colors)
    object_points = np.asarray(others.points)
    object_colors = np.asarray(others.colors)

    points_filtered, colors_filtered, main_colors = merge_objects_with_table_residual(
        table_points, table_colors,
        object_points, object_colors,
        threshold=30
    )

    points_combined = np.vstack([points_filtered, object_points])
    colors_combined = np.vstack([colors_filtered, object_colors])

    points_projected = project_to_xoy(points_combined)
    pcd_projected = o3d.geometry.PointCloud()
    pcd_projected.points = o3d.utility.Vector3dVector(points_projected)
    pcd_projected.colors = o3d.utility.Vector3dVector(colors_combined)
    pcd_projected = clean_pointcloud(pcd_projected, voxel_size=0.001)
    o3d.visualization.draw_geometries(([pcd_projected]), window_name='projected pcd')

    points_2d = np.asarray(pcd_projected.points)[:, :2]
    colors_2d = np.asarray(pcd_projected.colors)
    labels = cluster_plane_objects(points_2d, eps=0.015, min_samples=20, min_cluster_points=50)
    colors = np.asarray(pcd_projected.colors)
    cluster_colors_info = extract_main_colors_by_cluster(points_2d, colors, labels, k=2)

    template_path = "color_templates.txt"
    templates = load_color_templates_from_txt(template_path)
    classified = classify_clusters_with_templates_strict(cluster_colors_info, templates)

    for label, (obj_name, score) in classified.items():
        print(f"Cluster {label}: classified as '{obj_name}' with score {score:.4f}")

    name_map = {label: name for label, (name, _) in classified.items()}
    plot_cluster_with_names(points_2d, labels, name_map)

    # 创建每个聚类簇的详细信息字典
    cluster_dict = {}
    all_points = np.asarray(pcd_projected.points)
    all_colors = np.asarray(pcd_projected.colors)

    for label in set(labels):
        if label == -1:
            continue
        mask = (labels == label)
        cluster_dict[label] = {
            "name": classified[label][0],
            "score": classified[label][1],
            "points": all_points[mask],
            "colors": all_colors[mask],
            "points_2d": points_2d[mask],
        }

    for label, info in cluster_dict.items():
        pts = info["points_2d"]

        if info["name"] == "tip rack":
            rect = visualize_and_fit_min_rect(pts, alpha=0.0001)
            height = 0.10

        elif info["name"] == "pH sensor" or info["name"] == "pipette":
            rect = fit_rect_via_pca(pts)
            height = 0.30


        else:
            rect = None
            height = None

        cluster_dict[label]["rect"] = rect
        cluster_dict[label]["height"]=height

    draw_all_clusters_with_obb(cluster_dict)


if __name__ == "__main__":
    main()
