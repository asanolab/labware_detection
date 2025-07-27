import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import cv2
from sklearn.cluster import KMeans
from matplotlib.path import Path
import alphashape
from shapely.geometry import Point


def load_and_crop_cloud(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    o3d.visualization.draw_geometries([pcd],window_name='File input')
    points = np.asarray(pcd.points)
    mask = (points[:, 0] >= -0.35) & (points[:, 0] <= 0.06) & \
           (points[:, 1] >= 0.1) & (points[:, 1] <= 0.55) & \
           (points[:, 2] >= -0.02) & (points[:, 2] <= 0.4)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    print(f"Z range: {z_min:3f}-{z_max:3f}")
    cropped = pcd.select_by_index(np.where(mask)[0])
    # down_pcd = cropped.voxel_down_sample(voxel_size=0.001)
    print("Cropped successfully")
    # print(f"points: {np.asarray(cropped.points).shape[0]}")
    o3d.visualization.draw_geometries([cropped], window_name='Cropped')
    return cropped


def extract_table_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    table = pcd.select_by_index(inliers)
    others = pcd.select_by_index(inliers, invert=True)
    print("Table extracted successfully")
    o3d.visualization.draw_geometries([table], window_name='Table')
    return plane_model, table, others


# ---------- 点云投影为 RGB 图片 ----------
def point_cloud_to_rgb_image(points, colors, grid_size=0.005):
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    H = int((y_max - y_min) / grid_size) + 1
    W = int((x_max - x_min) / grid_size) + 1

    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # 默认白色填补

    point_mask = np.zeros((H, W), dtype=bool)

    for (x, y), (r, g, b) in zip(points[:, :2], colors * 255):
        xi = int((x - x_min) / grid_size)
        yi = int((y - y_min) / grid_size)
        img[yi, xi] = [r, g, b]
        point_mask[yi, xi] = True

    #img = np.rot90(img, 2)
    #point_mask = np.rot90(point_mask, 2)

    meta = (x_min, y_min, grid_size)
    plt.imshow(img)
    plt.title('original table rgb')
    plt.show()
    return img, point_mask, meta


def extract_table_mask(img, method='convex', alpha=1.0):
    """
    从 RGB 投影图提取桌面有效区域（排除桌面外白色区域）
    支持 Convex Hull 或 Alpha Shape（更适合桌面边缘内凹情况）

    Args:
        img (np.ndarray): RGB 投影图
        method (str): 'convex' | 'alpha'
        alpha (float): alpha shape 参数（仅当 method='alpha' 时生效）

    Returns:
        table_mask (np.ndarray): 桌面内部 bool mask (True 为桌面内部)
    """
    mask_not_white = ~np.all(img == [255, 255, 255], axis=-1)
    coords = np.column_stack(np.where(mask_not_white))

    if coords.shape[0] < 3:
        raise RuntimeError("桌面区域数据不足，无法拟合轮廓。")

    H, W = img.shape[:2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    points = np.vstack([yy.ravel(), xx.ravel()]).T

    if method == 'convex':
        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices]
        path = Path(hull_coords)
        table_mask = path.contains_points(points).reshape(H, W)

    elif method == 'alpha':
        coords_xy = np.flip(coords, axis=1)  # (y, x) -> (x, y)
        alpha_shape = alphashape.alphashape(coords_xy, alpha)
        table_mask = np.array([alpha_shape.contains(Point(x, y)) for x, y in points])
        table_mask = table_mask.reshape(H, W)

    else:
        raise ValueError(f"Unknown method: {method}")

    return table_mask

def separate_outside_and_holes(img):
    white_mask = np.all(img == [255, 255, 255], axis=-1).astype(np.uint8) * 255
    h, w = white_mask.shape
    mask_flood = white_mask.copy()

    # flood fill 整个四条边的白色像素
    for x in range(w):
        if mask_flood[0, x] == 255:
            cv2.floodFill(mask_flood, None, (x, 0), 128)
        if mask_flood[h-1, x] == 255:
            cv2.floodFill(mask_flood, None, (x, h-1), 128)

    for y in range(h):
        if mask_flood[y, 0] == 255:
            cv2.floodFill(mask_flood, None, (0, y), 128)
        if mask_flood[y, w-1] == 255:
            cv2.floodFill(mask_flood, None, (w-1, y), 128)

    mask_outside = (mask_flood == 128)
    mask_hole = (mask_flood == 255)
    return mask_outside, mask_hole

def extract_main_color(img):
    mask_valid = ~np.all(img == [255, 255, 255], axis=-1)  # 不是白色
    mask_valid &= ~np.all(img == [0, 0, 0], axis=-1)  # 不是黑色

    pixels = img[mask_valid].reshape(-1, 3)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(pixels)
    centers = kmeans.cluster_centers_  # [2, 3] 2个主色 RGB

    # 哪个中心更接近灰白，哪个更像桌面
    desk_color = centers[np.argmin(np.linalg.norm(centers - 255, axis=1))]

    print(f"Detected desk color (approx): {desk_color}")
    return desk_color


def remove_main_color(img, main_color, threshold=30):
    dist = np.linalg.norm(img.astype(float) - main_color, axis=-1)
    img[dist < threshold] = [255, 255, 255]
    return img


def extract_objects_by_dbscan(img, eps=5, min_samples=10):
    """
    用 DBSCAN 从 img 中提取物体簇（非白非黑颜色点，按距离成簇）
    Args:
        img (np.ndarray): RGB 图片
        eps (float): DBSCAN 距离阈值（像素单位）
        min_samples (int): 每簇最小点数量，过滤孤立噪声

    Returns:
        object_masks (list[np.ndarray]): 每个物体的 bool mask
    """
    # 提取有效像素位置
    mask = ~np.all(img == [255, 255, 255], axis=-1)
    # mask &= ~np.all(img == [0, 0, 0], axis=-1)

    plt.imshow(mask)
    plt.title('object mask')
    plt.show()

    coords = np.column_stack(np.where(mask))  # (H, W) -> (y, x)

    if len(coords) == 0:
        return []

    # DBSCAN 聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    object_masks = []
    H, W = img.shape[:2]
    for label in set(labels):
        if label == -1:
            continue  # 噪声
        mask_single = np.zeros((H, W), dtype=bool)
        cluster_coords = coords[labels == label]
        mask_single[cluster_coords[:, 0], cluster_coords[:, 1]] = True
        object_masks.append(mask_single)

    return object_masks


# ---------- 主流程 ----------
def main():
    pcd = load_and_crop_cloud("/home/wsy/merged_cloud_test.pcd")
    _, table, _ = extract_table_plane(pcd)
    points = np.asarray(table.points)
    colors = np.asarray(table.colors)

    img, point_mask, meta = point_cloud_to_rgb_image(points, colors)
    H, W, _ = img.shape

    mask_outside, mask_hole = separate_outside_and_holes(img)

    # 桌面内部洞补黑色
    img[mask_hole] = [0, 0, 0]

    # 桌面外部保持白色
    img[mask_outside] = [255, 255, 255]
    '''
    table_mask = extract_table_mask(img, method='alpha', alpha=1.0)

    # 桌面内部没点的区域 → 补黑色
    img[table_mask & ~point_mask] = [0, 0, 0]

    # 桌面外部仍然白色（已经是白了）
    img[~table_mask] = [255, 255, 255]
    '''


    plt.imshow(img)
    plt.title('table rgb')
    plt.show()

    main_color = extract_main_color(img)

    img_removed = remove_main_color(img.copy(), main_color, threshold=50)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img)
    axs[0].set_title('orginal table RGB')
    axs[1].imshow(img_removed)
    axs[1].set_title('removed table RGB')
    plt.show()

    object_masks = extract_objects_by_dbscan(img_removed, eps=5, min_samples=20)

    print(len(object_masks))

    fig, axs = plt.subplots(1, len(object_masks) + 1, figsize=(4 * len(object_masks), 4))
    if not object_masks:
        print("⚠ 没有找到任何有效的物体区域。")
        return
    if len(object_masks) == 1:
        axs = [axs]
    axs[0].imshow(img)
    axs[0].set_title('original rgb')
    for i, mask in enumerate(object_masks):
        axs[i + 1].imshow(mask, cmap='gray')
        axs[i + 1].set_title(f'Object {i + 1}')
    plt.show()


if __name__ == "__main__":
    main()
