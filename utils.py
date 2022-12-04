import numpy as np
from matplotlib import pyplot as plt
import pickle
import open3d
import cv2
from transforms3d.quaternions import quat2mat

VERTEX_COLORS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

def draw_mask(image, mask, k=0):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()
    plt.savefig(f"test_images/segmented_image_{k}.png")
    
def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(
    image, center, size, rotation, extrinsic, intrinsic, color=(0, 1, 0), thickness=1
):
    """Draw a projected 3D bounding box on the image.

    Args:
        image (np.ndarray): [H, W, 3] array.
        center: [3]
        size: [3]
        rotation (np.ndarray): [3, 3]
        extrinsic (np.ndarray): [4, 4]
        intrinsic (np.ndarray): [3, 3]
        color: [3]
        thickness (int): thickness of lines
    Returns:
        np.ndarray: updated image.
    """
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)

    for (i, j) in edges:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )

    for i, (u, v) in enumerate(uv):
        cv2.circle(image, (u, v), radius=1, color=VERTEX_COLORS[i], thickness=1)
    return image


def compute_rre(R_est: np.ndarray, R_gt: np.ndarray):
    """Compute the relative rotation error (geodesic distance of rotation)."""
    assert R_est.shape == (3, 3), 'R_est: expected shape (3, 3), received shape {}.'.format(R_est.shape)
    assert R_gt.shape == (3, 3), 'R_gt: expected shape (3, 3), received shape {}.'.format(R_gt.shape)
    # relative rotation error (RRE)
    rre = np.arccos(np.clip(0.5 * (np.trace(R_est.T @ R_gt) - 1), -1.0, 1.0))
    return rre

def compute_rte(t_est: np.ndarray, t_gt: np.ndarray):
    assert t_est.shape == (3,), 't_est: expected shape (3,), received shape {}.'.format(t_est.shape)
    assert t_gt.shape == (3,), 't_gt: expected shape (3,), received shape {}.'.format(t_gt.shape)
    # relative translation error (RTE)
    rte = np.linalg.norm(t_est - t_gt)
    return rte

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_pcd(depth, meta):
    '''
    Returns point cloud of the input depth image in camera frame
    '''
    intrinsic = meta['intrinsic']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    return np.array(points_viewer)

def get_o3d_pcd(pcd):
    '''
    Get point cloud data in Open3D format
    '''
    points = open3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    pcd = open3d.geometry.PointCloud()
    pcd.points = points
    return pcd

def visualize_2_pcds(pcd1, pcd2):
    '''
    Visualize 2 point clouds
    '''
    points = open3d.utility.Vector3dVector(pcd1.reshape([-1, 3]))
    pcd1 = open3d.geometry.PointCloud()
    pcd1.points = points
    points = open3d.utility.Vector3dVector(pcd2.reshape([-1, 3]))
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = points
    open3d.visualization.draw_geometries([pcd1, pcd2])

def visualize(pcd, rgb):
    '''
    Visualize 1 Point Cloud
    '''
    points = open3d.utility.Vector3dVector(pcd.reshape([-1, 3]))
    pcd = open3d.geometry.PointCloud()
    pcd.points = points
    open3d.visualization.draw_geometries([pcd])