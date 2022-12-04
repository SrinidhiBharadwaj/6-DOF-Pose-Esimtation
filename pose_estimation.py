import numpy as np
import utils
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d
import pickle
import os
from tqdm import tqdm
import json
from collections import defaultdict
import time

def draw_bbox(rgb, poses_world, meta, name):
    try:
        boxed_image = np.array(rgb)
        box_sizes = np.array([meta['extents'][idx] * meta['scales'][idx] for idx in meta['object_ids']])
        object_ids = meta["object_ids"]
        for i, id in enumerate(object_ids):
            if id in poses_world:
                utils.draw_projected_box3d(
                    boxed_image, poses_world[id][:3,3], box_sizes[i], poses_world[id][:3, :3], meta['extrinsic'], meta['intrinsic'],
                    thickness=2)

        plt.imshow(((boxed_image * 255).astype(np.uint8)))
        plt.show()
        plt.savefig(f"bboxes/{name}_bbox.png")
    except Exception as e:
        print(f"Failed to plot for image - {e}")

def get_pose_new(rgb, depth, label, meta, source, pred, gt=False):
    '''
    Core algorithm 
    Compares the object pointclouds in the input test image with the training dataset (source)
    rgb, depth, label: Paths to rgb, depth and label images
    meta: Path to meta information, containing objects IDs in the scene
    source: Dictionary of source point clouds in canonical frame along with their pose information
    pred: Return dictionary to dump data into JSON file
    gt: To test on validation data and to visualize the output
    '''
    name = rgb.split("/")[-1].split("_")[0]
    rgb_i, depth_i, label_i = read_images(rgb, depth, label)
    meta_i = load_pickle(meta)
    pointcloud_t = get_pcd(depth_i, meta_i)
    object_ids, object_names = meta_i["object_ids"], meta_i["object_names"]
    object_e, object_i = meta_i["extrinsic"], meta_i["intrinsic"]
    pred[name] = {"poses_world": [None] * 79,}
    poses_world = {}

    for j, id in enumerate(object_ids):
        label_obj = np.array(np.where(label_i == id)).T
        if id not in np.unique(label_i):
            pred[name]["poses_world"][id] = np.eye(4).tolist()
            continue
        #PCD of the object in test
        pcd = pointcloud_t[label_obj[:, 0], label_obj[:, 1], :]
        pcd = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
        pcd = (np.linalg.inv((object_e)) @ pcd.T).T   
        pcd = pcd[:, :3]   
        # _, ind = get_o3d_pcd(pcd).remove_statistical_outlier(nb_neighbors=20,
        #                                             std_ratio=1)
        pcd_o3d = get_o3d_pcd(pcd)#.select_by_index(ind)
        final_pose, rmse = 0, np.inf  

        for i in tqdm(range(0, len(source[id]), 2)):
            pointcloud = source[id][i]
            init_pose = pointcloud[1]
            points = pointcloud[0]       
            if len(points) < len(pcd) / 2:
                continue        
            source_down_sampled = get_o3d_pcd(points)
            
            trans_init = init_pose
            T = open3d.pipelines.registration.registration_icp(
                source_down_sampled, pcd_o3d, 7, trans_init,
                open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
            
            if T.inlier_rmse < rmse:
                final_pose = np.copy(T.transformation)
                rmse = T.inlier_rmse

        pred[name]["poses_world"][id] = final_pose.tolist()
        poses_world[id] = final_pose

        if gt:
            gt_T = meta_i["poses_world"][id]
            pose = final_pose
            rre = np.rad2deg(compute_rre(pose[:3, :3], gt_T[:3, :3]))
            rte = compute_rte(pose[:3, 3], gt_T[:3, 3])

            # visualize_2_pcds(pcd_pose[:, :3], np.array(source_down_sampled.points))   
            print(f"rre and rte for {object_names[j]} is {rre}, {rte}")

    draw_bbox(rgb_i, poses_world, meta_i, name)      

    return pred


# Helper functions
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

def read_images(rgb, depth, label):
    '''
    Function to read images
    '''
    rgb_i = np.array(Image.open(rgb)) / 255   # convert 0-255 to 0-1
    depth_i = np.array(Image.open(depth)) / 1000   # convert from mm to m
    label_i = np.array(Image.open(label))
    return rgb_i, depth_i, label_i

def create_source_pcd_list_dict(path):
    '''
    Create training data dictionary to be used for ICP
    '''
    with open(os.path.join(path, "splits/v2", "train.txt"), 'r') as f:
        prefix = [os.path.join(path, "v2.2", line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    
    pcd_dict = defaultdict(list)

    for i, files in tqdm(enumerate(rgb)):
        _, depth_i, label_i = read_images(rgb[i], depth[i], label[i])
        meta_i = load_pickle(meta[i])
        object_ids, object_names = meta_i["object_ids"], meta_i["object_names"]
        object_e, object_i = meta_i["extrinsic"], meta_i["intrinsic"]
        poses_i = meta_i["poses_world"]
        pointcloud = get_pcd(depth_i, meta_i)

        for _, id in enumerate(object_ids):
            label_obj = np.array(np.where(label_i == id)).T
            pcd = pointcloud[label_obj[:, 0], label_obj[:, 1], :]
            pcd = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
            #Transform to canonical frame
            pcd = (np.linalg.inv(poses_i[id]) @ np.linalg.inv((object_e)) @ pcd.T).T
            pcd_dict[id].append([pcd[:, :3], poses_i[id]])


    with open("source_dict_2.pkl", "wb") as f:
        print("Writing to source file")
        pickle.dump(pcd_dict, f)

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

def get_split_files(test_data_path, version, split):
    ''''
    Function to get the names of training and testing files
    '''
    with open(os.path.join(test_data_path, split), 'r') as f:
        prefix = [os.path.join(test_data_path, version, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect_unet.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta

if __name__ == "__main__":
    test_data_path = "testing_data_final_filtered/testing_data/"
    train_path = "training_data_filtered/training_data"
    val_path = "training_data_filtered/training_data/"
    model_path = "models/"
    test_results_path = "test_data_results.json"
    rgb, depth, label, meta = get_split_files(test_data_path, "v2.2", "test.txt")
    val_rgb, val_depth, val_label, val_meta = get_split_files(val_path, "v2.2", "splits/v2/val.txt")

    val = False #Set to true for validation data
    #Uncomment below line to create point cloud dictionaries for source objects
    # source_pcds = create_source_pcd_dict(train_path)
    # source_pcds = create_source_pcd_list_dict(train_path)
    
    print("Reading reference poses")
    with open("source_dict_2_updated.pkl", "rb") as f:
        source = pickle.load(f)

    print("Starting to calculate object pose")
    pred = {}
    starttime = time.time()
    for i in (range(len(rgb))):
        if not val:
            print(f"Registration of image {i+1} in progress..")
            pred = get_pose_new(rgb[i], depth[i], label[i], meta[i], source, pred)
        else:
            pred = get_pose_new(val_rgb[i], val_depth[i], val_label[i], val_meta[i], source, pred, gt=True)
    #Write to a JSON file
    with open(test_results_path, "w") as f:
        json.dump(pred, f, indent=4)
    print("Time elapsed: ", time.time() - starttime)
    print("ICP completed for the test set!")