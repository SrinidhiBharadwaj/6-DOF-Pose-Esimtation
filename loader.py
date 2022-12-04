import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2 
import pickle
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import RandomHorizontalFlip

class PointData(Dataset):
    def __init__(self, path, segment=False, train=True, transforms=None):
        super(PointData, self).__init__()
        self.path = path
        self.segment = False
        txt = "train.txt" if train else "val.txt"
        self.path_text = os.path.join(path, "splits/v2", txt)
        with open(self.path_text, "r") as f:
            self.data = f.read().splitlines()
        self.filepath = os.path.join(self.path, "v2.2")
        self.transforms = transforms
        self.num_classes = 80

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_num = self.data[index]
        png_name = "_kinect.png"
        image_path = os.path.join(self.filepath, file_num+"_color"+png_name) 
        depth_path = os.path.join(self.filepath, file_num+"_depth"+png_name) 
        label_path = os.path.join(self.filepath, file_num+"_label"+png_name) 
        meta_path = os.path.join(self.filepath, file_num+"_meta.pkl")
        image = cv2.imread(image_path)[:, :, ::-1]
        size = image.shape[:2]
        # image = cv2.resize(image, (int(size[1]/2), int(size[0]/2)))
        depth = cv2.imread(depth_path, 0)
        label = cv2.imread(label_path, 0)
        # label = cv2.resize(label, (int(size[1]/2), int(size[0]/2)))
        meta = self.load_pickle(meta_path)
        poses = meta["poses_world"]
        image = image/255.
        
        # label = F.one_hot(torch.as_tensor(label).long())
        return {"image":torch.Tensor(image).permute(2, 0, 1), "label":(torch.Tensor(label).type(torch.long))}
        
    def process_labels(self, label):
        ret_label = np.zeros((self.num_classes, *label.shape))
        for i in range(self.num_classes):
            indices = np.where(label == i)
            ret_label[i][indices] = 1
        return ret_label


    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    path = "training_data_filtered/training_data"
    dataset = PointData(path, train=True)
    dataset.__getitem__(0)
