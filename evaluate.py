import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from unet_multi import unet_model
import os
from tqdm import tqdm
from segnet import SegNet

class Evaluate():
    def __init__(self, path, model, device="cpu"):
        self.dir = path
        self.path_text = os.path.join(self.dir, "test.txt")
        with open(self.path_text, "r") as f:
            self.data = f.read().splitlines()
        # self.file_path = os.path.join(("/").join(self.dir.split("/")[:-1]), "v2.2")
        self.file_path = os.path.join(self.dir, "v2.2")
        self.model = model
        self.device = device

    def read_image(self, file):
        image  = cv2.imread(file)[:, :, ::-1]
        self.im_shape = image.shape[:2]
        # image = cv2.resize(image, (320, 160))
        image = image.transpose(2, 0, 1)
        image = torch.Tensor(image.copy())
        image = image.unsqueeze(0)
        return image

    def get_and_save_mask(self, file, id):
        image = self.read_image(file)
        labels = self.model(image.to(self.device))

        mask_single = torch.softmax(labels, dim=1)
        pred_mask = torch.argmax(mask_single, dim=1).squeeze()
        pred_mask_np = pred_mask.cpu().detach().numpy().astype(np.uint8)
        
        # pred_mask_np = cv2.resize(pred_mask_np, self.im_shape[::-1], cv2.INTER_NEAREST)
        save_name = os.path.join(("/").join(file.split("/")[:-1]), id + "_label_kinect_unet.png")
        # print(save_name)
        cv2.imwrite(save_name, pred_mask_np)
        return pred_mask
    
    def get_masks(self):
        for idx, id in enumerate(tqdm(self.data)):
            filename = os.path.join(self.file_path, id+"_color_kinect.png")
            self.get_and_save_mask(filename, id)

        # print(self.data, len(self.data))


if __name__ == "__main__":
    test_path = "testing_data_final_filtered/testing_data/"
    # val_path = "training_data_filtered/training_data/splits"
    # model_path = "saved_models_multi/model_31_bestval.pt"
    model_path = "saved_models_unet_a_new_hope/best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(model_path).to(device)
    eval = Evaluate(test_path, model, device)
    eval.get_masks()