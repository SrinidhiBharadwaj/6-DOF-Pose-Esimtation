import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from matplotlib import pyplot as plt
import argparse
from loader import PointData
from unet import UNet
import torch.nn as nn
from tqdm import tqdm
from utils import *
from loss_fn import CrossEntropyLoss2d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from unet_multi import unet_model
from segnet import SegNet
from evaluate import Evaluate


def evaluate_model(model, val_loader, device, epoch):
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            labels = model(data["image"].to(device))
            image = data["image"][0].cpu().detach().numpy().transpose(1, 2, 0)  
            mask_single = torch.softmax(labels, dim=1)
            pred_mask = torch.argmax(mask_single, dim=1)
            pred_mask_np = pred_mask[0].cpu().detach().numpy()
            labels_np = data["label"][0].cpu().detach().numpy()
            print(np.unique(pred_mask_np), np.unique(labels_np))
            draw_mask(image, pred_mask_np, epoch)
            return

def get_val_loss(model, val_loader, loss):
    with torch.no_grad():
        running_loss = 0
        for i, data in enumerate(val_loader):
            pred_labels = model(data["image"].to(device))
            running_loss += loss(pred_labels, data["label"].to(device)).item()
    return running_loss/len(val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", "--segment", default=True)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--lr", default=1e-4)
    
    args = parser.parse_args()
    NUM_TRAIN = 100
    segment = args.s
    epochs = int(args.epoch)
    lr = args.lr
    train_path = "training_data_filtered/training_data"
    dataset = PointData(train_path, train=True, segment=segment)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)#sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    val_set = PointData(train_path, train=False, segment=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=True)

    channel_list = [3, 32, 64, 128, 256]
    # model = unet_model(82).to("cuda")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SegNet(3, 82).to(device)
    model = UNet(channel_list, 82).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    loss = CrossEntropyLoss2d()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    # evaluate = Evaluate()
    writer = SummaryWriter(log_dir="tb_logs")

    prev_loss = np.inf
    for epoch in range(epochs):
        running_loss = 0
        for _, data in (enumerate(tqdm(loader))):
            image = data["image"].to(device)
            labels = data["label"].to(device)
            pred = model(image)
            epoch_loss = loss(pred, labels)
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()        
            running_loss+=epoch_loss.item()

        val_loss = get_val_loss(model, val_loader, loss)
        train_loss = running_loss/len(loader)

        scheduler.step(val_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        if epoch % 10 == 0:
            evaluate_model(model, val_loader, device, epoch)
        if train_loss < prev_loss:
            torch.save(model, f"saved_models_unet_half_size/model_{epoch}.pt")

        print(f"Epoch:{epoch} - Train Loss:{train_loss}, Val Loss: {val_loss}")


