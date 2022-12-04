
import torch
import torch.nn as nn
import torchvision as tv #Used only for cropping the skip connections
import torch.nn.functional as F

class conv_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_relu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class encoder(nn.Module):
    def __init__(self, c_list):
        super(encoder, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.blocks = nn.ModuleList([conv_relu(c_list[i], c_list[i+1]) for i in range(len(c_list)-1)])

    def forward(self, x):
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return skips

class decoder(nn.Module):
    def __init__(self, c_list):
        super(decoder, self).__init__()
        self.channels = c_list
        self.upsampler = nn.ModuleList([nn.ConvTranspose2d(c_list[i], c_list[i+1], 2, 2) for i in range(len(c_list) - 1)])
        self.blocks = nn.ModuleList([conv_relu(c_list[i], c_list[i+1]) for i in range(len(c_list) - 1)])
    
    def forward(self, x, skips):
        for i in range(len(self.channels) - 1):
            enc_skip = skips[i]
            x = self.upsampler[i](x)
            x = torch.cat([x, enc_skip], dim=1)
            x = self.blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self, c_list, num_classes):
        super().__init__()
        self.enc_blocks = encoder(c_list)
        self.dec_blocks = decoder(c_list[::-1][:-1])
        self.last_layer = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        filters = self.enc_blocks(x)
        x = self.dec_blocks(filters[-1], filters[::-1][1:])
        x = self.last_layer(x)
        #Interpolation function is to retain the same size
        return x#F.interpolate(self.last_layer(x), (160, 320))

if __name__ == "__main__":
    input_size = (1, 3, 360, 640)
    input_tensor = torch.rand(input_size)
    channel_list = [3, 64, 128, 256, 512]

    enc = encoder(channel_list)
    output = enc(input_tensor)
    x = output[-1]
    dec = decoder(channel_list[::-1][:-1])
    output = dec(x, output[::-1][1:])
    #print(output.size())

    model = UNet(channel_list, 1)
    print(model(input_tensor).size())