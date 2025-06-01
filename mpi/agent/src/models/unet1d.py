import torch.nn as nn
import torch

from agent.src.layers.block import UnetAttentionBlock,BasicResBlock

class Unet1D(nn.Module):
    def __init__(self, dim, channels=1, dim_mults=(1, 2, 4)):
        super().__init__()
        self.init_conv = nn.Conv1d(channels, dim, kernel_size=7, padding=3)

        dims = [dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Down blocks
        self.downs = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(nn.ModuleList([
                BasicResBlock(dim_in),
                UnetAttentionBlock(dim_in),
                nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1)  # downsample
            ]))

        # Middle block
        mid_dim = dims[-1]
        self.mid_block = nn.Sequential(
            BasicResBlock(mid_dim),
            UnetAttentionBlock(mid_dim),
            BasicResBlock(mid_dim)
        )

        # Up blocks
        self.ups = nn.ModuleList()
        reversed_in_out = list(reversed(in_out))  # [(dim4, dim3), (dim3, dim2), (dim2, dim1), ...]

        for dim_out, dim_in in reversed_in_out:
            self.ups.append(nn.ModuleList([
                BasicResBlock(2 * dim_in),        # cat之后的输入
                UnetAttentionBlock(2 * dim_in),    # cat之后的输入
                nn.ConvTranspose1d(2 * dim_in, dim_out, 4, stride=2, padding=1)  # upsample到下一个通道数
            ]))

        self.final_res_block = BasicResBlock(dim)
        self.final_conv = nn.Conv1d(dim, channels, 1)

    def forward(self, x):
        x = self.init_conv(x)
        skips = []

        # Downsampling
        for res_block, attn_block, downsample in self.downs:
            x = res_block(x)
            x = attn_block(x)
            skips.append(x)
            x = downsample(x)
        skips.append(x)

        # Middle
        x = self.mid_block(x)

        # Upsampling
        for res_block, attn_block, upsample in self.ups:
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = res_block(x)
            x = attn_block(x)
            x = upsample(x)

        x = self.final_res_block(x)
        return self.final_conv(x)
    

class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class SimpleUnet1D(nn.Module):
    def __init__(self, dim, channels=1, dim_mults=(1, 2, 4)):
        super().__init__()
        self.init_conv = nn.Conv1d(channels, dim, kernel_size=7, padding=3)

        dims = [dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Down blocks
        self.downs = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(nn.ModuleList([
                SimpleBlock(dim_in),
                SimpleBlock(dim_in),
                nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1)  # downsample
            ]))

        # Middle block
        mid_dim = dims[-1]
        self.mid_block = nn.Sequential(
            SimpleBlock(mid_dim),
            SimpleBlock(mid_dim),
            SimpleBlock(mid_dim)
        )

        # Up blocks
        self.ups = nn.ModuleList()
        reversed_in_out = list(reversed(in_out))

        for dim_out, dim_in in reversed_in_out:
            self.ups.append(nn.ModuleList([
                SimpleBlock(2 * dim_in),
                SimpleBlock(2 * dim_in),
                nn.ConvTranspose1d(2 * dim_in, dim_out, 4, stride=2, padding=1)  # upsample
            ]))

        self.final_res_block = SimpleBlock(dim)
        self.final_conv = nn.Conv1d(dim, channels, 1)

    def forward(self, x):
        x = self.init_conv(x)
        skips = []

        # Downsampling
        for res_block, attn_block, downsample in self.downs:
            x = res_block(x)
            x = attn_block(x)
            skips.append(x)
            x = downsample(x)
        skips.append(x)

        # Middle
        x = self.mid_block(x)

        # Upsampling
        for res_block, attn_block, upsample in self.ups:
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = res_block(x)
            x = attn_block(x)
            x = upsample(x)

        x = self.final_res_block(x)
        return self.final_conv(x)