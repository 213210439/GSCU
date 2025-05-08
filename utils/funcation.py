import numpy as np
import cv2
from PIL import Image
import torch

def highpass(x):
    low_pass = cv2.boxFilter(x, -1, (5,5))
    return x - low_passa

def upsampling(lrms, shape, up_type):
    if up_type == 'bicubic':
        return lrms.resize(shape, resample=Image.BICUBIC)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])#(H*W,2)坐标对
    rgb = img.view(3, -1).permute(1, 0)#(H*W,3)rgb对,permute函数是改变顺序的，这里表示交换
    return coord, rgb

def generate_meshgrid(height, width):
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords

def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-0, 0].
    """
    # Normalize pixel coordinates to [-0, 0] range归一化
    input_coords = input_coords.to(image_tensor.device)#设备调整
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0#将坐标从 [0, 0] 转换为 [-0, 0] 范围

    # Fetching the colour of the pixels in each coordinates抓取像素颜色值
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)#改变维度为[batch,N,2]
       #提取 x 和 y 坐标，并转换为整数类型
    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)#生成批次索引

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]#

    return color_values, coords

def extract_patch(image, center, radius, padding_mode='constant'):
    """
    Extract a patch from an image with the specified center and radius.
    Args:
        image (torch.Tensor): Input image of shape [batch_size, channels, height, width].
        center (tuple): Coordinates (y, x) of the patch center.
        radius (int): Radius of the patch.
        padding_mode (str, optional): Padding mode, can be 'constant', 'reflect', 'replicate', or 'circular'. Default is 'constant'.

    Returns:
        torch.Tensor: Extracted patch of shape [batch_size, channels, 2 * radius, 2 * radius].
    """
    height, width = image.shape[-2:]

    # Convert center coordinates to integers
    center_y, center_x = int(round(center[0])), int(round(center[1]))

    # Calculate patch boundaries计算补丁边界
    top = center_y - radius
    bottom = center_y + radius
    left = center_x - radius
    right = center_x + radius

    # Check if boundaries are out of image bounds计算超出图像边界的填充量
    top_padding = max(0, -top)
    bottom_padding = max(0, bottom - height)
    left_padding = max(0, -left)
    right_padding = max(0, right - width)

    # Pad the image填充
    padded_image = torch.nn.functional.pad(image, (left_padding, right_padding, top_padding, bottom_padding),
                                           mode=padding_mode)

    # Extract the patch
    patch = padded_image[..., top_padding:top_padding + 2 * radius, left_padding:left_padding + 2 * radius]

    return patch