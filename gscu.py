import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.funcation import make_coord,to_pixel_samples,generate_meshgrid,fetching_features_from_tensor,extract_patch

class Encoder(nn.Module):
    def __init__(self, channels=4, n_feats=48, n_classes=81):
        """
        channel:ms的通道
        n_feats:特征通道
        n_classed:高斯核的类数
        t：通道特异性向量V的通道
        """
        super(Encoder, self).__init__()
        self.n_feats = n_feats
        self.pan_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        ####通道特异性V的分支
        self.v_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, n_feats, kernel_size=1)
        )
        #SE通道注意力
        self.v_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_feats // 4, n_feats, kernel_size=1),
            nn.Sigmoid()
        )
        ####特征计算分支
        #多尺度下采样
        self.downsample_pan = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.downsample_fusion_mid = nn.Sequential(
            nn.Conv2d(channels * 2, n_feats, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.downsample_fusion_low = nn.Sequential(
            nn.Conv2d(channels * 2, n_feats, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, n_feats // 4, kernel_size=3, padding=ks, dilation=ks),
                nn.ReLU()
            ) for ks in [1, 2, 3]
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*2, n_feats // 4, kernel_size=1),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 2, n_feats, kernel_size=1),
            nn.ReLU()
        )
        self.lateral = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.feat_fusion = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        # Logits（基于 feat+coord）
        self.coord_conv = nn.Conv2d(2, channels*4, kernel_size=1)
        self.logits_branch = nn.Conv2d(n_feats+channels*4, n_classes, kernel_size=1)

    def forward(self, ms, pan):
        up_ms = nn.functional.interpolate(ms, scale_factor=4, mode='bilinear')  # (B, 4, 128, 128)
        pan_feat = self.pan_conv(pan)  # (B, 4, 128, 128)
        input = torch.cat([up_ms, pan_feat], dim=1)  # (B, 8, 128, 128)

        # V
        V = self.v_fusion(input)  # (B, 48, 128, 128)
        channel_weights = self.v_attention(V)  # (B, 48, 1, 1)
        V = V * channel_weights # (B, 48, 128, 128)

        # 跨模态特征
        # 多尺度下采样
        down_pan = self.downsample_pan(pan)  # (B, 4, 32, 32)
        down_input = torch.cat([ms, down_pan], dim=1)  # (B, 8, 32, 32)
        mid_feat = self.downsample_fusion_mid(down_input)  # (B, 48, 32, 32)
        low_feat = self.downsample_fusion_low(down_input)  # (B, 48, 8, 8)
        up_mid = F.interpolate(mid_feat, size=(128, 128), mode='bilinear')  # (B, 48, 128, 128)
        up_low = F.interpolate(low_feat, size=(128, 128), mode='bilinear')  # (B, 48, 128, 128)

        feats = [branch(input) for branch in self.dilated]  # 3x (B, 12, 128, 128)
        gp = self.global_pool(input) #(B, 12, 1, 1)
        gp = gp.expand(-1, -1, 128, 128)#(B, 12, 128, 128)
        feat = torch.cat(feats, dim=1)  # (B, 36, 128, 128)
        feat = torch.cat([feat,gp], dim=1)# (B, 48, 128, 128)
        residual = self.residual(input)  # (B, 48, 128, 128)
        feat = feat + residual # (B, 48, 128, 128)
        feat = self.lateral(feat) # (B, 48, 128, 128)
        feat = feat + up_mid + up_low  # FPN 风格融合(B, 48, 128, 128)
        feat = self.feat_fusion(feat) # (B, 48, 128, 128)

        # Logits
        coords = make_coord(pan.shape[-2:], flatten=False).to(pan.device)  # (128, 128, 2)
        coords = coords.permute(2, 0, 1).unsqueeze(0).expand(pan.shape[0], -1, -1, -1)  # (B, 2, 128, 128)
        coord_feat = self.coord_conv(coords)  # (B, 12, 128, 128)
        logits_input = torch.cat([feat, coord_feat], dim=1)  # (B, 60, 128, 128)
        logits = self.logits_branch(logits_input)# (B, 100, 128, 128)
        B, Class, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(B * H * W, Class)
        if self.training:
            logits = F.gumbel_softmax(logits, tau=1, hard=False)
        if not self.training:
            logits = F.gumbel_softmax(logits, tau=1, hard=True)
        logits = logits.view(B, H, W, Class).permute(0, 3, 1, 2).contiguous()
        return V, feat, logits

class GaussianSplatter(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, kernel_size=5, unfold_row=8, unfold_column=8, num_points=81, c1=6, channels=4, n_feats=48):
        """
        Initialize the 2D Gaussian Splatter module.
        Args:
            kernel_size (int): The size of the kernel to convert rasterization.
            unfold_row (int): The number of points in the row dimension of the Gaussian grid.
            unfold_column (int): The number of points in the column dimension of the Gaussian grid.
            num_points: The number of the kernel.
            c1: 2DGS's feat.
            channels: LRMS channels.
            n_feats: The channels of features from Encoder.
        """
        super(GaussianSplatter, self).__init__()
        self.encoder = Encoder(channels=channels, n_feats=n_feats, n_classes=num_points)
        self.feat, self.logits = None, None  # 后续会有编码器生成这个
        # Key parameter in 2D Gaussian Splatter参数
        self.kernel_size = kernel_size
        self.row = unfold_row
        self.column = unfold_column
        self.num_points = num_points  # 高斯核的数量
        self.c1 = c1
        self.channels = channels
        self.n_feats = n_feats
        self.c2 = n_feats - c1
        # V2 通道注意力（增强 feat2）
        self.v2_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c2, self.c2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 通道调整
        self.fine_adjust = nn.Sequential(
            nn.Conv2d(n_feats, channels, kernel_size=3, padding=1)
        )
        self.residual = nn.Conv2d(n_feats, channels, kernel_size=1)
        # Initialize Trainable Parameters
        grid_size = int(np.sqrt(num_points))  # 转换为整数
        sigma_x, sigma_y = torch.meshgrid(torch.linspace(0.2, 3.0, grid_size), torch.linspace(0.2, 3.0, grid_size))
        self.sigma_x = sigma_x.reshape(-1)
        self.sigma_y = sigma_y.reshape(-1)
        # 根据高斯核的数量设置初始化全1，然后通过sigmoid函数激活（透明度）
        self.opacity = torch.sigmoid(torch.ones(self.num_points, 1, requires_grad=True))
        # 根据高斯核的数量设置初始化全0，限制范围在(-0,0)，相关系数
        self.rho = torch.clamp(torch.zeros(self.num_points, 1, requires_grad=True), min=-1, max=1)
        # 通过parameter定义为可训练参数
        self.sigma_x = nn.Parameter(self.sigma_x)  # Standard deviation in x-axis
        self.sigma_y = nn.Parameter(self.sigma_y)  # Standard deviation in y-axis
        self.opacity = nn.Parameter(self.opacity)  # Transparency of feature, shape=[num_points, 0]
        self.rho = nn.Parameter(self.rho)

    def weighted_gaussian_parameters(self, logits):
        """
        Computes weighted Gaussian parameters based on logits and the Gaussian kernel parameters (sigma_x, sigma_y, opacity).
        The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial
        location across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension.
        Args:
            logits (torch.Tensor): Logits tensor of shape [batch, class, height, width].
        Returns:
            tuple: A tuple containing the weighted Gaussian parameters:
                - weighted_sigma_x (torch.Tensor): Tensor of shape [height * width] representing the weighted x-axis standard deviations.
                - weighted_sigma_y (torch.Tensor): Tensor of shape [height * width] representing the weighted y-axis standard deviations.
                - weighted_opacity (torch.Tensor): Tensor of shape [height * width] representing the weighted opacities.
        Description:
            This function computes weighted Gaussian parameters based on the input tensor, logits, and the provided Gaussian kernel parameters (sigma_x, sigma_y, and opacity). The logits tensor is used as a weight to compute a weighted sum of the Gaussian kernel parameters for each spatial location (height and width) across the batch dimension. The resulting weighted parameters are then averaged across the batch dimension, yielding tensors of shape [height * width] for the weighted sigma_x, sigma_y, and opacity.
        """
        batch_size, num_classes, height, width = logits.size()
        logits = logits.permute(0, 2, 3, 1)  # Reshape logits to [batch, height, width, class]

        # Compute weighted sum of Gaussian parameters across class dimension（按类别体现在sum(dim=-0)
        weighted_sigma_x = (logits * self.sigma_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_sigma_y = (logits * self.sigma_y.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_opacity = (logits * self.opacity[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        weighted_rho = (logits * self.rho[:, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # Reshape and average across batch dimension
        weighted_sigma_x = weighted_sigma_x.reshape(batch_size, -1).mean(dim=0)
        weighted_sigma_y = weighted_sigma_y.reshape(batch_size, -1).mean(dim=0)
        weighted_opacity = weighted_opacity.reshape(batch_size, -1).mean(dim=0)
        weighted_rho = weighted_rho.reshape(batch_size, -1).mean(dim=0)

        return weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho

    def gen_feat(self, ms, pan):
        """Generate feature and logits by encoder."""
        self.ms = ms
        self.V, self.feat, self.logits = self.encoder.forward(ms, pan)
        return self.feat, self.logits

    def query_rgb(self):
        # 1. Get LR feature and logits
        feat, lr_feat, logits = self.feat[:, :self.c1, :, :], self.feat[:, self.c1:, :,:], self.logits  # Channel decoupling
        # feat(B,c1,128,128) lr_feat(B,n_feats-c1,128,128) #logits(B,n_classed,128,128)
        feat_size, feat_device = feat.shape, feat.device
        V1, V2 = self.V[:, :self.c1, :, :], self.V[:, self.c1:, :, :]

        # 2. Calculate the high-resolution image size
        # scale = 4
        # hr_h = round(feat.shape[-2] * scale)  # shape: [batch size]
        # hr_w = round(feat.shape[-1] * scale)

        # 3. Unfold the feature / logits to many small patches to avoid extreme GPU memory consumption
        num_kernels_row = math.ceil(feat_size[-2] / self.row)
        num_kernels_column = math.ceil(feat_size[-1] / self.column)
        upsampled_size = (num_kernels_row * self.row, num_kernels_column * self.column)
        upsampled_inp = F.interpolate(feat, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_logits = F.interpolate(logits, size=upsampled_size, mode='bicubic', align_corners=False)
        upsampled_V1 = F.interpolate(V1, size=upsampled_size, mode='bicubic', align_corners=False)
        unfold = nn.Unfold(kernel_size=(self.row, self.column), stride=(self.row, self.column))
        unfolded_feature = unfold(upsampled_inp).contiguous()
        unfolded_logits = unfold(upsampled_logits).contiguous()
        unfolded_V1 = unfold(upsampled_V1).contiguous()
        # Unfolded_feature dimension becomes [Batch, C*K*K, L], where L is the number of columns after unfolding
        L = unfolded_feature.shape[-1]
        unfold_feat = unfolded_feature.transpose(1, 2).contiguous().reshape(feat_size[0] * L, feat_size[1], self.row,self.column)
        unfold_logits = unfolded_logits.transpose(1, 2).contiguous().reshape(logits.shape[0] * L, logits.shape[1],self.row, self.column)
        unfold_V1 = unfolded_V1.transpose(1, 2).contiguous().reshape(V1.shape[0] * L, V1.shape[1], self.row,self.column)

        # 4. Generate colors_(features) and coords_norm
        coords_ = generate_meshgrid(unfold_feat.shape[-2], unfold_feat.shape[-1])
        num_LR_points = unfold_feat.shape[-2] * unfold_feat.shape[-1]
        colors_, coords_norm = fetching_features_from_tensor(unfold_feat, coords_)

        # 5. Rasterization: Generating grid
        # 5.0. Spread Gaussian points over the whole feature map
        batch_size, channel, _, _ = unfold_feat.shape
        weighted_sigma_x, weighted_sigma_y, weighted_opacity, weighted_rho = self.weighted_gaussian_parameters(
            unfold_logits)
        sigma_x = weighted_sigma_x.view(num_LR_points, 1, 1)
        sigma_y = weighted_sigma_y.view(num_LR_points, 1, 1)
        rho = weighted_rho.view(num_LR_points, 1, 1)

        # 5.2. Gaussian expression
        covariance = torch.stack(
            [torch.stack([sigma_x ** 2 + 1e-5, rho * sigma_x * sigma_y], dim=-1),
             torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2 + 1e-5], dim=-1)], dim=-2
        )  # when correlation rou is set to zero, covariance will always be positive semi-definite
        inv_covariance = torch.inverse(covariance).to(feat_device)

        # 5.3. Choosing a broad range for the distribution [-5,5] to avoid any clipping
        start = torch.tensor([-5.0], device=feat_device).view(-1, 1)
        end = torch.tensor([5.0], device=feat_device).view(-1, 1)
        base_linspace = torch.linspace(0, 1, steps=self.kernel_size, device=feat_device)
        ax_batch = start + (end - start) * base_linspace
        # Expanding dims for broadcasting
        ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, self.kernel_size, -1)

        # 5.4. Creating a batch-wise meshgrid using broadcasting
        xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
        xy = torch.stack([xx, yy], dim=-1)
        z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
        kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=feat_device) *
                                 torch.sqrt(torch.det(covariance)).to(feat_device).view(num_LR_points, 1, 1))
        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
        kernel_normalized = kernel / kernel_max_2  # (num_LR_points,kernel_size,kernel_size)
        unfold_V1_reshaped = unfold_V1.view(batch_size, channel, num_LR_points)  # (B, 8, 64)
        unfold_V1_weights = torch.softmax(unfold_V1_reshaped, dim=1)  # (B, 8, 64)
        kernel_channel = kernel_normalized.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, channel, 1,1)  # (B, 64, 8, kernel_size, kernel_size)
        # V1添加通道注意力，每个通道在每个像素处的高斯核不相同
        kernel_color = kernel_channel * unfold_V1_weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 8, kernel_size, kernel_size)
        kernel_color = kernel_color.view(batch_size * num_LR_points, channel, self.kernel_size,self.kernel_size)  # (B*64, 8, kernel_size, kernel_size)

        # 5.5. Adding padding to make kernel size equal to the image size
        pad_h = round(unfold_feat.shape[-2]) - self.kernel_size
        pad_w = round(unfold_feat.shape[-1]) - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller or equal to the image size.")
        padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
        kernel_color_padded = torch.nn.functional.pad(kernel_color, padding, "constant", 0)

        # 5.6. Create a batch of 2D affine matrices
        _, c, h, w = kernel_color_padded.shape  # num_LR_points*batch_size, channel, hr_h, hr_w
        theta = torch.zeros(batch_size, num_LR_points, 2, 3, dtype=torch.float32, device=feat_device)
        theta[:, :, 0, 0] = 1.0
        theta[:, :, 1, 1] = 1.0
        theta[:, :, :, 2] = coords_norm
        grid = F.affine_grid(theta.view(-1, 2, 3), size=[batch_size * num_LR_points, c, h, w],align_corners=True).contiguous()
        kernel_color_padded_translated = F.grid_sample(kernel_color_padded.contiguous(), grid.contiguous(),align_corners=True)
        kernel_color_padded_translated = kernel_color_padded_translated.view(batch_size, num_LR_points, c, h, w)

        # 6. Apply Gaussian splatting
        # colors_.shape = [batch, num_LR_points, channel], colors.shape = [batch, num_LR_points, channel]
        colors = colors_ * weighted_opacity.to(feat_device).unsqueeze(-1).expand(batch_size, -1, -1)
        color_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1)
        final_image_layers = color_values_reshaped * kernel_color_padded_translated
        final_image = final_image_layers.sum(dim=1)
        final_image = torch.clamp(final_image, 0, 1)

        # 7. Fold the input back to the original size
        # Calculate the number of kernels needed to cover each dimension.
        kernel_h, kernel_w = round(self.row), round(self.column)
        fold = nn.Fold(output_size=(kernel_h * num_kernels_row, kernel_w * num_kernels_column),
                       kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        final_image = final_image.reshape(feat_size[0], L, feat_size[1] * kernel_h * kernel_w).transpose(1, 2)
        final_image = fold(final_image)
        final_image = F.interpolate(final_image, size=(128, 128), mode='bicubic', align_corners=False)

        # 8. Decoder and bicubic
        # V2 注意力增强 feat2
        v2_weights = self.v2_attention(V2)
        lr_feat = lr_feat * v2_weights  # (B,c2,128,128)
        # 拼接final_image 和 lr_feat
        feat = torch.cat([final_image, lr_feat], dim=1)  # (B, n_feats, 128, 128)
        out = self.fine_adjust(feat)  # (B, 4, 128, 128)
        out = out + self.residual(feat)  # (B, 4, 128, 128)
        return out

    def forward(self, lrms, pan):
        self.gen_feat(lrms, pan)
        return self.query_rgb()


if __name__ == '__main__':
    # A simple example of implementing class GaussianSplatter
    model = GaussianSplatter()
    lrms = torch.rand(1, 4, 32, 32)
    pan = torch.rand(1, 1, 128, 128)
    pred = model.forward(lrms, pan)
    print(pred.shape)
    # Encoder = Encoder(4,48,100)
    # V,feat,Logits = Encoder.forward(lrms,pan)
    # print(V.shape,feat.shape,Logits.shape)