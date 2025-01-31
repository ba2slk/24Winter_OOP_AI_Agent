import torch

# DINOv2
dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
dinov2_vitl14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
dinov2_vitg14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')

# DINOv2 with registers
dinov2_vits14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
dinov2_vitb14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
dinov2_vitl14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
dinov2_vitg14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc')