import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.voxelmorph.torch import layers
from models.backbones.layers import encoder
from transformers import AutoImageProcessor, AutoModel, Dinov2Config, AutoConfig
from einops import repeat, rearrange

from torchvision import transforms
transform_image = transforms.Compose([
    #transforms.Resize(image_size),
    #transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class dispWarp(nn.Module):

    def __init__(self, in_cs, ks=1, is_int=1):

        super(dispWarp, self).__init__()

        self.is_int = is_int

        self.disp_field_fea = nn.Sequential(
            nn.Conv3d(2*in_cs, 2*in_cs, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2*in_cs, (ks*2+1)**3, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.get_flow = nn.Conv3d((ks*2+1)**3, 3, 3, 1, 1)
        #self.up_tri = torch.nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)

    def disp_field(self, x, y):

        feas = self.disp_field_fea(torch.cat((y+x, y-x), dim=1))
        flow = self.get_flow(feas)

        return flow

    def forward(self,x,y,transformer,up_flow,integrate,upscale):

        if up_flow is not None:
            #print(x.shape, up_flow.shape)
            x = transformer(x, up_flow)

        flow = self.disp_field(x, y)
        preint_flow = flow
        #print(flow.shape)
        if self.is_int:
            flow = integrate(flow)

        if up_flow is not None:
            #flow = flow + up_flow
            flow = flow + transformer(up_flow, flow)

        #up_flow = self.up_tri(flow) * 2
        #up_flow = self.up_tri(flow) 
        up_flow = torch.nn.Upsample(scale_factor=(upscale[0],upscale[1],upscale[2]), mode='trilinear', align_corners=True)(flow)
        #print(up_flow.shape,torch.tensor((2,2,1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((up_flow.shape[0], 1, up_flow.shape[2], up_flow.shape[3], up_flow.shape[4])).shape)
        #input()
        #up_flow = up_flow * torch.tensor((2,2,1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((up_flow.shape[0], 1, up_flow.shape[2], up_flow.shape[3], up_flow.shape[4])).cuda()
        #print(upscale, torch.tensor(upscale).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((up_flow.shape[0], 1, up_flow.shape[2], up_flow.shape[3], up_flow.shape[4])))
        up_flow = up_flow * torch.tensor(upscale).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((up_flow.shape[0], 1, up_flow.shape[2], up_flow.shape[3], up_flow.shape[4])).cuda()

        return preint_flow, flow, up_flow


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024],
    ):
        super(DPTHead, self).__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.output_conv = nn.Conv2d(features*4, nclass, kernel_size=1, stride=1, padding=0)  
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        target_hw = layer_1_rn.shape[-2:]  
        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        out = self.scratch.output_conv(fused)
        return out


class regdino_mlp(nn.Module):

    def __init__(self, 
        img_size='(128,128,16)', # (128,128,16) for ACDC
        start_channel='8',
        lk_size= '3',
        cv_ks = '1',
        is_int = '1',
        #up_scale = '(2,2,1)'
    ):
        super(regdino_mlp, self).__init__()

        self.img_size = eval(img_size)
        self.start_channel = int(start_channel)
        self.lk_size = int(lk_size)
        self.cv_ks = int(cv_ks)
        self.is_int = int(is_int)

        print("img_size: {}, start_channel: {}, lk_size: {}, cv_ks: {}, is_int: {}".format(self.img_size, self.start_channel, self.lk_size, self.cv_ks, self.is_int))

        N_s = self.start_channel
        self.simple_encoder = nn.Sequential(
            encoder(257,N_s,3,1,1),
            encoder(N_s,2*N_s,3,1,1),
            encoder(2*N_s,N_s,3,1,1),
        )

        ss = self.img_size
        #self.transformers = nn.ModuleList([layers.SpatialTransformer((ss[0]//2**i,ss[1]//2**i,ss[2])) for i in range(5)])
        #self.integrates = nn.ModuleList([layers.VecInt((ss[0]//2**i,ss[1]//2**i,ss[2]), 7) for i in range(5)])
        #N_s = 768
        self.disp_warp_4 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_3 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_2 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_1 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_0 = dispWarp(N_s, self.cv_ks, self.is_int)
        #self.disp_warp = dispWarp(N_s, self.cv_ks, self.is_int)

    
    
    def forward(self, x, y, transformers, integrates, upscale, registration=False):

        
        feas = self.simple_encoder(torch.cat([x, y], 0))
        x_feas, y_feas = torch.chunk(feas, 2, dim=0)

        x_0, y_0 = x_feas, y_feas
        
        

        x_1 = F.interpolate(x_0, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_1 = F.interpolate(y_0, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)

        x_2 = F.interpolate(x_1, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_2 = F.interpolate(y_1, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)

        x_3 = F.interpolate(x_2, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_3 = F.interpolate(y_2, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)

        x_4 = F.interpolate(x_3, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_4 = F.interpolate(y_3, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        
        
        int_flow_4, pos_flow_4, up_flow_4 = self.disp_warp_4(x_4,y_4,transformers[4],None,integrates[4],upscale)
        int_flow_3, pos_flow_3, up_flow_3 = self.disp_warp_3(x_3,y_3,transformers[3],up_flow_4,integrates[3],upscale)
        
        
        int_flow_2, pos_flow_2, up_flow_2 = self.disp_warp_2(x_2,y_2,transformers[2],up_flow_3,integrates[2],upscale)
        int_flow_1, pos_flow_1, up_flow_1 = self.disp_warp_1(x_1,y_1,transformers[1],up_flow_2,integrates[1],upscale)
        int_flow_0, pos_flow_0, _ = self.disp_warp_0(x_0,y_0,transformers[0],up_flow_1,integrates[0],upscale)

        int_flows = [int_flow_0, int_flow_1, int_flow_2, int_flow_3, int_flow_4]
        pos_flows = [pos_flow_0, pos_flow_1, pos_flow_2, pos_flow_3, pos_flow_4]

        if not registration:
            return int_flows, pos_flows
        else:
            return pos_flows[0]


class regdino_mlp0(nn.Module):

    def __init__(self, 
        img_size='(128,128,16)', # (128,128,16) for ACDC
        start_channel='8',
        lk_size= '3',
        cv_ks = '1',
        is_int = '1',
        #up_scale = '(2,2,1)'
    ):
        super(regdino_mlp, self).__init__()

        self.img_size = eval(img_size)
        self.start_channel = int(start_channel)
        self.lk_size = int(lk_size)
        self.cv_ks = int(cv_ks)
        self.is_int = int(is_int)

        print("img_size: {}, start_channel: {}, lk_size: {}, cv_ks: {}, is_int: {}".format(self.img_size, self.start_channel, self.lk_size, self.cv_ks, self.is_int))

        N_s = self.start_channel
        self.simple_encoder = nn.Sequential(
            encoder(257,N_s,3,1,1),
            encoder(N_s,2*N_s,3,1,1),
            encoder(2*N_s,N_s,3,1,1),
        )

        ss = self.img_size
        #self.transformers = nn.ModuleList([layers.SpatialTransformer((ss[0]//2**i,ss[1]//2**i,ss[2])) for i in range(5)])
        #self.integrates = nn.ModuleList([layers.VecInt((ss[0]//2**i,ss[1]//2**i,ss[2]), 7) for i in range(5)])
        #N_s = 768
        self.disp_warp_4 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_3 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_2 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_1 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_0 = dispWarp(N_s, self.cv_ks, self.is_int)
        #self.disp_warp = dispWarp(N_s, self.cv_ks, self.is_int)

    
    
    def forward(self, x, y, transformers, integrates, upscale, registration=False):

        
        feas = self.simple_encoder(torch.cat([x, y], 0))
        x_feas, y_feas = torch.chunk(feas, 2, dim=0)

        x_0, y_0 = x_feas, y_feas
        
        

        x_1 = F.interpolate(x_0, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_1 = F.interpolate(y_0, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)

        x_2 = F.interpolate(x_1, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_2 = F.interpolate(y_1, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)

        x_3 = F.interpolate(x_2, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_3 = F.interpolate(y_2, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)

        x_4 = F.interpolate(x_3, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        y_4 = F.interpolate(y_3, scale_factor=(1/upscale[0], 1/upscale[1], 1/upscale[2]), mode='trilinear', align_corners=True)
        
        '''
        int_flow_4, pos_flow_4, up_flow_4 = self.disp_warp_4(x_4,y_4,transformers[4],None,integrates[4],upscale)
        int_flow_3, pos_flow_3, up_flow_3 = self.disp_warp_3(x_3,y_3,transformers[3],up_flow_4,integrates[3],upscale)
        '''
        int_flow_3, pos_flow_3, up_flow_3 = self.disp_warp_3(x_3,y_3,transformers[3],None,integrates[3],upscale)
        
        int_flow_2, pos_flow_2, up_flow_2 = self.disp_warp_2(x_2,y_2,transformers[2],up_flow_3,integrates[2],upscale)
        int_flow_1, pos_flow_1, up_flow_1 = self.disp_warp_1(x_1,y_1,transformers[1],up_flow_2,integrates[1],upscale)
        int_flow_0, pos_flow_0, _ = self.disp_warp_0(x_0,y_0,transformers[0],up_flow_1,integrates[0],upscale)

        int_flows = [int_flow_0, int_flow_1, int_flow_2, int_flow_3]
        pos_flows = [pos_flow_0, pos_flow_1, pos_flow_2, pos_flow_3]

        if not registration:
            return int_flows, pos_flows
        else:
            return pos_flows[0]