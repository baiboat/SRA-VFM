import torch
from torch import nn
from .FastSAM.fastsam import FastSAM
from torch.nn import functional as F
from typing import Dict, List
from utils.misc import initialize_weights
from timm.models.vision_transformer import _cfg, Mlp
import math

from mamba_ssm import Mamba
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out
        
def get_mamba_layer(
    spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims==2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims==3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer
    
    
    
class MambaBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 64,
        out_channels: int = 64,
        norm: tuple = ("GROUP", {"num_groups": 4}),
        kernel_size: int = 3,
        act: str = ("RELU", {"inplace": True}),
        stride: int = 1
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.

            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.

            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)      
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=out_channels, stride=stride
        )
        #self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #                              nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
  
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x) 

        
        return x



class ResMambaBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 64,
        out_channels: int = 64,
        norm:  tuple = ("GROUP", {"num_groups": 8}),
        kernel_size: int = 3,
        act: str = ("RELU", {"inplace": True}),
        stride: int = 1
    ) -> None:
        """

        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.

            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.

            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, stride=stride
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x
        
def get_dwconv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels, 
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)

class ResUpBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int=2,
        in_channels: int=64,
        norm: tuple = ("GROUP", {"num_groups": 8}),
        kernel_size: int = 3,
        act: tuple = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x




class LinearBlock(nn.Module):

    def __init__(self, dim=192, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_tokens=192):
        super().__init__()

        # First stage
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(dim)

        # Second stage
        self.mlp2 = Mlp(in_features=num_tokens, hidden_features=int(
            num_tokens * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(num_tokens)

        # Dropout (or a variant)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp1(self.norm1(x)))
        x = x.transpose(-2, -1)
        x = x + self.drop_path(self.mlp2(self.norm2(x)))
        x = x.transpose(-2, -1)
        return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()        
        A = self.SA(x)
        return A
        
class MScale_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(MScale_Attention, self).__init__()
        self.SA1 = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            
        )
        self.SA2 = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 3, padding=1),
            
        )
        self.SA3 = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 5, padding=2),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 5, padding=2),
           
        )
    
        self.sigmoid = nn.Sigmoid()  
    def forward(self, x):
        b, c, h, w = x.size()        
        A1 = self.SA1(x)
        A2 = self.SA2(x)
        A3 = self.SA3(x)
        A = self.sigmoid(A1 + A2 + A3)
        return A


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(CECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x, y):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(y, output)
        return output
        
class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        #output = self.act1(output)
        output = torch.multiply(x, output)
        return output

class Encoder(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, outplanes):
        super(Encoder, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(inplanes), nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mamba0 = MambaBlock(in_channels = planes, out_channels=outplanes)
        self.conv1 = nn.Sequential(nn.Conv2d(planes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(outplanes), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(outplanes+planes, outplanes, kernel_size=5, stride=1, padding=2, bias=False),
                                       nn.BatchNorm2d(outplanes), nn.ReLU())
        self.eca = CECANet(in_channels=outplanes)                            
    def forward(self, low, high):
        low = self.conv0(low)
        #low = self.pool(low)
        out = torch.cat([low, high],dim=1)
        out0 = self.conv1(out)
        out1 = self.mamba0(out)
        out = self.eca(out1, out0) 

        return out
        
class CANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(CANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x, y):
        outputx = self.fgp(x)
        outputx = outputx.squeeze(-1).transpose(-1, -2)
        outputx = self.con1(outputx).transpose(-1, -2).unsqueeze(-1)
        outputx = self.act1(outputx)
        
        outputy = self.fgp(y)
        outputy = outputy.squeeze(-1).transpose(-1, -2)
        outputy = self.con1(outputy).transpose(-1, -2).unsqueeze(-1)
        outputy = self.act1(outputy)
        
        outputx = torch.multiply(x, outputy) 
        outputy = torch.multiply(y, outputx)
        return outputx + outputy
        
class EnCross(nn.Module):
    expansion = 1
    def __init__(self, inplanes):
        super(EnCross, self).__init__()
        self.conv = get_dwconv_layer(
            spatial_dims=3, in_channels=inplanes, out_channels=inplanes, kernel_size=3
        )
          
        self.project = nn.Sequential(nn.Conv2d(inplanes*2, inplanes*2, kernel_size=3, stride=1, groups=2, padding=1, bias=False),
                                       nn.BatchNorm2d(inplanes*2), nn.ReLU())    
        self.projectA = nn.Sequential(nn.Conv2d(inplanes*4, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(inplanes), nn.ReLU())  
        self.projectB = nn.Sequential(nn.Conv2d(inplanes*4, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(inplanes), nn.ReLU())
        
        self.mambaA = MambaBlock(in_channels = inplanes, out_channels=inplanes)
        self.mambaB = MambaBlock(in_channels = inplanes, out_channels=inplanes)
        self.eca = CANet(in_channels=inplanes*2) 
        self.ecaB = ECANet(in_channels=inplanes) 
        self.spaceAtt = MScale_Attention(in_channels=inplanes*2, out_channels=inplanes*2)
        
    def forward(self, A, B):
        N,C,H,W = A.shape
        #Global = torch.cat([A, B], dim=1)
        Global = torch.empty((N,C*2,H,W)).to(A)
        Global[:, 0::2, :, :] = A
        Global[:, 1::2, :, :] = B
        Global = self.project(Global)
        Global_CA = self.eca(Global)
        Global_SA = self.spaceAtt(Global)
        Global = Global_CA + Global_SA
        
        A = self.projectA(Global) 
        B = self.projectB(Global) 
        A = self.mambaA(A)
        B = self.mambaB(B)
        return A, B
        
        
        
class DeCross(nn.Module):
    expansion = 1
    def __init__(self, inplanes):
        super(DeCross, self).__init__()
        self.conv = get_dwconv_layer(
            spatial_dims=3, in_channels=inplanes, out_channels=inplanes, kernel_size=3
        )
          
        self.project = nn.Sequential(nn.Conv2d(inplanes*2, inplanes, kernel_size=3, stride=1, groups=2, padding=1, bias=False),
                                       nn.BatchNorm2d(inplanes), nn.ReLU())    
        self.projectA = nn.Sequential(nn.Conv2d(inplanes*2, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(inplanes), nn.ReLU())  
        self.projectB = nn.Sequential(nn.Conv2d(inplanes*2, inplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(inplanes), nn.ReLU())
        
        self.mambaA = MambaBlock(in_channels = inplanes, out_channels=inplanes)
        self.mambaB = MambaBlock(in_channels = inplanes, out_channels=inplanes)
        self.ca = CANet(in_channels=inplanes) 
        self.eca = ECANet(in_channels=inplanes) 
        self.spaceAtt = MScale_Attention(in_channels=inplanes, out_channels=inplanes)
        self.spaceAttB = MScale_Attention(in_channels=inplanes, out_channels=inplanes)
        
    def forward(self, A, B):
        A, B = self.ca(A, B)
        N,C,H,W = A.shape
        #Global = torch.cat([A, B], dim=1)
        Global = torch.empty((N,C*2,H,W)).to(A)
        Global[:, 0::2, :, :] = A
        Global[:, 1::2, :, :] = B
        Global_AB = self.project(Global)
        Global_CA = self.eca(Global_AB)
        Global_SA = self.spaceAtt(Global_AB)
        Global_CS = Global_CA + Global_SA
        
        A = Global_CS * A
        B = Global_CS * B
        
        #A = self.mambaA(A)
        #B = self.mambaB(B)
        return A, B

class SAM_CD(nn.Module):
    def __init__(
        self,
        num_embed=8,
        model_name: str='FastSAM-x.pt',
        device: str='cuda',
        conf: float=0.4,
        iou: float=0.9,
        imgsz: int=1024,
        retina_masks: bool=True,
        ):
        super(SAM_CD, self).__init__()
        self.model = FastSAM(model_name)
        self.device = device
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_feats = None        
         
        self.Adapter32 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter16 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter8 = nn.Sequential(nn.Conv2d(320, 80, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(80), nn.ReLU())
        self.Adapter4 = nn.Sequential(nn.Conv2d(160, 40, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(40), nn.ReLU())
        
        self.Enc2 = Encoder(40, 120, 80)
        self.Enc1 = Encoder(80, 240, 160)
        self.Enc0 = Encoder(160, 320, 160)
        
        self.EnCross2 = EnCross(80)
        self.EnCross1 = EnCross(160)
        self.EnCross0 = EnCross(160)
        
        
                            
        self.Dec2 = _DecoderBlock(160, 160, 80)
        self.Dec1 = _DecoderBlock(80, 80, 40)  
        self.Dec0 = _DecoderBlock(40, 40, 64)
        
        self.DeCross2 = DeCross(80)
        self.DeCross1 = DeCross(40)
        self.DeCross0 = DeCross(64)


        
        self.SA = Space_Attention(16, 16, 4)
        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1) 
        self.resCD0 = self._make_layer(ResBlock, 128, 128, 6, stride=1)
        self.resCD1 = self._make_layer(ResBlock, 80, 80, 4, stride=1)
        self.up = nn.ConvTranspose2d(80, 80, kernel_size=2, stride=2)
        self.alignc = nn.Sequential(nn.Conv2d(80, 128, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        #self.resEG0 = self._make_layer(ResBlock, 128, 128, 4, stride=1)
        #self.resEG1 = self._make_layer(ResBlock, 128, 128, 2, stride=1)
        
        self.headCD = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.headEG = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        #self.headEG = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.CA0 = CANet(128)
        #self.CA1 = CANet(128)
        self.segmenterCD = nn.Conv2d(16, 1, kernel_size=1)
        self.segmenterEG = nn.Conv2d(16, 1, kernel_size=1)
                                        
        for param in self.model.model.parameters():
            param.requires_grad = False
        initialize_weights(self.Adapter32, self.Adapter16, self.Adapter8, self.Adapter4, self.Dec2, self.Dec1, self.Dec0,\
                           self.segmenter, self.headCD, self.segmenterCD)

    def run_encoder(self, image):
        self.image = image
        feats = self.model(
            self.image,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
            )
        return feats

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
    
        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        featsB = self.run_encoder(x2)
        
  
        featA_s4 = self.Adapter4(featsA[3].clone())#40*128*128
        featA_s8 = self.Adapter8(featsA[0].clone())#80*64*64
        featA_s16 = self.Adapter16(featsA[1].clone())#160*32*32
        featA_s32 = self.Adapter32(featsA[2].clone())#160*16*16
        
        featB_s4 = self.Adapter4(featsB[3].clone())
        featB_s8 = self.Adapter8(featsB[0].clone())
        featB_s16 = self.Adapter16(featsB[1].clone())
        featB_s32 = self.Adapter32(featsB[2].clone())   
       
       
        featA_s8 = self.Enc2(featA_s4, featA_s8)
        featA_s16 = self.Enc1(featA_s8, featA_s16)
        featA_s32 = self.Enc0(featA_s16, featA_s32)
        
        decA_2 = self.Dec2(featA_s32, featA_s16)
        decA_1 = self.Dec1(decA_2, featA_s8)
        decA_0 = self.Dec0(decA_1, featA_s4)
        outA = self.segmenter(decA_0)
        
        featB_s4 = self.Adapter4(featsB[3].clone())
        featB_s8 = self.Adapter8(featsB[0].clone())
        featB_s16 = self.Adapter16(featsB[1].clone())
        featB_s32 = self.Adapter32(featsB[2].clone())   
        
        featB_s8 = self.Enc2(featB_s4, featB_s8)
        featB_s16 = self.Enc1(featB_s8, featB_s16)
        featB_s32 = self.Enc0(featB_s16, featB_s32)
        
        decB_2 = self.Dec2(featB_s32, featB_s16)
        decB_1 = self.Dec1(decB_2, featB_s8)
        decB_0 = self.Dec0(decB_1, featB_s4)
        outB = self.segmenter(decB_0)
              

        #decA_0, decB_0 = self.DeCross0(decA_0, decB_0)

        
        A = self.SA(torch.cat([outA, outB], dim=1))  
        featC = torch.cat([decA_0, decB_0], 1)
        featE = torch.cat([decA_1, decB_1], 1)
        featCD = self.resCD0(featC)
        featEG = self.resCD1(featE)
        featEG = self.up(featEG)
        featEG = self.alignc(featEG)
        #featEG = self.CA0(featCD, featEG)
        
        #featEG = self.resCD1(featEG)
        #featCD = self.CA0(featCD, featEG)
        #featCD = self.resCD1(featCD)
        
        featEG = self.headEG(torch.cat([featEG, featCD], dim=1)) 
        #featEG = self.headEG(featEG) 
        featCD = self.headCD(featCD)
        
        edge = self.segmenterEG(featEG)
        outCD = self.segmenterCD(featCD)
        
        outCD = outCD + edge
        
        
        return F.interpolate(outCD, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(edge, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outA, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outB, input_shape, mode="bilinear", align_corners=True)
