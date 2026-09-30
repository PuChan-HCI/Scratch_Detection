"""Lightweight pretrained coarse segmentation plus native-resolution refinement."""
import torch
from torch import nn
import torch.nn.functional as F

class OriginalBackbone(nn.Module):
    """Exact trainable convolutional portion of the supplied one_ai_model.onnx."""
    def __init__(self):
        super().__init__()
        specs={'input_conv_1':(1,16,3,1),
               'down_conv_1':(16,32,3,2),'down_conv_2':(32,32,3,1),'down_conv_3':(16,32,1,2),
               'down_conv_4':(32,32,3,1),'down_conv_5':(32,32,3,1),
               'down_conv_6':(32,64,3,2),'down_conv_7':(64,64,3,1),'down_conv_8':(32,64,1,2),
               'down_conv_9':(64,64,3,1),'down_conv_10':(64,64,3,1),
               'up_conv_1':(64,64,3,1),'up_conv_2':(64,64,3,1),'up_conv_3':(64,64,1,1),
               'up_conv_4':(96,32,3,1),'up_conv_5':(32,32,3,1),'up_conv_6':(96,32,1,1),
               'output_1_conv_1':(32,16,1,1),'output_1_conv_2':(16,16,3,1),'output_1_binary_conv_1':(16,1,3,1)}
        self.layers=nn.ModuleDict({name:nn.Conv2d(a,b,k,stride=s,padding=k//2 if s==1 else 0) for name,(a,b,k,s) in specs.items()})
    def load_original_onnx(self,path):
        import onnx
        from onnx import numpy_helper
        graph=onnx.load(str(path)); arrays={a.name:numpy_helper.to_array(a).copy() for a in graph.graph.initializer}
        with torch.no_grad():
            for node in graph.graph.node:
                if node.op_type=='Conv':
                    name=node.name.split('/')[-2]; layer=self.layers[name]
                    layer.weight.copy_(torch.from_numpy(arrays[node.input[1]])); layer.bias.copy_(torch.from_numpy(arrays[node.input[2]]))
    def conv(self,name,x):
        layer=self.layers[name]
        if layer.stride==(2,2) and layer.kernel_size==(3,3): x=F.pad(x,(0,1,0,1))
        return layer(x)
    def forward(self,rgb):
        x=rgb[:,0:1]  # The supplied original ONNX selects the first (red) channel.
        x=F.max_pool2d(F.relu(self.conv('input_conv_1',x)),2)
        c=self.conv
        x=F.relu(c('down_conv_2',F.relu(c('down_conv_1',x)))+c('down_conv_3',x))
        a=F.relu(c('down_conv_5',F.relu(c('down_conv_4',x)))+x)
        x=F.relu(c('down_conv_7',F.relu(c('down_conv_6',a)))+c('down_conv_8',a))
        x=F.relu(c('down_conv_10',F.relu(c('down_conv_9',x)))+x)
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        x=F.relu(c('up_conv_2',F.relu(c('up_conv_1',x)))+c('up_conv_3',x))
        x=F.interpolate(torch.cat([x,a],1),scale_factor=2,mode='nearest')
        x=F.relu(c('up_conv_5',F.relu(c('up_conv_4',x)))+c('up_conv_6',x))
        return c('output_1_binary_conv_1',F.relu(c('output_1_conv_2',F.relu(c('output_1_conv_1',x)))))

class DetailRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(5,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,8,3,padding=1),nn.ReLU())
        self.head=nn.Conv2d(8,1,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,rgb,prior):
        gray=rgb.mean(1,keepdim=True)
        contrast=gray-F.avg_pool2d(gray,5,stride=1,padding=2)
        features=torch.cat([rgb*2-1,prior.sigmoid(),contrast*4],1)
        correction=self.head(self.body(features))
        return prior+correction

class ScratchLite(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=OriginalBackbone(); self.refiner=DetailRefiner()
    def prior(self,rgb):
        coarse=F.interpolate(rgb,size=(256,505),mode='bilinear',align_corners=False)
        logits=self.backbone(coarse)
        return F.interpolate(logits,size=rgb.shape[-2:],mode='bilinear',align_corners=False).clamp(-8,8)
    def forward(self,rgb):
        return self.refiner(rgb,self.prior(rgb))
