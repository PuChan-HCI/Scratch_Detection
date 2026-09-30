import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

def block(a, b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.ReLU(inplace=True))

class ScratchNet(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        r=resnet18(weights=None)
        if pretrained: r.load_state_dict(torch.load(pretrained,weights_only=True))
        self.stem=nn.Sequential(r.conv1,r.bn1,r.relu)
        self.pool=r.maxpool
        self.e1,self.e2,self.e3,self.e4=r.layer1,r.layer2,r.layer3,r.layer4
        self.d3=block(512+256,128); self.d2=block(128+128,64)
        self.d1=block(64+64,32); self.d0=block(32+64,24)
        self.full=block(24+3,16); self.head=nn.Conv2d(16,1,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def forward(self,x):
        z=(x-self.mean)/self.std
        s=self.stem(z); a=self.e1(self.pool(s)); b=self.e2(a); c=self.e3(b); d=self.e4(c)
        def up(x,y): return torch.cat([F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False),y],1)
        d=self.d3(up(d,c)); d=self.d2(up(d,b)); d=self.d1(up(d,a)); d=self.d0(up(d,s))
        return self.head(self.full(up(d,z)))
