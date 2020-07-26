from torch import nn
import torch

upsample = nn.ConvTranspose2d(48, 48, 3,1,0, bias=False)
in_ = torch.randn(8, 48, 32, 32)
out_ = upsample(in_)
print(out_.size())

