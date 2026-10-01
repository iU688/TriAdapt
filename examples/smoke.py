"""One synthetic optimization step; not a scientific experiment."""
import torch
from torch import nn
from triadapt import TriAdapt, make_optimizer, train_step


class InterfaceFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(2,512)
        self.head=nn.Linear(512,3)

    def forward(self,x):
        return self.head(self.projection(x.mean(1)))[:,None]


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model=TriAdapt(InterfaceFixture(),stage='stage2')
    x=torch.randn(2,1,9,17,2)
    batch={'detector':x,'target':torch.randn(2,1,17,3)*100,
           'gt2d':x.clone(),'gt_confidence':torch.ones(2,1,9,17)}
    stats=train_step(model,batch,make_optimizer(model),torch.Generator().manual_seed(100042))
    print('Synthetic interface check passed:',stats)
    print('Stage-2 trainable parameters:',sum(p.numel() for p in model.adaptation_parameters()))


if __name__=='__main__': main()
