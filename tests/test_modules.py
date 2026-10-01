import unittest
import torch
from torch import nn
from triadapt import (TriAdapt, FeatureAdapter, TemporalGraph2DCorrector,
                      KinematicGraphRefiner, make_optimizer, train_step)


class TestLifter(nn.Module):
    """Synthetic interface fixture, not an implementation of a pose backbone."""
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(2,512)
        self.head=nn.Linear(512,3)

    def forward(self, tracks):
        return self.head(self.projection(tracks[:,:, :, :].mean(1)))[:,None]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_parameter_counts(self):
        modules=[TemporalGraph2DCorrector(),FeatureAdapter(),KinematicGraphRefiner()]
        self.assertEqual([sum(p.numel() for p in m.parameters()) for m in modules],
                         [914,67553,17347])

    def test_zero_initialization(self):
        x=torch.randn(2,1,9,17,2)
        out,delta=TemporalGraph2DCorrector()(x)
        torch.testing.assert_close(out,x,rtol=0,atol=0)
        self.assertEqual(delta.abs().sum().item(),0)
        f=torch.randn(2,1,17,512)
        torch.testing.assert_close(FeatureAdapter()(f)[0],f,rtol=0,atol=0)
        pose=torch.randn(2,1,17,3)
        torch.testing.assert_close(KinematicGraphRefiner()(pose)[0],pose-pose[:,:,0:1])

    def test_freezing_and_gradient_through_backbone(self):
        m=TriAdapt(TestLifter(),stage='stage2')
        self.assertFalse(any(p.requires_grad for p in m.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.fam.parameters()))
        self.assertEqual(sum(p.numel() for p in m.adaptation_parameters()),18261)
        m.train()
        self.assertFalse(m.backbone.training)
        self.assertFalse(m.fam.training)
        y=m(torch.randn(2,1,9,17,2))
        y.square().sum().backward()
        self.assertGreater(m.tcm.output_projection.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in m.fam.parameters()))
        self.assertFalse(m.backbone.head._forward_pre_hooks)

    def test_arms_and_budget(self):
        for arm,n in [('fam_tcm',914),('fam_kgr',17347),('triadapt',18261)]:
            m=TriAdapt(TestLifter(),arm=arm,stage='stage2')
            self.assertEqual(sum(p.numel() for p in m.adaptation_parameters()),n)
            self.assertEqual(m(torch.randn(2,2,9,17,2)).shape,(2,2,17,3))
        for width in [11,22,32,53,74]:
            m=TriAdapt(TestLifter(),fam_width=width)
            self.assertEqual(sum(p.numel() for p in m.adaptation_parameters()),19798+2063*width)

    def test_save_restore(self):
        m=TriAdapt(TestLifter())
        n=TriAdapt(TestLifter())
        n.backbone.load_state_dict(m.backbone.state_dict())
        n.load_adapter_state(m.adapter_state())
        x=torch.randn(2,1,9,17,2)
        torch.testing.assert_close(m(x),n(x),rtol=0,atol=0)

    def test_training_steps_preserve_frozen_tensors(self):
        x=torch.randn(2,1,9,17,2)
        batch={'detector':x,'target':torch.randn(2,1,17,3)*100,
               'gt2d':x.clone(),'gt_confidence':torch.ones(2,1,9,17)}
        for stage,arm in [('stage1','fam'),('joint','triadapt'),
                          ('stage2','triadapt'),('stage2','fam_tcm'),('stage2','fam_kgr')]:
            m=TriAdapt(TestLifter(),arm=arm,stage=stage)
            frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
            value=train_step(m,batch,make_optimizer(m),torch.Generator().manual_seed(7))
            self.assertTrue(torch.isfinite(torch.tensor(value['loss'])))
            for n,p in m.named_parameters():
                if n in frozen: torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
