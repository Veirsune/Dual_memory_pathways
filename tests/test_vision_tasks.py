import unittest
import numpy as np
import torch
from spikingjelly.clock_driven import functional
from vision_tasks.models import SequentialClassifier, GestureClassifier
from vision_tasks.prepare_gesture import integrate

class VisualBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_event_binning_preserves_count_and_polarity(self):
        events=np.array([[0,0,0,10],[3,3,0,10],[127,127,1,19],[4,8,1,15]])
        frames=integrate(events,10)
        self.assertEqual(int(frames.sum()),4)
        self.assertEqual(int(frames[0,0,0,0]),2)
        self.assertEqual(int(frames[9,1,31,31]),1)
        self.assertEqual(int(frames[5,1,2,1]),1)
        # Coarser temporal integration preserves every event.
        self.assertEqual(int(frames.reshape(2,5,2,32,32).sum(1).sum()),4)

    def test_original_sequential_logits_and_batch_one(self):
        torch.manual_seed(9)
        model=SequentialClassifier(memory=4).eval()
        x=torch.rand(2,784,1)
        with torch.no_grad():
            wrapped=model(x)
            functional.reset_net(model)
            original=model.core(x)
            one=model(x[:1])
        torch.testing.assert_close(wrapped,original,rtol=0,atol=0)
        self.assertEqual(tuple(one.shape),(1,10))

    def test_gesture_reset_and_training_gradient(self):
        torch.manual_seed(12)
        model=GestureClassifier(steps=8,memory=4)
        # Exercise the memory branch explicitly; a random ReLU drive may be
        # inactive on a tiny 8-step synthetic clip. This is test-only setup.
        with torch.no_grad():model.fc1.W_u.bias.fill_(0.2)
        x=torch.rand(1,8,2,32,32)*30
        model.eval()
        with torch.no_grad():
            a=model(x);b=model(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(tuple(a.shape),(1,11))
        model.train();loss=torch.nn.functional.cross_entropy(model(x),torch.tensor([3]));loss.backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertGreater(float(model.conv1.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.fc1.weight.weight.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
