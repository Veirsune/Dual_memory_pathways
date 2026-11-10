import unittest,torch
from vision_tasks.models import GestureClassifier
from vision_tasks.cpu_lif import enable_cpu_scan

class CommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_delayed_commit_zero_permission_and_gradients(self):
        torch.manual_seed(812)
        model=GestureClassifier(steps=12,memory=5,family='commit');enable_cpu_scan(model)
        c=model.fc1;c.capture=True
        # Explicitly activate presynaptic spikes and both memory reads in this
        # tiny synthetic test; production initialization is unchanged.
        with torch.no_grad():
            c.W_u.bias.fill_(.3);c.read_m.weight.fill_(.02);model.conv2.bias.fill_(2.)
        x=torch.rand(2,12,2,32,32)*40;x[:,6:9]=0
        out=model(x);tr=c.trace
        self.assertEqual(tuple(tr['u'].shape),(2,12,5))
        torch.testing.assert_close(tr['m'][:,0],torch.zeros_like(tr['m'][:,0]),rtol=0,atol=0)
        expected=(1-tr['gate'][:,1:])*tr['m'][:,:-1]+tr['gate'][:,1:]*tr['p'][:,:-1]
        torch.testing.assert_close(tr['m'][:,1:],expected,rtol=1e-5,atol=1e-7)
        self.assertEqual(float(tr['gate'][:,6:9].abs().sum()),0.)
        torch.nn.functional.cross_entropy(out,torch.tensor([1,7])).backward()
        for param in (c.W_u.weight,c.read_m.weight,c.gate.weight,c.permit.net[-1].bias):
            self.assertTrue(torch.isfinite(param.grad).all());self.assertGreater(float(param.grad.abs().sum()),0.)
    def test_permission_causality(self):
        torch.manual_seed(813);model=GestureClassifier(steps=12,memory=5,family='commit').eval()
        x=torch.rand(1,12,2,32,32);y=x.clone();y[:,7:]*=10
        with torch.no_grad():
            a=model.fc1.permit(x);b=model.fc1.permit(y)
        torch.testing.assert_close(a[:,:7],b[:,:7],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
