import copy,unittest
import torch
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from vision_tasks.cpu_lif import CpuLIF,enable_cpu_scan
from vision_tasks.models import GestureClassifier

class CpuScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_outputs_and_both_output_gradients(self):
        for dtype in (torch.float32,torch.float64):
            torch.manual_seed(702)
            x=(torch.randn(127,3,7,dtype=dtype)+.8).requires_grad_()
            y=x.detach().clone().requires_grad_()
            a=MultiStepLIFNode(tau=2.,detach_reset=True)
            b=CpuLIF(tau=2.,detach_reset=True)
            s=a(x);t=b(y)
            torch.testing.assert_close(s,t,rtol=0,atol=0)
            torch.testing.assert_close(a.v_seq,b.v_seq,rtol=0,atol=0)
            ws=torch.randn_like(s);wv=torch.randn_like(s)
            (s*ws+a.v_seq*wv).sum().backward();(t*ws+b.v_seq*wv).sum().backward()
            torch.testing.assert_close(x.grad,y.grad,rtol=2e-6,atol=3e-7)

    def test_chunked_state_and_initial_gradient(self):
        torch.manual_seed(703);x=torch.randn(19,2,3,requires_grad=True)
        y=x.detach().clone().requires_grad_()
        a=MultiStepLIFNode(tau=3.,detach_reset=True);b=CpuLIF(tau=3.,detach_reset=True)
        sa=torch.cat((a(x[:8]),a(x[8:])));sb=torch.cat((b(y[:8]),b(y[8:])))
        wa=torch.randn_like(sa)
        (sa*wa).sum().backward();(sb*wa).sum().backward()
        torch.testing.assert_close(sa,sb,rtol=0,atol=0)
        torch.testing.assert_close(x.grad,y.grad,rtol=2e-6,atol=3e-7)

    def test_complete_model_parameter_gradients(self):
        torch.manual_seed(704);a=GestureClassifier(steps=12,memory=4)
        with torch.no_grad():a.fc1.W_u.bias.fill_(.3)
        b=copy.deepcopy(a);self.assertEqual(enable_cpu_scan(b),4)
        x=torch.rand(2,12,2,32,32)*50;labels=torch.tensor([2,8])
        ya=a(x);yb=b(x)
        torch.testing.assert_close(ya,yb,rtol=0,atol=0)
        torch.nn.functional.cross_entropy(ya,labels).backward()
        torch.nn.functional.cross_entropy(yb,labels).backward()
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,m)
            if p.grad is None:self.assertIsNone(q.grad)
            else:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=3e-7,msg=n)

if __name__=='__main__':unittest.main()
