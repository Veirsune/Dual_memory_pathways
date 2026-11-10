"""Behavioral tests for the public architecture; no benchmark score assertions."""
import copy,io,unittest
import torch
from commit_memory import MemoryConfig,MemoryClassifier
from commit_memory.vector_ops import VectorLMU

class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,reader='fixed'):
        torch.manual_seed(717)
        m=MemoryClassifier(MemoryConfig(classes=5,hidden=32,memory=10,reader=reader))
        with torch.no_grad():
            for c in m.cells():c.W_x.bias.fill_(1.)
        return m

    def sample(self):
        return torch.poisson(torch.full((3,24,140),.2),generator=torch.Generator().manual_seed(91))

    def test_online_partition_and_future_causality(self):
        m=self.model('dynamic').set_stage('robust').eval();x=self.sample()
        whole=m.stream_chunk(x);m.reset_state()
        chunks=torch.cat([m.stream_chunk(x[:,i:i+7]) for i in range(0,24,7)],1)
        torch.testing.assert_close(chunks,whole,rtol=0,atol=0)
        m.reset_state();prefix=m.stream_chunk(x[:,:9]);changed=x.clone();changed[:,9:]+=3
        m.reset_state();altered=m.stream_chunk(changed)
        torch.testing.assert_close(prefix,altered[:,:9],rtol=0,atol=0)
        with self.assertRaises(ValueError):m.stream_chunk(x[:2,:1])
        m.reset_state();self.assertEqual(m.stream_chunk(x[:2,:1]).shape,(2,1,5))

    def test_fresh_forward_does_not_reuse_another_stream(self):
        m=self.model().eval();x=self.sample()
        with torch.no_grad():
            # Warm the profiled operators before an exact reproducibility check.
            for _ in range(3):m(x)
            first=m(x);m.stream_chunk(x+2);again=m(x)
        torch.testing.assert_close(first,again,rtol=0,atol=0)

    def test_reader_step_preserves_all_backbone_tensors(self):
        m=self.model('dynamic').set_stage('reader').train();x=self.sample()
        before={k:v.clone() for k,v in m.state_dict().items() if not k.startswith('adapter.')}
        opt=torch.optim.AdamW(m.adapter.parameters(),lr=.01)
        loss=torch.nn.functional.cross_entropy(m(x).mean(1),torch.tensor([0,1,2]));loss.backward()
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith('adapter.')))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.adapter.parameters() if p.grad is not None),0)
        opt.step()
        for name,tensor in before.items():torch.testing.assert_close(m.state_dict()[name],tensor,rtol=0,atol=0)

    def test_optimizer_checkpoint_resume(self):
        m=self.model().set_stage('content').train();x=self.sample();target=torch.tensor([0,1,2])
        opt=torch.optim.AdamW(m.parameters(),lr=.003)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4)
        def step(model,optimizer,scheduler,seed):
            model.train();torch.manual_seed(seed);optimizer.zero_grad(set_to_none=True)
            logits=model(x).mean(1);aux=model.memory_logits([(0,24)])[:,0]
            loss=torch.nn.functional.cross_entropy(logits,target)+torch.nn.functional.cross_entropy(aux,target)
            loss.backward();optimizer.step();scheduler.step()
        step(m,opt,schedule,10)
        buffer=io.BytesIO();torch.save({'model':m.checkpoint(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict()},buffer)
        step(m,opt,schedule,11)
        buffer.seek(0);saved=torch.load(buffer,map_location='cpu');restored=MemoryClassifier.from_checkpoint(saved['model'])
        new_opt=torch.optim.AdamW(restored.parameters(),lr=.003);new_opt.load_state_dict(saved['optimizer'])
        new_schedule=torch.optim.lr_scheduler.CosineAnnealingLR(new_opt,4);new_schedule.load_state_dict(saved['scheduler'])
        step(restored,new_opt,new_schedule,11)
        for key,value in m.state_dict().items():torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)

    def test_vector_memory_gradient(self):
        torch.manual_seed(3)
        u=torch.randn(2,4,3,dtype=torch.double,requires_grad=True)
        a=.8*torch.eye(3,dtype=torch.double);b=torch.tensor([.1,-.2,.3],dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda x:VectorLMU.apply(x,a,b),(u,),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_configuration_rejects_unsupported_architecture(self):
        with self.assertRaises(ValueError):MemoryConfig(inputs=1)
        with self.assertRaises(ValueError):MemoryConfig(family='dmp',reader='dynamic')

if __name__=='__main__':unittest.main()
