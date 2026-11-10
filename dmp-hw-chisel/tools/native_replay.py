"""Replay a trained fixture directly in Verilator, avoiding per-cycle Scala IO maps."""
import argparse,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--rtl',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();j=json.loads(a.fixture.read_text());c=j['config'];assert c==dict(inputs=1,neurons=4,d=40,bits=24,frac=12,permissionHidden=32)
a.output.mkdir(parents=True,exist_ok=True);a.output=a.output.resolve();mask=(1<<c['bits'])-1;T=len(j['raw']);assert T==944
header=['#pragma once','#include <cstdint>',f'constexpr int T={T};',f'constexpr int32_t raw_values[T]={{{",".join(map(str,j["raw"]))}}};']
for key,width in [('p',40),('m',40),('g',40),('current',4)]:
 values=','.join('{'+','.join(map(str,r[key]))+'}' for r in j['expected']);header.append(f'constexpr int32_t expected_{key}[T][{width}]={{{values}}};')
(a.output/'fixture.h').write_text('\n'.join(header))
fields={'drive':'drive','bias':'driveBias','a':'a','b':'b','rp':'readP','rpbias':'readPBias','rm':'readM','local':'local','localBias':'localBias','hidden':'permissionHidden','hiddenBias':'permissionHiddenBias','out':'permissionOut','outBias':'permissionOutBias'}
setters=[]
for key,port in fields.items():
 for i,value in enumerate(j['weights'][key]):
  if isinstance(value,list):
   for k,v in enumerate(value):setters.append(f'top.io_weights_{port}_{i}_{k} = {int(v)&mask}u;')
  else:setters.append(f'top.io_weights_{port}_{i} = {int(value)&mask}u;')
checks=[]
for key,port,width in [('p','candidate',40),('m','stable',40),('g','gate',40),('current','current',4)]:
 for i in range(width):
  value=f'top.io_{port}_{i}' if key=='g' else f'signed24(top.io_{port}_{i})'
  checks.append(f'check({value}, expected_{key}[t][{i}], "{key}[{i}]", t);')
program='''#include "VCommitMemoryCore.h"
#include "verilated.h"
#include "fixture.h"
#include <iostream>
#include <stdexcept>
#include <string>
int64_t signed24(uint32_t x) { return (x & 0x800000u) ? int64_t(x)-16777216 : x; }
void check(int64_t actual,int64_t expected,const char* field,int t) {
 if(actual!=expected) throw std::runtime_error(std::string(field)+" mismatch step "+std::to_string(t)+": "+std::to_string(actual)+" != "+std::to_string(expected));
}
int main(int argc,char** argv) {
 try {
 Verilated::commandArgs(argc,argv);VCommitMemoryCore top;
 auto tick=[&]() {top.clock=0;top.eval();top.clock=1;top.eval();};
 top.io_valid=0;top.io_clear=0;top.io_x_0=0;top.io_raw=0;
SETTERS
 top.reset=1;tick();tick();top.reset=0;tick();
 uint64_t total_cycles=0;int skipped=0;
 for(int sample=0;sample<=T;sample++) {
  int t=sample==T ? 0 : sample;
  if(sample==T) {check(top.io_ready,1,"ready before clear",t);top.io_clear=1;tick();top.io_clear=0;top.eval();}
  check(top.io_ready,1,"ready",t);
  top.io_raw=uint32_t(raw_values[t])&0xffffffu;top.io_x_0=top.io_raw;top.io_valid=1;tick();top.io_valid=0;
  top.io_x_0=uint32_t(-123)&0xffffffu;top.io_raw=999;
  int cycles=0;while(!top.io_done && cycles<2000) {tick();cycles++;}
  check(top.io_done,1,"completion",t);
CHECKS
  int opened=0;for(int i=0;i<40;i++)opened+=expected_g[t][i]>0;
  int rows=opened ? 4 : 0;
  check(top.io_stableReadRows,rows,"stable read rows",t);check(top.io_stableWrites,opened,"stable writes",t);
  check(top.io_cycles,196+rows,"processing cycles",t);check(cycles,196+rows,"observed latency",t);
  if(sample<T) {total_cycles+=cycles;if(!rows)skipped++;}
  if(sample%128==0)std::cerr << "RTL replay verified " << sample+1 << "/944 steps\\n";
  tick();
 }
 std::cout << "{\\"passed\\":true,\\"steps\\":944,\\"checked_values\\":117056,\\"cycles\\":" << total_cycles << ",\\"stable_read_skipped_steps\\":" << skipped << ",\\"reset_replay_passed\\":true,\\"full_classifier\\":false}" << std::endl;
 top.final();return 0;
 } catch(const std::exception& e) {std::cerr << e.what() << std::endl;return 1;}
}
'''.replace('SETTERS','\n'.join(setters)).replace('CHECKS','\n'.join(checks))
cpp=a.output/'replay.cpp';cpp.write_text(program);obj=a.output/'obj'
with (a.output/'build.log').open('w') as f:
 subprocess.run(['verilator','--cc','--exe','--build','-j','4','-Wno-fatal','--top-module','CommitMemoryCore','--Mdir',str(obj),'-CFLAGS','-O2 -std=c++17',str(a.rtl.resolve()),str(cpp)],check=True,stdout=f,stderr=subprocess.STDOUT)
r=subprocess.run([str(obj/'VCommitMemoryCore')],capture_output=True,text=True)
(a.output/'execution.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(source_checkpoint_sha256=j['checkpoint_sha256'],rtl_sha256=hashlib.sha256(a.rtl.read_bytes()).hexdigest(),fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),backend='native Verilator 5.006')
(a.output.parent/'results/trained-replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
