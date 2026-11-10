package dmpsnn

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files,Paths}

case class CommitParameters(c: CommitConfig, drive: Array[Array[BigInt]], bias: Array[BigInt],
 a: Array[Array[BigInt]], b: Array[BigInt], rp: Array[Array[BigInt]], rpbias: Array[BigInt], rm: Array[Array[BigInt]],
 local: Array[Array[BigInt]], localBias: Array[BigInt], hidden: Array[Array[BigInt]], hiddenBias: Array[BigInt],
 out: Array[Array[BigInt]], outBias: Array[BigInt])
object CommitFixture {
 def make(c: CommitConfig): CommitParameters = {
  val q=BigInt(c.one)
  CommitParameters(c,Array.tabulate(c.d,c.inputs)((i,j)=>if(j==i%c.inputs) q else BigInt(0)),Array.fill(c.d)(BigInt(0)),
   Array.tabulate(c.d,c.d)((i,j)=>if(i==j) q*3/4 else if(j==(i+1)%c.d) q/16 else BigInt(0)),Array.fill(c.d)(q/8),
   Array.tabulate(c.neurons,c.d)((i,j)=>if(j==i%c.d) q/4 else BigInt(0)),Array.fill(c.neurons)(BigInt(0)),
   Array.tabulate(c.neurons,c.d)((i,j)=>if(j==i%c.d) q else BigInt(0)),
   Array.fill(c.d,8)(BigInt(0)),Array.fill(c.d)(BigInt(0)),
   Array.tabulate(c.permissionHidden,8)((i,j)=>if((i==0 && j==0)||(i==1 && j==4)) q else BigInt(0)),Array.fill(c.permissionHidden)(BigInt(0)),
   Array.tabulate(c.d,c.permissionHidden)((i,j)=>if(j==0) -6*q else if(j==1) 2*q else BigInt(0)),Array.fill(c.d)(2*q))
 }
 def load(io: CommitWeights,w: CommitParameters): Unit = {
  def matrix(port: Vec[Vec[SInt]],values: Array[Array[BigInt]]): Unit = for(i<-values.indices;j<-values(i).indices)port(i)(j).poke(values(i)(j).S)
  def vector(port: Vec[SInt],values: Array[BigInt]): Unit = for(i<-values.indices)port(i).poke(values(i).S)
  matrix(io.drive,w.drive);vector(io.driveBias,w.bias);matrix(io.a,w.a);vector(io.b,w.b)
  matrix(io.readP,w.rp);vector(io.readPBias,w.rpbias);matrix(io.readM,w.rm);matrix(io.local,w.local);vector(io.localBias,w.localBias)
  matrix(io.permissionHidden,w.hidden);vector(io.permissionHiddenBias,w.hiddenBias);matrix(io.permissionOut,w.out);vector(io.permissionOutBias,w.outBias)
 }
}
/** Independent arbitrary-precision oracle; rounding only at declared boundaries. */
class CommitOracle(w: CommitParameters,commit: Boolean) {
 val c=w.c;val q=BigInt(c.one)
 var p=Array.fill(c.d)(BigInt(0));var m=p.clone();var ema=Array.fill(4)(BigInt(0));var rawEMA=ema.clone()
 var previousMean=BigInt(0);var previousRaw=BigInt(0)
 def sat(x: BigInt): BigInt=x.max(-(BigInt(1)<<(c.bits-1))).min((BigInt(1)<<(c.bits-1))-1)
 def dot(a: Array[BigInt],b: Array[BigInt]): BigInt=a.zip(b).map{case(x,y)=>x*y}.sum
 def affine(a: Array[BigInt],x: Array[BigInt],bias: BigInt): BigInt=sat((dot(a,x)+(bias<<c.frac))>>c.frac)
 def features(x: Array[BigInt],prev: BigInt): Array[BigInt] = {
  val mean=sat(x.sum/x.length)
  Array(mean,sat((mean-prev).abs),sat((x.map(v=>(v-mean)*(v-mean)).sum/x.length)>>c.frac),sat(BigInt(x.count(_>0))*q/x.length))
 }
 def sigmoid(v: BigInt): BigInt = {
  val index=((v+8*q+q/32)>>(c.frac-4)).max(BigInt(0)).min(BigInt(256)).toInt
  BigInt(math.round(c.one/(1+math.exp(-(-8.0+index/16.0)))))
 }
 def step(x: Array[BigInt],raw: BigInt): (Array[BigInt],Array[BigInt],Array[BigInt],Array[BigInt]) = {
  val old=p.clone();val u=w.drive.indices.map(i=>affine(w.drive(i),x,w.bias(i)).max(BigInt(0))).toArray
  p=w.a.indices.map(i=>sat((dot(w.a(i),old)+w.b(i)*u(if(commit)i else 0))>>c.frac)).toArray
  var gate=Array.fill(c.d)(BigInt(0))
  if(commit) {
   val f=features(x,previousMean);val rf=features(Array(raw),previousRaw);val local=f++ema;val input=rf++rawEMA
   val h=w.hidden.indices.map(i=>affine(w.hidden(i),input,w.hiddenBias(i)).max(BigInt(0))).toArray
   gate=w.local.indices.map {i=>
    val probability=sigmoid(affine(w.local(i),local,w.localBias(i)))
    if(raw>0 && affine(w.out(i),h,w.outBias(i))>=0 && probability>=math.ceil(.02*c.one).toInt)probability else BigInt(0)
   }.toArray
   m=m.indices.map(i=>sat(((q-gate(i))*m(i)+gate(i)*old(i))>>c.frac)).toArray
   val rho=BigInt(math.round(.8*c.one));ema=ema.indices.map(i=>sat((rho*ema(i)+(q-rho)*f(i))>>c.frac)).toArray
   rawEMA=rawEMA.indices.map(i=>sat((rho*rawEMA(i)+(q-rho)*rf(i))>>c.frac)).toArray;previousMean=f(0);previousRaw=raw
  }
  val out=w.rp.indices.map {i=>sat(affine(w.rp(i),p,w.rpbias(i))+(if(commit)sat(dot(w.rm(i),m)>>c.frac) else BigInt(0)))}.toArray
  (p.clone(),m.clone(),gate,out)
 }
}
class MemoryResult(c: CommitConfig) extends Bundle {
 val p=Vec(c.d,SInt(c.bits.W));val m=Vec(c.d,SInt(c.bits.W));val g=Vec(c.d,UInt((c.frac+1).W));val current=Vec(c.neurons,SInt(c.bits.W))
 val cycles=UInt(32.W);val reads=UInt(32.W);val writes=UInt(32.W)
}
class MemoryComparison(c: CommitConfig) extends Module {
 val io=IO(new Bundle {
  val valid=Input(Bool());val ready=Output(Bool());val clear=Input(Bool());val x=Input(Vec(c.inputs,SInt(c.bits.W)));val raw=Input(SInt(c.bits.W));val weights=Input(new CommitWeights(c));val done=Output(Bool());val result=Output(Vec(3,new MemoryResult(c)))
 })
 val modules=Seq(Module(new CommitMemoryCore(c,false)),Module(new CommitMemoryCore(c,true,false)),Module(new CommitMemoryCore(c,true,true)))
 val finished=RegInit(VecInit(Seq.fill(3)(false.B)));val result=Reg(Vec(3,new MemoryResult(c)))
 io.ready:=modules.map(_.io.ready).reduce(_ && _);io.done:=finished.reduce(_ && _);io.result:=result
 when(io.clear || (io.valid && io.ready)){finished:=VecInit(Seq.fill(3)(false.B))}
 for((m,i)<-modules.zipWithIndex) {
  m.io.valid:=io.valid && io.ready;m.io.clear:=io.clear;m.io.x:=io.x;m.io.raw:=io.raw;m.io.weights:=io.weights
  when(m.io.done) {finished(i):=true.B;result(i).p:=m.io.candidate;result(i).m:=m.io.stable;result(i).g:=m.io.gate;result(i).current:=m.io.current;result(i).cycles:=m.io.cycles;result(i).reads:=m.io.stableReadRows;result(i).writes:=m.io.stableWrites}
 }
}
class CommitMemorySpec extends AnyFlatSpec with ChiselScalatestTester {
 behavior of "Commit memory hardware"
 val destination=Paths.get(sys.env.getOrElse("COMMIT_HW_RESULTS","functional-results"));Files.createDirectories(destination)
 def runSequence(c: CommitConfig,w: CommitParameters,sequence: Seq[(String,Array[BigInt],BigInt)],name: String,waves: Boolean=false): Unit = {
  val annotations=Seq(VerilatorBackendAnnotation) ++ (if(waves)Seq(WriteVcdAnnotation) else Seq.empty)
  test(new MemoryComparison(c)).withAnnotations(annotations) {dut=>
   dut.clock.setTimeout(0);CommitFixture.load(dut.io.weights,w);dut.io.valid.poke(false.B);dut.io.clear.poke(false.B);dut.io.raw.poke(0.S);dut.io.x.foreach(_.poke(0.S));dut.clock.step()
   val refs=Seq(new CommitOracle(w,false),new CommitOracle(w,true),new CommitOracle(w,true))
   val lines=scala.collection.mutable.ArrayBuffer("step,condition,variant,d,neurons,cycles,stable_read_rows,stable_writes,open_coordinates")
   for(((condition,x,raw),step)<-sequence.zipWithIndex) {
    while(!dut.io.ready.peekBoolean())dut.clock.step()
    x.indices.foreach(i=>dut.io.x(i).poke(x(i).S));dut.io.raw.poke(raw.S);dut.io.valid.poke(true.B);dut.clock.step();dut.io.valid.poke(false.B)
    // Exercise input latching: live inputs change while the transaction is busy.
    dut.io.x.foreach(_.poke((-123).S));dut.io.raw.poke(999.S)
    var cycles=0;while(!dut.io.done.peekBoolean() && cycles<2000){dut.clock.step();cycles+=1}
    assert(dut.io.done.peekBoolean(),s"transaction timeout at $step")
    for(i<-0 until 3) {
     val (p,m,g,out)=refs(i).step(x,raw);val r=dut.io.result(i)
     p.indices.foreach(j=>r.p(j).expect(p(j).S));m.indices.foreach(j=>r.m(j).expect(m(j).S));g.indices.foreach(j=>r.g(j).expect(g(j).U));out.indices.foreach(j=>r.current(j).expect(out(j).S))
     val open=g.count(_>0);val readRows=if(i==0 || (i==2 && open==0))0 else c.neurons
     r.reads.expect(readRows.U);r.writes.expect(open.U)
     val expectedCycles=if(i==0)1+c.d+c.neurons else 4*c.d+c.permissionHidden+c.neurons+readRows
     r.cycles.expect(expectedCycles.U)
     lines+=s"$step,$condition,${Seq("dmp","commit_dense","commit_cached")(i)},${c.d},${c.neurons},$expectedCycles,$readRows,$open,$open"
    }
    dut.clock.step()
   }
   // Reset must invalidate cached reads and histories as well as both states.
   while(!dut.io.ready.peekBoolean())dut.clock.step()
   dut.io.clear.poke(true.B);dut.clock.step();dut.io.clear.poke(false.B)
   dut.io.raw.poke(0.S);dut.io.x.foreach(_.poke(0.S));dut.io.valid.poke(true.B);dut.clock.step();dut.io.valid.poke(false.B)
   var n=0;while(!dut.io.done.peekBoolean() && n<2000){dut.clock.step();n+=1};assert(n<2000)
   for(i<-0 until 3) {dut.io.result(i).p.foreach(_.expect(0.S));dut.io.result(i).m.foreach(_.expect(0.S));dut.io.result(i).current.foreach(_.expect(0.S))}
   Files.write(destination.resolve(name+".csv"),(lines.mkString("\n")+"\n").getBytes("UTF-8"))
  }
 }
 it should "match independent fixed-point states and reads through silence noise and switching" in {
  val c=CommitConfig();val q=BigInt(c.one)
  val signal=(0 until 12).map(i=>("signal",Array(q/4,if(i%2==0)q/8 else q/2),q/4))
  val silence=Seq.fill(80)(("silence80",Array(BigInt(0),BigInt(0)),BigInt(0)))
  val noise=Seq.fill(80)(("noise80",Array(q,q/2),q))
  val next=(0 until 12).map(i=>("new_signal",Array(q/8,q/4),q/4))
  runSequence(c,CommitFixture.make(c),signal++silence++next++noise++next,"retention",true)
 }
 it should "use previous candidate rather than the current candidate for a half-open gate" in {
  val c=CommitConfig();val q=BigInt(c.one);val w=CommitFixture.make(c)
  runSequence(c,w,Seq.fill(5)(("half_gate",Array(q/4,q/2),q/4)),"delay")
 }
 it should "close only low-logit coordinates and recompute the whole stable read when any opens" in {
  val c=CommitConfig();val q=BigInt(c.one);val w=CommitFixture.make(c);w.localBias(0)= -6*q
  runSequence(c,w,Seq.fill(5)(("partial_gate",Array(q/4,q/8),q/4)),"partial")
 }
 it should "match signed saturation and mixed transition matrices without wrapping" in {
  val c=CommitConfig(bits=18);val q=BigInt(c.one);val w=CommitFixture.make(c)
  w.a(0)(0)=4*q;w.a(0)(1)= -q;w.a(1)(0)= -q;w.a(1)(1)=4*q;w.b(0)=4*q;w.b(1)= -4*q
  runSequence(c,w,Seq.fill(20)(("saturation",Array(6*q,2*q),q/4)),"saturation")
 }
 it should "match a randomized stream and programmable feature-dependent local gates" in {
  val c=CommitConfig(inputs=4,d=4,neurons=8);val q=BigInt(c.one);val w=CommitFixture.make(c);val rng=new scala.util.Random(20261104L)
  for(i<-0 until c.d;j<-0 until 8)w.local(i)(j)=BigInt(rng.nextInt(c.one)-c.one/2)
  for(i<-0 until c.d)w.localBias(i)=BigInt((i-2)*c.one)
  val seq=(0 until 60).map{i=>val raw=if(i%7==0)BigInt(0) else BigInt(rng.nextInt(c.one));("random",Array.fill(c.inputs)(BigInt(rng.nextInt(c.one))),raw)}
  runSequence(c,w,seq,"random")
 }
 for(d<-Seq(10,30)) it should s"measure matched scalar and vector memory schedules at d=$d" in {
  val c=CommitConfig(inputs=4,neurons=32,d=d);val q=BigInt(c.one)
  val s=Seq.fill(4)(("signal",Array(q/4,q/8,q/2,q/4),q/4))++Seq.fill(4)(("silence",Array.fill(4)(BigInt(0)),BigInt(0)))++Seq.fill(4)(("noise",Array(q,q/2,q,q/2),q))
  runSequence(c,CommitFixture.make(c),s,s"capacity$d")
 }
}
