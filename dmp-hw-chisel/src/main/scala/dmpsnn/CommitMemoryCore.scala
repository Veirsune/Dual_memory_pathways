package dmpsnn

import chisel3._
import chisel3.util._

/** Functional fixed-point memory-layer implementation, NOT a 22FDX macro mapping.
  * All values are signed Q(bits-frac-1).frac, except gates which represent [0,1].
  * Weight ports must remain stable until clear/reset. A scalar raw pixel stream
  * supplies the causal permission features; x can be the pixel or hidden spikes.
  * Includes vector drive, full-A candidate, delayed commit, local learned gate,
  * pixel permission MLP, independent reads and optional stable-read caching.
  * Fast/LIF/classifier paths are outside this module and are not counted here.
  */
case class CommitConfig(inputs: Int = 2, neurons: Int = 4, d: Int = 2,
                        bits: Int = 24, frac: Int = 12, permissionHidden: Int = 32) {
  require(inputs >= 1 && neurons >= 1 && d >= 1 && permissionHidden >= 1)
  require(frac >= 5 && bits > frac + 4 && bits <= 30)
  val one: Int = 1 << frac
}
class CommitWeights(c: CommitConfig) extends Bundle {
  val drive = Vec(c.d, Vec(c.inputs, SInt(c.bits.W)))
  val driveBias = Vec(c.d, SInt(c.bits.W))
  val a = Vec(c.d, Vec(c.d, SInt(c.bits.W)))
  val b = Vec(c.d, SInt(c.bits.W))
  val readP = Vec(c.neurons, Vec(c.d, SInt(c.bits.W)))
  val readPBias = Vec(c.neurons, SInt(c.bits.W))
  val readM = Vec(c.neurons, Vec(c.d, SInt(c.bits.W)))
  val local = Vec(c.d, Vec(8, SInt(c.bits.W)))
  val localBias = Vec(c.d, SInt(c.bits.W))
  val permissionHidden = Vec(c.permissionHidden, Vec(8, SInt(c.bits.W)))
  val permissionHiddenBias = Vec(c.permissionHidden, SInt(c.bits.W))
  val permissionOut = Vec(c.d, Vec(c.permissionHidden, SInt(c.bits.W)))
  val permissionOutBias = Vec(c.d, SInt(c.bits.W))
}
class CommitMemoryCore(c: CommitConfig, commit: Boolean = true, cached: Boolean = true) extends Module {
  val io = IO(new Bundle {
    val valid = Input(Bool()); val ready = Output(Bool()); val clear = Input(Bool())
    val x = Input(Vec(c.inputs, SInt(c.bits.W))); val raw = Input(SInt(c.bits.W))
    val weights = Input(new CommitWeights(c))
    val done = Output(Bool()); val current = Output(Vec(c.neurons, SInt(c.bits.W)))
    val candidate = Output(Vec(c.d, SInt(c.bits.W))); val stable = Output(Vec(c.d, SInt(c.bits.W)))
    val gate = Output(Vec(c.d, UInt((c.frac+1).W)))
    val cycles = Output(UInt(32.W)); val stableReadRows = Output(UInt(32.W)); val stableWrites = Output(UInt(32.W))
  })
  def zeros(n: Int) = VecInit(Seq.fill(n)(0.S(c.bits.W)))
  def sat(v: SInt): SInt = Mux(v > ((BigInt(1)<<(c.bits-1))-1).S,
    ((BigInt(1)<<(c.bits-1))-1).S(c.bits.W),
    Mux(v < (-(BigInt(1)<<(c.bits-1))).S, (-(BigInt(1)<<(c.bits-1))).S(c.bits.W), v)).asSInt
  def dot(a: Seq[SInt], b: Seq[SInt]): SInt = a.zip(b).map { case (x,y) => x*y }.reduce(_ +& _)
  def affine(w: Seq[SInt], x: Seq[SInt], bias: SInt): SInt = sat((dot(w,x) +& (bias << c.frac)) >> c.frac)
  def features(x: Seq[SInt], previousMean: SInt): Vec[SInt] = {
    val mean = sat(x.reduce(_ +& _) / x.size.S)
    val delta = mean -& previousMean
    val variance = x.map(v => { val diff=v -& mean; diff*diff }).reduce(_ +& _)
    VecInit(Seq(mean, sat(Mux(delta < 0.S, -delta, delta)),
      sat((variance / x.size.S) >> c.frac),
      sat(((PopCount(x.map(_ > 0.S)).zext << c.frac) / x.size.S))))
  }
  def sigmoid(v: SInt): UInt = {
    // Nearest LUT point, 1/16 logit spacing on [-8,8], clamped outside.
    val index = Mux(v <= (-8*c.one).S, 0.U,
      Mux(v >= (8*c.one).S, 256.U, ((v +& (8*c.one+c.one/32).S) >> (c.frac-4)).asUInt))
    val table=VecInit((0 to 256).map(i => math.round(c.one/(1.0+math.exp(-(-8.0+i/16.0)))).toInt.U((c.frac+1).W)))
    table(index)
  }
  val p=RegInit(zeros(c.d)); val previousP=RegInit(zeros(c.d)); val m=RegInit(zeros(c.d))
  val input=RegInit(zeros(c.inputs)); val u=RegInit(zeros(c.d)); val raw=RegInit(0.S(c.bits.W))
  val localFeatures=RegInit(zeros(8)); val permissionFeatures=RegInit(zeros(8))
  val localEMA=RegInit(zeros(4)); val rawEMA=RegInit(zeros(4))
  val previousMean=RegInit(0.S(c.bits.W)); val previousRaw=RegInit(0.S(c.bits.W))
  val hidden=RegInit(zeros(c.permissionHidden)); val g=RegInit(VecInit(Seq.fill(c.d)(0.U((c.frac+1).W))))
  val rp=RegInit(zeros(c.neurons)); val rm=RegInit(zeros(c.neurons))
  val idx=RegInit(0.U(log2Ceil(Seq(c.inputs,c.neurons,c.d,c.permissionHidden).max+1).W))
  val anyGate=RegInit(false.B); val cycleCount=RegInit(0.U(32.W)); val reads=RegInit(0.U(32.W));val writes=RegInit(0.U(32.W))
  val idle :: drive :: candidate :: permission :: gate :: stable :: readP :: readM :: output :: Nil = Enum(9)
  val state=RegInit(idle)
  io.ready := state === idle && !io.clear; io.done := state === output
  io.candidate := p; io.stable := m;io.gate := g
  for (n <- 0 until c.neurons) io.current(n) := sat(rp(n) +& (if(commit) rm(n) else 0.S))
  io.cycles := cycleCount;io.stableReadRows := reads;io.stableWrites := writes
  when(state =/= idle && state =/= output) { cycleCount := cycleCount+1.U }
  switch(state) {
    is(idle) {
      when(io.clear) {
        p:=zeros(c.d);previousP:=zeros(c.d);m:=zeros(c.d);rp:=zeros(c.neurons);rm:=zeros(c.neurons)
        localEMA:=zeros(4);rawEMA:=zeros(4);previousMean:=0.S;previousRaw:=0.S
        g:=VecInit(Seq.fill(c.d)(0.U((c.frac+1).W)));cycleCount:=0.U;reads:=0.U;writes:=0.U
      }.elsewhen(io.valid) {
        input:=io.x;raw:=io.raw;previousP:=p;idx:=0.U;cycleCount:=0.U;reads:=0.U;writes:=0.U;anyGate:=false.B
        if(commit) {
          val f=features(io.x.toSeq,previousMean);val rf=features(Seq(io.raw),previousRaw)
          val rho=math.round(.8*c.one).toInt
          for(i <- 0 until 4) {
            localFeatures(i):=f(i);localFeatures(i+4):=localEMA(i)
            permissionFeatures(i):=rf(i);permissionFeatures(i+4):=rawEMA(i)
            localEMA(i):=sat((rho.S*localEMA(i) +& (c.one-rho).S*f(i)) >> c.frac)
            rawEMA(i):=sat((rho.S*rawEMA(i) +& (c.one-rho).S*rf(i)) >> c.frac)
          }
          previousMean:=f(0);previousRaw:=io.raw
        }
        state:=drive
      }
    }
    is(drive) {
      val v=affine(io.weights.drive(idx).toSeq,input.toSeq,io.weights.driveBias(idx))
      u(idx):=Mux(v>0.S,v,0.S)
      when(idx === (if(commit) c.d-1 else 0).U) {idx:=0.U;state:=candidate}.otherwise {idx:=idx+1.U}
    }
    is(candidate) {
      val value=dot(io.weights.a(idx).toSeq,previousP.toSeq) +& (io.weights.b(idx)*u(if(commit) idx else 0.U))
      p(idx):=sat(value >> c.frac)
      when(idx === (c.d-1).U) {idx:=0.U;state:=(if(commit) permission else readP)}.otherwise {idx:=idx+1.U}
    }
    is(permission) {
      val v=affine(io.weights.permissionHidden(idx).toSeq,permissionFeatures.toSeq,io.weights.permissionHiddenBias(idx))
      hidden(idx):=Mux(v>0.S,v,0.S)
      when(idx === (c.permissionHidden-1).U) {idx:=0.U;state:=gate}.otherwise {idx:=idx+1.U}
    }
    is(gate) {
      val local=affine(io.weights.local(idx).toSeq,localFeatures.toSeq,io.weights.localBias(idx))
      val allowed=affine(io.weights.permissionOut(idx).toSeq,hidden.toSeq,io.weights.permissionOutBias(idx)) >= 0.S
      val probability=sigmoid(local)
      val opened=allowed && raw>0.S && probability >= math.ceil(.02*c.one).toInt.U
      g(idx):=Mux(opened,probability,0.U);anyGate:=anyGate || opened
      when(idx === (c.d-1).U) {idx:=0.U;state:=stable}.otherwise {idx:=idx+1.U}
    }
    is(stable) {
      when(g(idx) =/= 0.U) {
        val value=(c.one.U((c.frac+1).W)-g(idx)).zext*m(idx) +& g(idx).zext*previousP(idx)
        m(idx):=sat(value >> c.frac);writes:=writes+1.U
      }
      when(idx === (c.d-1).U) {idx:=0.U;state:=readP}.otherwise {idx:=idx+1.U}
    }
    is(readP) {
      rp(idx):=affine(io.weights.readP(idx).toSeq,p.toSeq,io.weights.readPBias(idx))
      when(idx === (c.neurons-1).U) {
        idx:=0.U
        if(commit) {state:=Mux(anyGate || (!cached).B,readM,output)} else {state:=output}
      }.otherwise {idx:=idx+1.U}
    }
    is(readM) {
      rm(idx):=sat(dot(io.weights.readM(idx).toSeq,m.toSeq) >> c.frac);reads:=reads+1.U
      when(idx === (c.neurons-1).U) {idx:=0.U;state:=output}.otherwise {idx:=idx+1.U}
    }
    is(output) {state:=idle}
  }
  assert(!(io.clear && state =/= idle), "clear must be issued while idle")
}

object CommitMemoryVerilog extends App {
  chisel3.emitVerilog(new CommitMemoryCore(CommitConfig()),Array("--target-dir","generated-commit"))
}
object MatchedDmpMemoryVerilog extends App {
  chisel3.emitVerilog(new CommitMemoryCore(CommitConfig(),commit=false),Array("--target-dir","generated-dmp-matched"))
}
object TrainedCommitMemoryVerilog extends App {
  chisel3.emitVerilog(new CommitMemoryCore(CommitConfig(inputs=1,neurons=4,d=40)),Array("--target-dir","generated-commit-trained"))
}
