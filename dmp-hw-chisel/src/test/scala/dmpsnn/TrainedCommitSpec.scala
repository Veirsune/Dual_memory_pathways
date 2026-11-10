package dmpsnn

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files,Paths}

class TrainedCommitSpec extends AnyFlatSpec {
 behavior of "Trained Commit memory replay"
 it should "match the independent Python integer oracle for all944 steps" in {
  val fixture=sys.env.get("COMMIT_TRAINED_TRACE")
  assume(fixture.isDefined,"Set COMMIT_TRAINED_TRACE to the exported trained-trace.json")
  val j=ujson.read(Files.readString(Paths.get(fixture.get)));val cfg=j("config")
  val c=CommitConfig(inputs=cfg("inputs").num.toInt,neurons=cfg("neurons").num.toInt,d=cfg("d").num.toInt,bits=cfg("bits").num.toInt,frac=cfg("frac").num.toInt,permissionHidden=cfg("permissionHidden").num.toInt)
  val wj=j("weights")
  def vector(key:String):Array[BigInt]=wj(key).arr.map(v=>BigInt(v.num.toLong)).toArray
  def matrix(key:String):Array[Array[BigInt]]=wj(key).arr.map(row=>row.arr.map(v=>BigInt(v.num.toLong)).toArray).toArray
  val w=CommitParameters(c,matrix("drive"),vector("bias"),matrix("a"),vector("b"),matrix("rp"),vector("rpbias"),matrix("rm"),matrix("local"),vector("localBias"),matrix("hidden"),vector("hiddenBias"),matrix("out"),vector("outBias"))
  val oracle=new CommitOracle(w,true)
  var checked=0
  for((v,index)<-j("raw").arr.zipWithIndex) {
   val raw=BigInt(v.num.toLong);val(p,m,g,out)=oracle.step(Array(raw),raw)
   for((key,values)<-Seq("p"->p,"m"->m,"g"->g,"current"->out)) {
    assert(values.toSeq==j("expected")(index)(key).arr.map(v=>BigInt(v.num.toLong)).toSeq,s"oracle mismatch $index/$key")
    checked+=values.length
   }
  }
  assert(j("raw").arr.length==944)
  val dest=Paths.get(sys.env.getOrElse("COMMIT_HW_RESULTS","functional-results"));Files.createDirectories(dest)
  Files.writeString(dest.resolve("trained-oracle.json"),s"""{"passed":true,"steps":944,"checked_values":$checked}""")
 }
}
