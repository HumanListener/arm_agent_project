const BASE_Z=-0.25, BASE_Y=0.06, L1=0.15, L2=0.15, L3=0.06;
const LIM=1.57;
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
function fk(A,t1,t2,wrist){
  const r=L1*Math.cos(t1)+L2*Math.cos(t1+t2)+L3*Math.cos(t1+t2+wrist);
  const h=BASE_Y+L1*Math.sin(t1)+L2*Math.sin(t1+t2)+L3*Math.sin(t1+t2+wrist);
  return {x:-r*Math.sin(A), z:BASE_Z+r*Math.cos(A), y:h};
}
/** 两分支 IK，返回 fk 误差最小的解 */
function ik(x,z,y,wrist=0.3){
  const A=clamp(Math.atan2(-x,z-BASE_Z),-LIM,LIM);
  const planar=Math.hypot(x,z-BASE_Z);
  let best=null;
  for (const sign of [-1,1]) {
    let t1=0,t2=0;
    for(let i=0;i<3;i++){
      const wdx=L3*Math.cos(t1+t2+wrist), wdy=L3*Math.sin(t1+t2+wrist);
      const h0=planar-wdx, v0=(y-BASE_Y)-wdy;
      const d=Math.hypot(h0,v0), phi=Math.atan2(v0,h0);
      const c2=clamp((d*d-L1*L1-L2*L2)/(2*L1*L2),-1,1);
      t2=sign*Math.acos(c2);
      t1=phi-Math.atan2(L2*Math.sin(t2),L1+L2*Math.cos(t2));
    }
    if (Math.abs(t1)>LIM || Math.abs(t2)>LIM) continue;
    const p=fk(A,t1,t2,wrist);
    const err=Math.hypot(p.x-x,p.z-z);
    if(!best||err<best.err) best={A,t1,t2,err,p};
  }
  return best;
}
console.log('== 候选坑位可达性 (y=0.03) ==');
for (const [x,z] of [[-0.25,0.15],[0.25,0.15],[0,-0.18],[-0.18,0.05],[0.18,0.05],
                     [-0.16,0.02],[0.16,0.02],[0,-0.02],[-0.20,0.28],[0.20,0.28],[0,0.30],[0,0.18]]) {
  const s=ik(x,z,0.03);
  console.log(`(${x},${z}) reach=${Math.hypot(x,z-BASE_Z).toFixed(3)} -> ` +
    (s?`err=${(s.err*1000).toFixed(1)}mm t=(${s.A.toFixed(2)},${s.t1.toFixed(2)},${s.t2.toFixed(2)})`:'不可达'));
}
console.log('\n== 工作台可达范围扫描 (y=0.03) ==');
let ok=[], bad=[];
for(let x=-0.30;x<=0.301;x+=0.05){
  let row='';
  for(let z=-0.20;z<=0.301;z+=0.05){
    const s=ik(+x.toFixed(2),+z.toFixed(2),0.03);
    row += (s && s.err<0.01) ? ' O ' : ' . ';
  }
  console.log(`x=${x.toFixed(2)} ${row}`);
}
console.log('列 z 从 -0.20 到 0.30，步长 0.05');
