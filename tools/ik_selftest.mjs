/**
 * 直接从 sim_web_static/index.html 抽取真实运行的 IK 代码做自检。
 * 这样能保证"页面里跑的数学"和"工具脚本验证的数学"是同一份，不会有手抄错误。
 *
 * 用法：node tools/ik_selftest.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, '..', 'sim_web_static', 'index.html'), 'utf8');
const js = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];

// 抽出 IK 相关声明（BASE_*/L1..L3、DEST_BINS、clampJ、baseAngleFor、ikArm、poseToReach）
const grab = (startRe) => {
  const i = js.search(startRe);
  if (i < 0) throw new Error('未找到代码块: ' + startRe);
  return i;
};
const ikStart = grab(/^const BASE_Z/m);
const ikEnd = js.indexOf('// ============================================================\n// 3) setRobotJoints');
const ikCode = js.slice(ikStart, ikEnd);

// 坑位坐标定义在场景搭建段落里，单独抽出来注入（与 config/settings.py 的 DEST_BINS 对应）
const binsStart = js.indexOf('const DEST_BINS = {');
const binsEnd = js.indexOf('};', binsStart) + 2;
const binsCode = js.slice(binsStart, binsEnd);

const mod = new Function(binsCode + '\n' + ikCode + `
  return { ikArm, poseToReach, baseAngleFor, DEST_BINS, clampJ, BASE_Z, BASE_Y, L1, L2, L3 };
`)();
const { ikArm, poseToReach, DEST_BINS, BASE_Z, BASE_Y, L1, L2, L3 } = mod;

// 正运动学（独立实现，用来交叉验证）
function fk(pose, wrist = 0.3) {
  const A = pose.Rotation, t1 = pose.Rotation1, t2 = pose.Rotation2;
  const r = L1 * Math.cos(t1) + L2 * Math.cos(t1 + t2) + L3 * Math.cos(t1 + t2 + wrist);
  const h = BASE_Y + L1 * Math.sin(t1) + L2 * Math.sin(t1 + t2) + L3 * Math.sin(t1 + t2 + wrist);
  return { x: -r * Math.sin(A), z: BASE_Z + r * Math.cos(A), y: h };
}

console.log('=== 1) 三个分拣坑位必须可达（误差 < 10mm）===');
let fail = 0;
for (const [key, bin] of Object.entries(DEST_BINS)) {
  const pose = poseToReach(bin.x, 0.03, bin.z, 0);
  const p = fk(pose);
  const err = Math.hypot(p.x - bin.x, p.z - bin.z);
  const flag = err < 0.01 ? 'OK ' : 'FAIL';
  if (err >= 0.01) fail++;
  console.log(`  ${flag} ${key}(${bin.label}) 目标(${bin.x},${bin.z}) 实际(${p.x.toFixed(4)},${p.z.toFixed(4)}) err=${(err * 1000).toFixed(1)}mm angles=[${pose.Rotation.toFixed(2)},${pose.Rotation1.toFixed(2)},${pose.Rotation2.toFixed(2)}]`);
}

console.log('=== 2) 关节限位必须守住 ±1.57 / 夹爪 [0,1.57] ===');
for (let x = -0.35; x <= 0.351; x += 0.05) {
  for (let z = -0.30; z <= 0.351; z += 0.05) {
    const p = poseToReach(+x.toFixed(2), 0.05, +z.toFixed(2), 0);
    for (const k of ['Rotation', 'Rotation1', 'Rotation2']) {
      if (Math.abs(p[k]) > 1.5701) { console.log(`  FAIL ${k}=${p[k]} at (${x},${z})`); fail++; }
    }
    if (p.Rotation5 < 0 || p.Rotation5 > 1.57) { console.log('  FAIL 夹爪越界'); fail++; }
  }
}

console.log('=== 3) 抓取点（钢板上的工件）可达 ===');
for (const [x, z] of [[0.0094, 0.0075], [0.03, 0.03], [0.06, 0.02], [-0.02, 0.06]]) {
  const pose = poseToReach(x, 0.03, z, 0);
  const p = fk(pose);
  const err = Math.hypot(p.x - x, p.z - z);
  const flag = err < 0.01 ? 'OK ' : 'FAIL';
  if (err >= 0.01) fail++;
  console.log(`  ${flag} (${x},${z}) err=${(err * 1000).toFixed(1)}mm`);
}

console.log(fail === 0 ? '\n✅ IK 自检全部通过' : `\n❌ 有 ${fail} 项未通过`);
process.exit(fail === 0 ? 0 : 1);
