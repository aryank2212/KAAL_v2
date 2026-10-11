// src/lib/SwarmEngine.js

// Constants
export const W = 1920;
export const H = 1080;
export const MAX_N = 1.6;
export const MAX_K = 1.2;
export const PERC = 85;
export const KILL_R = 80;
export const WALL_R = 40;
export const FORM_SP = 42;
export const ATK_RANGE = 30;

// Vector utilities
export const vecAdd = (a, b) => [a[0] + b[0], a[1] + b[1]];
export const vecSub = (a, b) => [a[0] - b[0], a[1] - b[1]];
export const vecMul = (v, s) => [v[0] * s, v[1] * s];
export const vecDot = (a, b) => a[0] * b[0] + a[1] * b[1];
export const vecMagSq = (v) => v[0] * v[0] + v[1] * v[1];
export const vecMag = (v) => Math.sqrt(vecMagSq(v));
export const vecNorm = (v) => {
  const m = vecMag(v);
  return m === 0 ? [0, 0] : [v[0] / m, v[1] / m];
};
export const vecDist = (a, b) => vecMag(vecSub(a, b));

// Classes
export class Wall {
  constructor(p1, p2) {
    this.p1 = p1;
    this.p2 = p2;
  }
}

export class Ghost {
  constructor(pos) {
    this.pos = [...pos];
    this.life = 220;
  }
}

export class Drone {
  constructor(x, y, idx) {
    this.pos = [x, y];
    const a = Math.random() * 2 * Math.PI;
    const speed = 0.4 + Math.random() * 0.8;
    this.vel = [Math.cos(a) * speed, Math.sin(a) * speed];
    this.acc = [0, 0];
    this.idx = idx;
    this.knows = false;
    this.role = 'mesh';
    this.status = 'IDLE';
    this.wander = Math.random() * 2 * Math.PI;
    this.slot = [0, 0];
    this.atk = 0;
    this.grid_pos = [0, 0];
    this.seen_pos = null;
  }
}

// Geometry methods
export function segHit(a, b, c, d) {
  const ab = vecSub(b, a);
  const cd = vecSub(d, c);
  const den = ab[0] * cd[1] - ab[1] * cd[0];
  if (Math.abs(den) < 1e-9) return [false, null];
  
  const t = ((c[0] - a[0]) * cd[1] - (c[1] - a[1]) * cd[0]) / den;
  const u = ((c[0] - a[0]) * ab[1] - (c[1] - a[1]) * ab[0]) / den;
  
  if (t > 0.001 && t < 0.999 && u > 0.001 && u < 0.999) {
    return [true, vecAdd(a, vecMul(ab, t))];
  }
  return [false, null];
}

export function losClear(p1, p2, walls) {
  for (const w of walls) {
    if (segHit(p1, p2, w.p1, w.p2)[0]) {
      return false;
    }
  }
  return true;
}

export function ptSeg(pt, a, b) {
  const ab = vecSub(b, a);
  const lsq = vecDot(ab, ab);
  if (lsq < 1e-9) return [[...a], vecDist(pt, a)];
  let t = vecDot(vecSub(pt, a), ab) / lsq;
  t = Math.max(0, Math.min(1, t));
  const cp = vecAdd(a, vecMul(ab, t));
  return [cp, vecDist(pt, cp)];
}

export function reflectVel(vel, a, b) {
  const ab = vecSub(b, a);
  const n = [-ab[1], ab[0]];
  const nl = vecMag(n);
  if (nl < 1e-9) return vecMul(vel, -1);
  const nu = vecMul(n, 1 / nl);
  const dot = vecDot(vel, nu);
  return vecSub(vel, vecMul(nu, 2 * dot));
}

export function makeSlots(n) {
  const meshCount = Math.floor(n * 0.7);
  let side = Math.floor(Math.sqrt(meshCount));
  if (side < 2) side = 2;
  const meshN = side * side;
  const res = [];
  for (let i = 0; i < n; i++) {
    if (i < meshN) {
      const c = i % side;
      const r = Math.floor(i / side);
      const s = [
        (c - (side - 1) / 2) * FORM_SP,
        (r - (side - 1) / 2) * FORM_SP
      ];
      res.push({ role: 'mesh', slot: s, grid: [c, r] });
    } else {
      res.push({ role: 'scout', slot: [0, 0], grid: [0, 0] });
    }
  }
  return res;
}

// Drone AI methods
export function applySeparation(drone, peers, radius = 26, strength = 0.7) {
  for (const o of peers) {
    if (o === drone) continue;
    const diff = vecSub(drone.pos, o.pos);
    const dist = vecMag(diff);
    if (dist < radius) {
      const d = Math.max(dist, 0.01);
      const f = Math.pow((radius - d) / radius, 2);
      drone.acc = vecAdd(drone.acc, vecMul(diff, (f * strength * 6.0) / d));
    }
  }
}

export function applyWallRepulsion(drone, walls) {
  for (const w of walls) {
    const [cp, dist] = ptSeg(drone.pos, w.p1, w.p2);
    if (dist > 0.1 && dist < WALL_R) {
      const away = vecSub(drone.pos, cp);
      const factor = Math.pow((WALL_R - dist) / WALL_R, 2.0);
      drone.acc = vecAdd(drone.acc, vecMul(away, (factor * 12.0) / dist));
      
      const wv = vecSub(w.p2, w.p1);
      const wn = vecMag(wv);
      if (wn > 1e-9) {
        const u = vecMul(wv, 1 / wn);
        const velAcc = vecAdd(drone.vel, drone.acc);
        const side_push = vecDot(velAcc, u) > 0 ? u : vecMul(u, -1);
        drone.acc = vecAdd(drone.acc, vecMul(side_push, 2.0 * factor));
      }
    }
  }
}

export function aiNormal(drone, peers, walls, tgt) {
  const canSee = losClear(drone.pos, tgt, walls);
  if (canSee) {
    const toT = vecSub(tgt, drone.pos);
    const dist = Math.max(vecMag(toT), 1.0);
    if (dist > ATK_RANGE) {
      drone.acc = vecAdd(drone.acc, vecMul(toT, 0.32 / dist));
    } else {
      const perp = [-toT[1], toT[0]];
      const dirT = vecMul(toT, 1 / dist);
      const dirP = vecMul(perp, 1 / dist);
      drone.acc = vecAdd(drone.acc, vecAdd(vecMul(dirP, 0.25), vecMul(dirT, 0.05)));
      drone.atk = Math.max(drone.atk, 12);
    }
    drone.knows = true;
  } else {
    drone.wander += (Math.random() * 0.76) - 0.38;
    drone.acc = vecAdd(drone.acc, vecMul([Math.cos(drone.wander), Math.sin(drone.wander)], 0.14));
    drone.knows = false;
  }
  applySeparation(drone, peers);
  applyWallRepulsion(drone, walls);
}

export function aiKaal(drone, peers, walls, fc, meshTgt, mp, jammed) {
  if (jammed) return;

  if (drone.role === 'mesh') {
    const distM = vecDist(drone.pos, mp);
    if (distM < PERC + 20 && losClear(drone.pos, mp, walls)) {
      drone.atk = Math.max(drone.atk, 15);
      meshTgt[0] = mp[0]; meshTgt[1] = mp[1];
      
      const toM = vecSub(mp, drone.pos);
      const dM = Math.max(distM, 1.0);
      const perp = vecMul([-toM[1], toM[0]], 1 / dM);
      drone.acc = vecAdd(drone.acc, vecAdd(vecMul(perp, 0.65), vecMul(toM, 0.2 / dM)));
    }
    const goal = vecAdd(fc, drone.slot);
    const toG = vecSub(goal, drone.pos);
    const dg = vecMag(toG);
    if (dg > 1.0) {
      let pull = 0.85;
      for (const w of walls) {
        const [, dist] = ptSeg(drone.pos, w.p1, w.p2);
        if (dist < WALL_R * 1.2) {
          pull *= 0.25;
          break;
        }
      }
      drone.acc = vecAdd(drone.acc, vecMul(toG, pull / dg));
    }
  } else if (drone.role === 'scout') {
    const meshActive = meshTgt[0] !== 0 || meshTgt[1] !== 0;
    if (meshActive) {
      const toAction = vecSub(meshTgt, drone.pos);
      const da = vecMag(toAction);
      if (da > PERC * 1.2) {
        drone.acc = vecAdd(drone.acc, vecMul(toAction, 0.45 / Math.max(da, 1.0)));
      } else {
        if (da < PERC && losClear(drone.pos, meshTgt, walls)) {
          drone.atk = Math.max(drone.atk, 10);
          const toM = vecSub(meshTgt, drone.pos);
          const dM = Math.max(da, 1.0);
          const perp = vecMul([-toM[1], toM[0]], 1 / dM);
          drone.acc = vecAdd(drone.acc, vecAdd(vecMul(perp, 0.5), vecMul(toM, 0.15 / dM)));
        } else {
          drone.wander += (Math.random() * 0.2) - 0.1;
          drone.acc = vecAdd(drone.acc, vecMul([Math.cos(drone.wander), Math.sin(drone.wander)], 0.2));
        }
      }
    } else if (drone.status === 'SEARCH') {
      const distM = vecDist(drone.pos, mp);
      if (distM < PERC && losClear(drone.pos, mp, walls)) {
        drone.seen_pos = [...mp];
        drone.status = 'REPORT';
      } else {
        drone.wander += (Math.random() * 0.3) - 0.15;
        drone.acc = vecAdd(drone.acc, vecMul([Math.cos(drone.wander), Math.sin(drone.wander)], 0.28));
      }
    } else if (drone.status === 'REPORT') {
      const toFc = vecSub(fc, drone.pos);
      const dfc = vecMag(toFc);
      if (dfc < PERC && losClear(drone.pos, fc, walls)) {
        if (drone.seen_pos) {
          meshTgt[0] = drone.seen_pos[0];
          meshTgt[1] = drone.seen_pos[1];
        }
        drone.status = 'LEAD';
      } else {
        drone.acc = vecAdd(drone.acc, vecMul(toFc, 0.65 / Math.max(dfc, 1.0)));
      }
    } else if (drone.status === 'LEAD') {
      const toSeen = vecSub(drone.seen_pos, drone.pos);
      const ds = vecMag(toSeen);
      if (ds > 30) {
        drone.acc = vecAdd(drone.acc, vecMul(toSeen, 0.6 / Math.max(ds, 1.0)));
      } else {
        drone.status = 'SEARCH';
      }
    }
  }

  applySeparation(drone, peers);
  applyWallRepulsion(drone, walls);
}

export function integrate(drone, maxSpd, walls) {
  drone.vel = vecAdd(drone.vel, drone.acc);
  const spd = vecMag(drone.vel);
  if (spd > maxSpd) {
    drone.vel = vecMul(drone.vel, maxSpd / spd);
  }

  const old = [...drone.pos];
  let nxt = vecAdd(drone.pos, drone.vel);

  for (const w of walls) {
    const [hit] = segHit(old, nxt, w.p1, w.p2);
    if (hit) {
      drone.vel = vecMul(reflectVel(drone.vel, w.p1, w.p2), 0.50);
      nxt = old;
      break;
    }
  }

  drone.pos = nxt;

  const M = 14;
  if (drone.pos[0] < M) { drone.vel[0] = Math.abs(drone.vel[0]); drone.pos[0] = M; }
  else if (drone.pos[0] > W - M) { drone.vel[0] = -Math.abs(drone.vel[0]); drone.pos[0] = W - M; }
  
  if (drone.pos[1] < M) { drone.vel[1] = Math.abs(drone.vel[1]); drone.pos[1] = M; }
  else if (drone.pos[1] > H - M) { drone.vel[1] = -Math.abs(drone.vel[1]); drone.pos[1] = H - M; }

  drone.acc = [0, 0];
  if (drone.atk > 0) drone.atk -= 1;
}
