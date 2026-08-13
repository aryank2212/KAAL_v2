import React, { useEffect, useRef, useState } from 'react';
import * as Engine from '../lib/SwarmEngine';

const BG = '#05080c';
const GREEN = 'rgb(0, 255, 100)';
const CYAN = 'rgb(0, 200, 255)';
const RED = 'rgb(255, 40, 40)';
const YELLOW = 'rgb(255, 200, 0)';
const BLUE = 'rgb(50, 140, 255)';
const ORANGE = 'rgb(255, 130, 20)';
const DG = 'rgb(0, 70, 35)';
const DC = 'rgb(0, 55, 80)';
const GRID_C = 'rgb(12, 22, 14)';

export default function KaalSwarmCanvas({ onTelemetry }) {
  const canvasRef = useRef(null);
  
  // Simulation State Refs
  const state = useRef({
    scene: 'start',
    inp: '40',
    drones: [],
    walls: [],
    ghosts: [],
    casualties: 0,
    mode: 'normal',
    jammed: false,
    tgt: [Engine.W - 160.0, Engine.H / 2.0],
    lk: [Engine.W - 160.0, Engine.H / 2.0],
    fc: [Engine.W - 160.0, Engine.H / 2.0],
    mesh_tgt: [0, 0],
    wall_start: null,
    frame: 0,
    mp: [0, 0]
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationId;

    const s = state.current;

    const deploy = (n) => {
      n = Math.max(5, Math.min(150, n));
      s.drones = [];
      for(let i=0; i<n; i++){
        s.drones.push(new Engine.Drone(55 + Math.random() * 130, 25 + Math.random() * (Engine.H - 50), i));
      }
      let avgX = 0, avgY = 0;
      s.drones.forEach(d => { avgX += d.pos[0]; avgY += d.pos[1]; });
      avgX /= n; avgY /= n;
      
      s.walls = [];
      s.ghosts = [];
      s.casualties = 0;
      s.mode = 'normal';
      s.jammed = false;
      s.tgt = [Engine.W - 160.0, Engine.H / 2.0];
      s.lk = [...s.tgt];
      s.fc = [avgX, avgY];
      s.mesh_tgt = [0, 0];
      s.scene = 'sim';
    };

    const setupKaal = () => {
      s.mode = 'kaal';
      s.mesh_tgt = [0, 0];
      const slotsData = Engine.makeSlots(s.drones.length);
      s.drones.forEach((d, i) => {
        d.role = slotsData[i].role;
        d.slot = slotsData[i].slot;
        d.grid_pos = slotsData[i].grid;
        d.status = 'SEARCH';
        d.seen_pos = null;
      });
      const meshDrones = s.drones.filter(d => d.role === 'mesh');
      if (meshDrones.length > 0) {
        let ax = 0, ay = 0;
        meshDrones.forEach(d => { ax += d.pos[0]; ay += d.pos[1]; });
        s.fc = [ax / meshDrones.length, ay / meshDrones.length];
      }
    };

    const respawn = (count = 10) => {
      if (s.drones.length >= 200) return;
      const startIdx = s.drones.length;
      for (let i=0; i<count; i++) {
        s.drones.push(new Engine.Drone(55 + Math.random() * 100, 25 + Math.random() * (Engine.H - 50), startIdx + i));
      }
      if (s.mode === 'kaal') {
        const sd = Engine.makeSlots(s.drones.length);
        s.drones.forEach((d, i) => {
          d.role = sd[i].role; d.slot = sd[i].slot; d.grid_pos = sd[i].grid;
        });
      }
    };

    // Events
    const onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      s.mp = [ (e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY ];
    };

    const onMouseDown = (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      if (s.scene === 'start') {
        const rx = Engine.W / 2 - 95, ry = Engine.H / 2 + 16, rw = 190, rh = 46;
        if (x > rx && x < rx + rw && y > ry && y < ry + rh) {
          deploy(parseInt(s.inp) || 40);
        }
      } else if (s.scene === 'sim') {
        if (e.button === 0) {
          const sw_rect = {x: 16, y: 48, w: 100, h: 20};
          const re_rect = {x: 124, y: 48, w: 100, h: 20};
          if (x > sw_rect.x && x < sw_rect.x + sw_rect.w && y > sw_rect.y && y < sw_rect.y + sw_rect.h) {
            s.mode === 'normal' ? setupKaal() : (s.mode = 'normal');
          } else if (x > re_rect.x && x < re_rect.x + re_rect.w && y > re_rect.y && y < re_rect.y + re_rect.h) {
            respawn(10);
          } else {
            s.wall_start = [x, y];
          }
        } else if (e.button === 2) {
          const survivors = [];
          s.drones.forEach(d => {
            if (Engine.vecDist(d.pos, s.mp) < Engine.KILL_R) {
              s.ghosts.push(new Engine.Ghost(d.pos));
              s.casualties++;
            } else {
              survivors.push(d);
            }
          });
          if (survivors.length < s.drones.length) {
            s.drones = survivors;
            if (s.mode === 'kaal') {
              const sd = Engine.makeSlots(s.drones.length);
              s.drones.forEach((d, i) => {
                d.role = sd[i].role; d.slot = sd[i].slot; d.grid_pos = sd[i].grid;
              });
            }
          }
        }
      }
    };

    const onMouseUp = (e) => {
      if (e.button === 0 && s.wall_start) {
        if (Engine.vecDist(s.wall_start, s.mp) > 15) {
          s.walls.push(new Engine.Wall([...s.wall_start], [...s.mp]));
        }
        s.wall_start = null;
      }
    };

    const onKeyDown = (e) => {
      const k = e.key.toLowerCase();
      if (s.scene === 'start') {
        if (e.key === 'Backspace') s.inp = s.inp.slice(0, -1);
        else if (/\d/.test(k) && s.inp.length < 3) s.inp += k;
        else if (e.key === 'Enter') deploy(parseInt(s.inp) || 40);
      } else {
        if (k === 'm') s.mode === 'normal' ? setupKaal() : (s.mode = 'normal');
        if (k === 'n') respawn(10);
        if (k === 'j') {
          s.jammed = !s.jammed;
          if (!s.jammed) s.drones.forEach(d => d.knows = false);
        }
        if (k === 'c') s.walls = [];
        if (k === 'r' || e.key === 'Escape') { s.scene = 'start'; s.inp = '40'; }
      }
    };

    // Render Helpers
    const drawGrid = () => {
      ctx.strokeStyle = GRID_C;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < Engine.W; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, Engine.H); }
      for (let y = 0; y < Engine.H; y += 40) { ctx.moveTo(0, y); ctx.lineTo(Engine.W, y); }
      ctx.stroke();
    };

    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    window.addEventListener('keydown', onKeyDown);

    // Initial Start screen
    s.scene = 'start';

    const loop = () => {
      s.frame++;
      
      // Update Sim
      if (s.scene === 'sim') {
        if (!s.jammed) {
          s.tgt = [...s.mp];
          s.lk = [...s.tgt];
        }

        if (s.mode === 'kaal') {
          s.drones.forEach(d => d.knows = false);
          const sources = s.drones.filter(d => d.role === 'scout' && d.status === 'LEAD');
          s.drones.forEach(d => {
            if (d.role === 'mesh' && Engine.losClear(d.pos, s.mp, s.walls) && Engine.vecDist(d.pos, s.mp) < Engine.PERC) sources.push(d);
          });
          sources.forEach(src => src.knows = true);

          const queue = [...sources];
          let idx = 0;
          while (idx < queue.length) {
            const curr = queue[idx++];
            s.drones.forEach(o => {
              if (!o.knows && Engine.vecDist(curr.pos, o.pos) < Engine.PERC && Engine.losClear(curr.pos, o.pos, s.walls)) {
                o.knows = true;
                queue.push(o);
              }
            });
          }

          if (!s.drones.some(d => d.role === 'mesh' && d.knows)) {
            s.mesh_tgt = [0, 0];
          }

          if (s.mesh_tgt[0] !== 0 || s.mesh_tgt[1] !== 0) {
            const toMt = Engine.vecSub(s.mesh_tgt, s.fc);
            const distK = Engine.vecMag(toMt);
            if (distK < 20) {
              const sees = s.drones.some(d => Engine.losClear(d.pos, s.mp, s.walls) && Engine.vecDist(d.pos, s.mp) < Engine.PERC);
              if (!sees) s.mesh_tgt = [0, 0];
            }
            if (distK > 2.0) {
              let steering = Engine.vecMul(toMt, Math.min(distK * 0.08, Engine.MAX_K * 1.6) / distK);
              const lookahead = Engine.vecAdd(s.fc, Engine.vecMul(steering, 10.0));
              s.walls.forEach(w => {
                const [cp, dist] = Engine.ptSeg(lookahead, w.p1, w.p2);
                if (dist < Engine.WALL_R * 1.5) {
                  steering = Engine.vecAdd(steering, Engine.vecMul(Engine.vecSub(lookahead, cp), 0.5 / Math.max(dist, 1.0)));
                }
              });
              
              const newFc = Engine.vecAdd(s.fc, steering);
              let collisionWall = null;
              for (const w of s.walls) {
                if (Engine.segHit(s.fc, newFc, w.p1, w.p2)[0]) { collisionWall = w; break; }
              }
              if (!collisionWall) s.fc = newFc;
              else {
                const wv = Engine.vecSub(collisionWall.p2, collisionWall.p1);
                const u = Engine.vecMul(wv, 1 / (Engine.vecMag(wv) || 1.0));
                const s1 = Engine.vecMul(u, Engine.vecDot(steering, u));
                const s2 = Engine.vecMul(s1, -1);
                let bestS = Engine.vecDist(Engine.vecAdd(s.fc, s1), s.mesh_tgt) < Engine.vecDist(Engine.vecAdd(s.fc, s2), s.mesh_tgt) ? s1 : s2;
                let n = [-u[1], u[0]];
                if (Engine.vecDot(n, toMt) < 0) n = [-n[0], -n[1]];
                bestS = Engine.vecAdd(bestS, Engine.vecMul(n, 1.2));
                const testFc = Engine.vecAdd(s.fc, bestS);
                let blocked = false;
                for (const w of s.walls) { if(Engine.segHit(s.fc, testFc, w.p1, w.p2)[0]) blocked = true; }
                if(!blocked) s.fc = testFc;
              }
            }
          }
        }

        s.drones.forEach(d => {
          if (s.mode === 'normal') {
            Engine.aiNormal(d, s.drones, s.walls, s.jammed ? s.lk : s.tgt);
            Engine.integrate(d, Engine.MAX_N, s.walls);
            if (Engine.losClear(d.pos, s.jammed ? s.lk : s.tgt, s.walls)) {
              if (Engine.vecDist(d.pos, s.jammed ? s.lk : s.tgt) < Engine.ATK_RANGE) d.atk = Math.max(d.atk, 12);
            }
          } else {
            Engine.aiKaal(d, s.drones, s.walls, s.fc, s.mesh_tgt, s.mp, s.jammed);
            Engine.integrate(d, Engine.MAX_K, s.walls);
          }
        });

        s.ghosts.forEach(g => g.life -= 5);
        s.ghosts = s.ghosts.filter(g => g.life > 0);

        // Emit telemetry to parent
        if (onTelemetry) {
          const status = s.jammed ? 'JAMMED' : (s.drones.some(d => d.atk > 0) ? 'ENGAGING' : 'NOMINAL');
          onTelemetry({
            status,
            alive: s.drones.length,
            kia: s.casualties,
            mode: s.mode.toUpperCase()
          });
        }
      }

      // Draw Loop
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, Engine.W, Engine.H);

      if (s.scene === 'start') {
        drawGrid();

        // Title
        ctx.fillStyle = GREEN;
        ctx.font = 'bold 30px "Courier New"';
        ctx.textAlign = 'center';
        ctx.fillText('▲  KAAL  SWARM  KINEMATICS', Engine.W/2, Engine.H/2 - 148);
        ctx.font = 'bold 13px "Courier New"';
        ctx.fillStyle = 'rgb(0, 75, 38)';
        ctx.fillText('AUTONOMOUS MESH DRONE TACTICAL ENGINE  v3.0', Engine.W/2, Engine.H/2 - 106);

        // Input
        ctx.fillStyle = 'rgb(0, 155, 75)';
        ctx.textAlign = 'left';
        ctx.fillText('DRONE COUNT :', Engine.W/2 - 160, Engine.H/2 - 32);
        
        ctx.fillStyle = 'rgb(8, 17, 8)';
        ctx.fillRect(Engine.W/2 + 20, Engine.H/2 - 52, 115, 38);
        ctx.strokeStyle = 'rgb(0, 195, 78)';
        ctx.strokeRect(Engine.W/2 + 20, Engine.H/2 - 52, 115, 38);
        ctx.fillStyle = GREEN;
        ctx.fillText(s.inp + (Math.floor(Date.now() / 500) % 2 ? '_' : ''), Engine.W/2 + 30, Engine.H/2 - 28);

        // Deploy Button
        ctx.fillStyle = 'rgb(0, 28, 12)';
        ctx.fillRect(Engine.W/2 - 95, Engine.H/2 + 16, 190, 46);
        ctx.strokeRect(Engine.W/2 - 95, Engine.H/2 + 16, 190, 46);
        ctx.fillStyle = GREEN;
        ctx.textAlign = 'center';
        ctx.fillText('▶   DEPLOY SWARM', Engine.W/2, Engine.H/2 + 44);
        
      } else {
        drawGrid();
        
        if (s.mode === 'kaal') {
          s.drones.forEach(d => {
            ctx.beginPath();
            ctx.arc(d.pos[0], d.pos[1], Engine.PERC, 0, Math.PI*2);
            if (d.knows) ctx.strokeStyle = d.role === 'mesh' ? 'rgb(0, 160, 180)' : 'rgb(255, 140, 0)';
            else if (d.role === 'scout' && d.status === 'REPORT') ctx.strokeStyle = 'rgb(200, 180, 0)';
            else ctx.strokeStyle = 'rgb(25, 30, 35)';
            ctx.stroke();

            if (d.knows) {
              s.drones.forEach(o => {
                if (o.knows && d.idx < o.idx && Engine.vecDist(d.pos, o.pos) < Engine.PERC && Engine.losClear(d.pos, o.pos, s.walls)) {
                  ctx.beginPath();
                  ctx.moveTo(d.pos[0], d.pos[1]);
                  ctx.lineTo(o.pos[0], o.pos[1]);
                  ctx.strokeStyle = 'rgb(0, 220, 240)';
                  ctx.stroke();
                }
              });
            }
          });
        }

        s.walls.forEach(w => {
          ctx.beginPath(); ctx.moveTo(w.p1[0], w.p1[1]); ctx.lineTo(w.p2[0], w.p2[1]);
          ctx.strokeStyle = BLUE; ctx.lineWidth = 3; ctx.stroke();
          ctx.beginPath(); ctx.arc(w.p1[0], w.p1[1], 4, 0, Math.PI*2); ctx.fillStyle = BLUE; ctx.fill();
          ctx.beginPath(); ctx.arc(w.p2[0], w.p2[1], 4, 0, Math.PI*2); ctx.fillStyle = BLUE; ctx.fill();
        });
        if (s.wall_start) {
          ctx.beginPath(); ctx.moveTo(s.wall_start[0], s.wall_start[1]); ctx.lineTo(s.mp[0], s.mp[1]);
          ctx.strokeStyle = 'rgb(55, 90, 190)'; ctx.lineWidth = 2; ctx.stroke();
        }

        const effTgt = s.jammed ? s.lk : s.tgt;
        if (s.mode !== 'kaal') {
          const col = s.jammed ? YELLOW : RED;
          const r = 10 + 5 * Math.sin(s.frame * 0.13);
          ctx.beginPath(); ctx.arc(effTgt[0], effTgt[1], r, 0, Math.PI*2); ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.stroke();
          ctx.beginPath(); ctx.arc(effTgt[0], effTgt[1], 5, 0, Math.PI*2); ctx.fillStyle = col; ctx.fill();
        }

        const attackTgt = s.mode === 'normal' ? effTgt : s.mp;
        s.drones.forEach(d => {
          if (d.atk > 0) {
            ctx.beginPath(); ctx.moveTo(d.pos[0], d.pos[1]); ctx.lineTo(attackTgt[0], attackTgt[1]);
            const a = Math.min(1.0, d.atk / 12.0);
            ctx.strokeStyle = `rgba(255, 130, 20, ${a})`; ctx.stroke();
          }
        });

        s.drones.forEach(d => {
          const angle = Math.atan2(d.vel[1], d.vel[0]);
          let col = DG;
          if (d.atk > 0) col = ORANGE;
          else if (s.mode === 'normal') col = d.knows ? GREEN : DG;
          else {
            if (d.role === 'mesh') col = d.knows ? CYAN : DC;
            else if (d.role === 'scout') col = d.status === 'REPORT' ? YELLOW : (d.status === 'LEAD' ? ORANGE : 'rgb(100,100,100)');
            else col = 'rgb(100,100,100)';
          }

          const sSize = 8;
          ctx.beginPath();
          ctx.moveTo(d.pos[0] + Math.cos(angle) * sSize, d.pos[1] + Math.sin(angle) * sSize);
          ctx.lineTo(d.pos[0] + Math.cos(angle + 2.45) * sSize * 0.62, d.pos[1] + Math.sin(angle + 2.45) * sSize * 0.62);
          ctx.lineTo(d.pos[0] + Math.cos(angle - 2.45) * sSize * 0.62, d.pos[1] + Math.sin(angle - 2.45) * sSize * 0.62);
          ctx.closePath();
          ctx.fillStyle = col; ctx.fill();
        });

        s.ghosts.forEach(g => {
          const c = `rgb(${Math.min(210, g.life + 40)}, 40, 40)`;
          ctx.beginPath(); ctx.arc(g.pos[0], g.pos[1], 6, 0, Math.PI*2); ctx.strokeStyle = c; ctx.stroke();
          ctx.beginPath(); ctx.moveTo(g.pos[0]-5, g.pos[1]-5); ctx.lineTo(g.pos[0]+5, g.pos[1]+5); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(g.pos[0]+5, g.pos[1]-5); ctx.lineTo(g.pos[0]-5, g.pos[1]+5); ctx.stroke();
        });

        // HUD Text
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(8, 8, 235, s.jammed ? 135 : 102);
        ctx.font = 'bold 13px "Courier New"';
        ctx.textAlign = 'left';
        ctx.fillStyle = s.mode === 'kaal' ? CYAN : GREEN;
        ctx.fillText(s.mode === 'kaal' ? '◆ KAAL MESH PROTOCOL' : '◇ STANDARD OPS', 16, 24);
        ctx.font = '10px "Courier New"';
        ctx.fillStyle = 'rgb(100, 140, 100)';
        ctx.fillText(`ACTIVE: ${s.drones.length}    KIA: ${s.casualties}`, 16, 40);

        ctx.fillStyle = 'rgb(0, 30, 15)'; ctx.fillRect(16, 48, 105, 22);
        ctx.strokeStyle = s.mode === 'kaal' ? CYAN : GREEN; ctx.strokeRect(16, 48, 105, 22);
        ctx.fillStyle = s.mode === 'kaal' ? CYAN : GREEN; ctx.fillText('SWITCH MODE [M]', 22, 63);

        ctx.fillStyle = 'rgb(30, 15, 0)'; ctx.fillRect(125, 48, 105, 22);
        ctx.strokeStyle = ORANGE; ctx.strokeRect(125, 48, 105, 22);
        ctx.fillStyle = ORANGE; ctx.fillText('RESPAWN +10 [N]', 131, 63);

        ctx.fillStyle = 'rgb(60, 90, 60)';
        ctx.fillText('M:MODE  N:RESPAWN  J:JAMMER  C:CLEAR  R:RESET', 16, 88);

        if (s.jammed) {
          ctx.fillStyle = 'rgba(255, 200, 0, 0.18)'; ctx.fillRect(12, 100, 215, 30);
          ctx.font = 'bold 13px "Courier New"'; ctx.fillStyle = YELLOW; ctx.fillText('⚡ COMMS JAMMED', 16, 114);
          ctx.font = '10px "Courier New"'; ctx.fillStyle = 'rgb(130, 100, 0)'; ctx.fillText('LOCKED TO LAST KNOWN POS', 16, 126);
        }
      }

      animationId = requestAnimationFrame(loop);
    };

    animationId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animationId);
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mousedown', onMouseDown);
      canvas.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [onTelemetry]);

  return (
    <div className="w-full h-full aspect-video bg-black rounded-xl overflow-hidden shadow-[0_0_20px_rgba(0,0,0,0.5)]">
      <canvas 
        ref={canvasRef} 
        width={Engine.W} 
        height={Engine.H} 
        style={{ width: '100%', height: '100%', objectFit: 'cover', cursor: 'crosshair' }} 
      />
    </div>
  );
}
