#!/usr/bin/env python3
"""
KAAL SWARM v3 — TACTICAL SIMULATION
======================================
Controls:
  Mouse move    → Target follows cursor
  Left drag     → Draw wall segment
  Right click   → EMP strike (kills drones in radius)
  M             → Toggle Normal / KAAL mode
  J             → Toggle comms jammer
  C             → Clear all walls
  R / ESC       → Return to start screen
"""

import pygame
import numpy as np
import random
import math
import json
import sys

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════
W, H      = 1920, 1080
FPS       = 60
MAX_N     = 1.6    # Normal mode speed cap
MAX_K     = 1.2    # KAAL mode speed cap (slower, disciplined)
PERC      = 85     # KAAL perception / mesh radius
KILL_R    = 80     # EMP blast radius
WALL_R    = 40     # Wall repulsion start distance
FORM_SP   = 42     # Formation grid spacing
ATK_RANGE = 30     # Attack trigger distance

# ── Colours ──────────────────────────────────────────────────
BG      = (5,   8,  12)
GREEN   = (0,  255, 100)
CYAN    = (0,  200, 255)
RED     = (255,  40,  40)
YELLOW  = (255, 200,   0)
BLUE    = (50,  140, 255)
ORANGE  = (255, 130,  20)
DG      = (0,   70,  35)    # dim green (normal, lost)
DC      = (0,   55,  80)    # dim cyan  (kaal, lost)
GRID_C  = (12,  22,  14)
GHOST_C = (210,  50,  50)


# ════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════
class Wall:
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)

class Ghost:
    def __init__(self, pos):
        self.pos  = np.array(pos, dtype=float)
        self.life = 220    # frames until fade

class Drone:
    def __init__(self, x, y, idx):
        self.pos    = np.array([x, y], dtype=float)
        a           = random.uniform(0, 2 * math.pi)
        self.vel    = np.array([math.cos(a), math.sin(a)]) * random.uniform(0.4, 1.2)
        self.acc    = np.zeros(2)
        self.idx    = idx
        self.knows  = False        # Part of current awareness chain
        self.role   = 'mesh'       # 'mesh' or 'scout'
        self.status = 'IDLE'       # 'IDLE', 'SEARCH', 'REPORT', 'LEAD'
        self.wander = random.uniform(0, 2 * math.pi)
        self.slot   = np.zeros(2)  
        self.atk    = 0            
        self.grid_pos = (0, 0)     
        self.seen_pos = None       # STATIC location where target was spotted

def make_slots_v4(n):
    """Creates a core mesh and designates external scouts"""
    # Use 70% of drones for the core structure
    mesh_count = int(n * 0.7)
    side = int(math.sqrt(mesh_count))
    if side < 2: side = 2
    mesh_n = side * side
    res = []
    for i in range(n):
        if i < mesh_n:
            c, r = i % side, i // side
            s = np.array([(c - (side-1)/2) * FORM_SP,
                          (r - (side-1)/2) * FORM_SP])
            res.append({'role': 'mesh', 'slot': s, 'grid': (c, r)})
        else:
            # Drones outside the square are scouts
            res.append({'role': 'scout', 'slot': np.zeros(2), 'grid': (0,0)})
    return res


# ════════════════════════════════════════════════════════════
#  GEOMETRY
# ════════════════════════════════════════════════════════════
def seg_hit(a, b, c, d):
    """Segment AB ∩ CD → (bool, point or None)"""
    ab  = b - a;  cd = d - c
    den = ab[0] * cd[1] - ab[1] * cd[0]
    if abs(den) < 1e-9:
        return False, None
    t = ((c[0]-a[0])*cd[1] - (c[1]-a[1])*cd[0]) / den
    u = ((c[0]-a[0])*ab[1] - (c[1]-a[1])*ab[0]) / den
    if 0.001 < t < 0.999 and 0.001 < u < 0.999:
        return True, a + t * ab
    return False, None

def los_clear(p1, p2, walls):
    """True if no wall blocks the line p1→p2"""
    for w in walls:
        if seg_hit(p1, p2, w.p1, w.p2)[0]:
            return False
    return True

def pt_seg(pt, a, b):
    """Closest point on segment AB to pt, and distance"""
    ab  = b - a
    lsq = np.dot(ab, ab)
    if lsq < 1e-9:
        return a.copy(), np.linalg.norm(pt - a)
    t   = max(0.0, min(1.0, np.dot(pt - a, ab) / lsq))
    cp  = a + t * ab
    return cp, np.linalg.norm(pt - cp)

def wall_normal(a, b, side_pt):
    """Unit normal of wall AB pointing toward side_pt"""
    ab = b - a
    n  = np.array([-ab[1], ab[0]], dtype=float)
    n /= (np.linalg.norm(n) or 1.0)
    if np.dot(n, side_pt - a) < 0:
        n = -n
    return n

def reflect_vel(vel, a, b):
    ab = b - a
    n  = np.array([-ab[1], ab[0]], dtype=float)
    nl = np.linalg.norm(n)
    if nl < 1e-9:
        return -vel
    n /= nl
    return vel - 2.0 * np.dot(vel, n) * n

# ════════════════════════════════════════════════════════════
#  DRONE AI
# ════════════════════════════════════════════════════════════
def apply_separation(drone, peers, radius=26, strength=0.7):
    """Prevents drones from overlapping with robust safety check"""
    for o in peers:
        if o is drone: continue
        diff = drone.pos - o.pos
        dist = np.linalg.norm(diff)
        if dist < radius:
            d = max(dist, 0.01) # Safety floor
            f = ((radius - d) / radius) ** 2
            drone.acc += (diff / d) * f * strength * 6.0

def apply_wall_repulsion(drone, walls):
    """Smarter wall avoidance: Radial Push + Tangential Steer"""
    for w in walls:
        cp, dist = pt_seg(drone.pos, w.p1, w.p2)
        if 0.1 < dist < WALL_R:
            # ── Stronger Radial Push (Push away) ──
            away = drone.pos - cp
            factor = ((WALL_R - dist) / WALL_R) ** 2.0
            drone.acc += (away / dist) * factor * 12.0
            
            # ── Smart Tangential Steer (Slide through gaps) ──
            wv = w.p2 - w.p1
            wn = np.linalg.norm(wv)
            if wn > 1e-9:
                u = wv / wn
                # Slide along the wall in the direction of current intent
                side_push = u if np.dot(drone.vel + drone.acc, u) > 0 else -u
                drone.acc += side_push * (2.0 * factor)

def ai_normal(drone, peers, walls, tgt):
    """Normal: each drone individually seeks target if it has LOS"""
    can_see = los_clear(drone.pos, tgt, walls)
    if can_see:
        to_t = tgt - drone.pos
        dist = np.linalg.norm(to_t) or 1.0
        if dist > ATK_RANGE:
            drone.acc += (to_t / dist) * 0.32   # approach
        else:
            # Circle the target rather than pile up
            perp       = np.array([-to_t[1], to_t[0]]) / dist
            drone.acc += perp * 0.25 + (to_t / dist) * 0.05
            drone.atk  = max(drone.atk, 12)
        drone.knows = True
    else:
        # Lost — wander randomly
        drone.wander += random.uniform(-0.38, 0.38)
        drone.acc    += np.array([math.cos(drone.wander),
                                   math.sin(drone.wander)]) * 0.14
        drone.knows   = False

    apply_separation(drone, peers)
    apply_wall_repulsion(drone, walls)

def ai_kaal(drone, peers, walls, fc, mesh_tgt, mp, jammed):
    """
    KAAL: Core mesh is fixed relative to fc. Scouts wander and report back.
    Awareness only spreads through LOS Chain.
    """
    if jammed: return

    # ── Role Specific Logic ───────────────────────────────
    if drone.role == 'mesh':
        # ── Combat Engagement & Revolving ──
        dist_m = np.linalg.norm(drone.pos - mp)
        if dist_m < PERC + 20 and los_clear(drone.pos, mp, walls):
            drone.atk = max(drone.atk, 15)
            # Update mesh target in real-time while engaging
            mesh_tgt[:] = mp.copy()
            
            # Orbit force: Stronger perpendicular movement
            to_m = mp - drone.pos
            d_m  = max(dist_m, 1.0)
            perp = np.array([-to_m[1], to_m[0]]) / d_m
            # Circle tightly around the target
            drone.acc += perp * 0.65 + (to_m / d_m) * 0.2
            
        # Seek slot relative to fc (formation center)
        goal = fc + drone.slot
        to_g = goal - drone.pos
        dg   = np.linalg.norm(to_g)
        if dg > 1.0:
            # ── MESH ELASTICITY ──
            # Significantly reduce slot pull when near walls to allow flow
            pull = 0.85
            for w in walls:
                _, dist = pt_seg(drone.pos, w.p1, w.p2)
                if dist < WALL_R * 1.2:
                    pull *= 0.25 
                    break
            drone.acc += (to_g / dg) * pull
    
    elif drone.role == 'scout':
        # If mesh is active, scouts 'get along' and move to support
        if np.any(mesh_tgt != 0):
            to_action = mesh_tgt - drone.pos
            da = np.linalg.norm(to_action)
            if da > PERC * 1.2:
                # Fly toward the action
                drone.acc += (to_action / max(da, 1.0)) * 0.45
            else:
                # Hover/Support near the target and REVOLVE if attacking
                if da < PERC and los_clear(drone.pos, mesh_tgt, walls):
                    drone.atk = max(drone.atk, 10)
                    to_m = mesh_tgt - drone.pos
                    d_m  = max(da, 1.0)
                    perp = np.array([-to_m[1], to_m[0]]) / d_m
                    drone.acc += perp * 0.5 + (to_m / d_m) * 0.15
                else:
                    drone.wander += random.uniform(-0.1, 0.1)
                    drone.acc += np.array([math.cos(drone.wander), math.sin(drone.wander)]) * 0.2
        
        elif drone.status == 'SEARCH':
            # Look for the mouse
            dist_m = np.linalg.norm(drone.pos - mp)
            if dist_m < PERC and los_clear(drone.pos, mp, walls):
                # Target Spotted! Remember STATIC location
                drone.seen_pos = mp.copy()
                drone.status   = 'REPORT'
            else:
                # Wander randomly
                drone.wander += random.uniform(-0.15, 0.15)
                drone.acc    += np.array([math.cos(drone.wander), math.sin(drone.wander)]) * 0.28
        
        elif drone.status == 'REPORT':
            # Fly back to formation center
            to_fc = fc - drone.pos
            dfc   = np.linalg.norm(to_fc)
            if dfc < PERC and los_clear(drone.pos, fc, walls):
                # REPORT IN! Mesh now knows where to go
                if drone.seen_pos is not None:
                    mesh_tgt[:] = drone.seen_pos
                drone.status = 'LEAD'
            else:
                drone.acc += (to_fc / max(dfc, 1.0)) * 0.65

        elif drone.status == 'LEAD':
            # Lead mesh to target, but transition back once near live target
            to_seen = drone.seen_pos - drone.pos
            ds      = np.linalg.norm(to_seen)
            if ds > 30:
                drone.acc += (to_seen / max(ds, 1.0)) * 0.6
            else:
                # Mission complete, resume scouting/searching
                drone.status = 'SEARCH'

    # ── GLOBAL PHYSICS (Must apply to ALL drones) ───────────
    apply_separation(drone, peers)
    apply_wall_repulsion(drone, walls)

def integrate(drone, max_spd, walls):
    """Velocity + position update with wall collision resolution"""
    drone.vel += drone.acc
    spd = np.linalg.norm(drone.vel)
    if spd > max_spd:
        drone.vel = drone.vel / spd * max_spd

    old = drone.pos.copy()
    nxt = drone.pos + drone.vel

    # ── Wall collision ───────────────────────────────────────
    for w in walls:
        hit, _ = seg_hit(old, nxt, w.p1, w.p2)
        if hit:
            # Reflect and damp; stay at old pos this frame
            drone.vel = reflect_vel(drone.vel, w.p1, w.p2) * 0.50
            nxt       = old      # do not pass the wall
            break

    drone.pos = nxt

    # ── Screen boundary bounce ───────────────────────────────
    M = 14
    if drone.pos[0] < M:
        drone.vel[0] = abs(drone.vel[0]);  drone.pos[0] = M
    elif drone.pos[0] > W - M:
        drone.vel[0] = -abs(drone.vel[0]); drone.pos[0] = W - M
    if drone.pos[1] < M:
        drone.vel[1] = abs(drone.vel[1]);  drone.pos[1] = M
    elif drone.pos[1] > H - M:
        drone.vel[1] = -abs(drone.vel[1]); drone.pos[1] = H - M

    drone.acc[:] = 0
    if drone.atk > 0:
        drone.atk -= 1


# ════════════════════════════════════════════════════════════
#  RENDERING
# ════════════════════════════════════════════════════════════
def draw_grid(surf):
    for x in range(0, W, 40):
        pygame.draw.line(surf, GRID_C, (x, 0), (x, H))
    for y in range(0, H, 40):
        pygame.draw.line(surf, GRID_C, (0, y), (W, y))

def draw_walls(surf, walls, preview_start, mpos):
    for w in walls:
        pygame.draw.line(surf, BLUE, w.p1.astype(int), w.p2.astype(int), 3)
        pygame.draw.circle(surf, BLUE, w.p1.astype(int), 4)
        pygame.draw.circle(surf, BLUE, w.p2.astype(int), 4)
    if preview_start:
        pygame.draw.line(surf, (55, 90, 190), preview_start, mpos, 2)

def draw_mesh_lines(surf, drones, walls):
    """Draws awareness chain links"""
    for d in drones:
        # Communication circle
        if d.knows:
            col = (0, 160, 180) if d.role == 'mesh' else (255, 140, 0)
        elif d.role == 'scout' and d.status == 'REPORT':
            col = (200, 180, 0)
        else:
            col = (25, 30, 35)
        pygame.draw.circle(surf, col, d.pos.astype(int), int(PERC), 1)

        # Draw chain links
        if d.knows:
            for o in drones:
                if o.knows and d.idx < o.idx:
                    dist = np.linalg.norm(d.pos - o.pos)
                    if dist < PERC and los_clear(d.pos, o.pos, walls):
                        pygame.draw.line(surf, (0, 220, 240), 
                                         d.pos.astype(int), o.pos.astype(int), 1)

def draw_target(surf, tpos, jammed, frame, mode):
    """Animated crosshair target marker (hidden in KAAL mode)"""
    if mode == 'kaal': return
    p   = tpos.astype(int)
    col = YELLOW if jammed else RED
    r   = int(10 + 5 * math.sin(frame * 0.13))

    pygame.draw.circle(surf, col, tuple(p), r, 1)
    pygame.draw.circle(surf, col, tuple(p), 5)

    for dx, dy in ((22, 0), (-22, 0), (0, 22), (0, -22)):
        start = (p[0] + dx // 4, p[1] + dy // 4)
        end   = (p[0] + dx,      p[1] + dy)
        pygame.draw.line(surf, col, start, end, 1)

def draw_attack_lines(surf, drones, tpos, jammed):
    """Orange lines from attacking drones to target"""
    p = tpos.astype(int)
    for d in drones:
        if d.atk > 0:
            # Clamp alpha to [0, 1] to avoid invalid color values (>255)
            alpha = min(1.0, d.atk / 12.0)
            col   = (int(ORANGE[0] * alpha),
                     int(ORANGE[1] * alpha),
                     int(ORANGE[2] * alpha))
            pygame.draw.line(surf, col, d.pos.astype(int), p, 1)

def draw_drone(surf, d, mode, mesh_tgt):
    """Tactical triangle with heading indicator"""
    angle = math.atan2(d.vel[1], d.vel[0])

    if d.atk > 0:
        col = ORANGE
    elif mode == 'normal':
        col = GREEN if d.knows else DG
    else:
        # KAAL coloring by role/status
        if d.role == 'mesh':
            col = CYAN if d.knows else DC
        elif d.role == 'scout':
            if d.status == 'REPORT': col = YELLOW
            elif d.status == 'LEAD': col = ORANGE
            else: col = (100, 100, 100) # Search
        else: col = (100, 100, 100)

    s  = 8
    p1 = d.pos + np.array([math.cos(angle),        math.sin(angle)])        * s
    p2 = d.pos + np.array([math.cos(angle + 2.45),  math.sin(angle + 2.45)]) * (s * 0.62)
    p3 = d.pos + np.array([math.cos(angle - 2.45),  math.sin(angle - 2.45)]) * (s * 0.62)

    pygame.draw.polygon(surf, col,
                        [p1.astype(int), p2.astype(int), p3.astype(int)])

def draw_ghosts(surf, ghosts):
    for g in ghosts:
        a = g.life
        c = (min(210, a + 40), 40, 40)
        x, y = g.pos.astype(int)
        pygame.draw.circle(surf, c, (x, y), 6, 1)
        pygame.draw.line(surf, c, (x-5, y-5), (x+5, y+5), 1)
        pygame.draw.line(surf, c, (x+5, y-5), (x-5, y+5), 1)

def draw_emp_cursor(surf, mp):
    s = pygame.Surface((KILL_R * 2 + 4, KILL_R * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(s, (255, 50, 50, 35), (KILL_R+2, KILL_R+2), KILL_R, 1)
    pygame.draw.circle(s, (255, 50, 50,  9), (KILL_R+2, KILL_R+2), KILL_R)
    surf.blit(s, (mp[0] - KILL_R - 2, mp[1] - KILL_R - 2))

def draw_kaal_extras(surf, fc, frame):
    """Pulsing tactical circle around formation center"""
    p = fc.astype(int)
    r = 30 + int(8 * math.sin(frame * 0.15))
    pygame.draw.circle(surf, (0, 40, 60), p, r, 1)
    
    # Tiny dots at compass points
    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
        dx = int(math.cos(angle) * r)
        dy = int(math.sin(angle) * r)
        pygame.draw.circle(surf, CYAN, (p[0] + dx, p[1] + dy), 2)

def draw_hud(surf, drones, casualties, mode, jammed, f_big, f_sm):
    hc    = CYAN if mode == 'kaal' else GREEN
    hud_h = 135 if jammed else 102

    # Semi-transparent background
    s = pygame.Surface((235, hud_h), pygame.SRCALPHA)
    s.fill((*hc, 14))
    surf.blit(s, (8, 8))
    pygame.draw.rect(surf, (*hc, 50), (8, 8, 235, hud_h), 1)

    mode_label = '◆ KAAL MESH PROTOCOL' if mode == 'kaal' else '◇ STANDARD OPS'
    surf.blit(f_big.render(mode_label, True, hc),              (16, 14))
    surf.blit(f_sm.render(f'ACTIVE: {len(drones)}    KIA: {casualties}',
                          True, (100, 140, 100)),              (16, 33))

    # Mode Switch Button (Visual)
    sw_rect = pygame.Rect(16, 48, 105, 22)
    pygame.draw.rect(surf, (0, 30, 15), sw_rect)
    pygame.draw.rect(surf, hc, sw_rect, 1)
    surf.blit(f_sm.render('SWITCH MODE [M]', True, hc), (22, 54))

    # Respawn Button (Visual)
    re_rect = pygame.Rect(125, 48, 105, 22)
    pygame.draw.rect(surf, (30, 15, 0), re_rect)
    pygame.draw.rect(surf, ORANGE, re_rect, 1)
    surf.blit(f_sm.render('RESPAWN +10 [N]', True, ORANGE), (131, 54))

    surf.blit(f_sm.render('M:MODE  N:RESPAWN  J:JAMMER  C:CLEAR  R:RESET',
                          True, (60, 90, 60)),                 (16, 80))

    if jammed:
        js = pygame.Surface((215, 30), pygame.SRCALPHA)
        js.fill((255, 200, 0, 18))
        surf.blit(js, (12, 100))
        pygame.draw.rect(surf, (255, 200, 0, 45), (12, 100, 215, 30), 1)
        surf.blit(f_big.render('⚡ COMMS JAMMED', True, YELLOW), (16, 105))
        surf.blit(f_sm.render('LOCKED TO LAST KNOWN POSITION', True, (130, 100, 0)), (16, 119))


def start_screen(surf, inp, f_title, f_big, f_sm, frame):
    surf.fill(BG)
    draw_grid(surf)

    # Scanlines
    scan = pygame.Surface((W, H), pygame.SRCALPHA)
    for y in range(0, H, 3):
        pygame.draw.line(scan, (0, 0, 0, 18), (0, y), (W, y))
    surf.blit(scan, (0, 0))

    # Title
    t1 = f_title.render('▲  KAAL  SWARM  KINEMATICS', True, GREEN)
    surf.blit(t1, (W // 2 - t1.get_width() // 2, H // 2 - 148))

    t2 = f_big.render('AUTONOMOUS MESH DRONE TACTICAL ENGINE  v3.0', True, (0, 75, 38))
    surf.blit(t2, (W // 2 - t2.get_width() // 2, H // 2 - 106))

    pygame.draw.line(surf, (0, 45, 22),
                     (W // 2 - 320, H // 2 - 82),
                     (W // 2 + 320, H // 2 - 82))

    # ── Drone count input ─────────────────────────────────────
    lab = f_big.render('DRONE COUNT :', True, (0, 155, 75))
    surf.blit(lab, (W // 2 - 160, H // 2 - 46))

    ir = pygame.Rect(W // 2 + 20, H // 2 - 52, 115, 38)
    pygame.draw.rect(surf, (8, 17, 8), ir)
    pygame.draw.rect(surf, (0, 195, 78), ir, 1)
    blink = (pygame.time.get_ticks() // 500) % 2
    surf.blit(f_big.render(inp + ('_' if blink else ''), True, GREEN),
              (ir.x + 10, ir.y + 10))

    # ── Deploy button ─────────────────────────────────────────
    dr = pygame.Rect(W // 2 - 95, H // 2 + 16, 190, 46)
    pygame.draw.rect(surf, (0, 28, 12), dr)
    pygame.draw.rect(surf, GREEN, dr, 1)
    dt = f_big.render('▶   DEPLOY SWARM', True, GREEN)
    surf.blit(dt, (dr.x + (190 - dt.get_width()) // 2, dr.y + 13))

    # ── Info block ────────────────────────────────────────────
    rows = [
        ('NORMAL MODE', '→  Each drone seeks target independently — walls block LOS, lost drones wander'),
        ('KAAL MODE',   '→  Mesh grid formation — one LOS = all know; formation auto-heals after losses'),
        ('CONTROLS',   '→  LEFT DRAG: wall    RIGHT CLICK: EMP    M: mode    J: jammer    C: clear'),
    ]
    for i, (lbl, txt) in enumerate(rows):
        y = H // 2 + 98 + i * 26
        surf.blit(f_sm.render(lbl, True, (0, 120, 58)),  (W // 2 - 300, y))
        surf.blit(f_sm.render(txt, True, (0,  55, 32)),  (W // 2 - 210, y))

    return dr   # caller checks click against this rect


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('KAAL SWARM v3 — TACTICAL SIM')
    clock = pygame.time.Clock()

    f_title = pygame.font.SysFont('Courier New', 30, bold=True)
    f_big   = pygame.font.SysFont('Courier New', 13, bold=True)
    f_sm    = pygame.font.SysFont('Courier New', 10)

    # ── State ────────────────────────────────────────────────
    scene      = 'start'
    inp        = '40'
    drones     = []
    walls      = []
    ghosts     = []
    casualties = 0
    mode       = 'normal'
    jammed     = False
    tgt        = np.array([W - 160.0, H / 2.0])
    lk         = tgt.copy()   
    fc         = tgt.copy()   
    mesh_tgt   = np.array([0.0, 0.0]) # Mesh memory
    wall_start = None          
    frame      = 0

    # ── Deploy helper ────────────────────────────────────────
    def deploy(n):
        nonlocal drones, walls, ghosts, casualties
        nonlocal mode, jammed, tgt, lk, fc, scene, mesh_tgt
        n      = max(5, min(150, n))
        drones = [Drone(55 + random.random() * 130,
                        25 + random.random() * (H - 50), i) for i in range(n)]
        
        # Initial slots for Normal mode (random/wandering start)
        for d in drones:
            d.slot = np.zeros(2)
            d.is_mesh = False

        # Calculate initial swarm center for fc
        avg_x = sum(d.pos[0] for d in drones) / n
        avg_y = sum(d.pos[1] for d in drones) / n
        
        walls      = []
        ghosts     = []
        casualties = 0
        mode       = 'normal'
        jammed     = False
        tgt        = np.array([W - 160.0, H / 2.0])
        lk         = tgt.copy()
        fc         = np.array([avg_x, avg_y]) # Swarm center
        mesh_tgt   = np.array([0.0, 0.0])
        scene      = 'sim'

    def respawn(count=10):
        nonlocal drones
        if len(drones) >= 200: return
        start_idx = len(drones)
        for i in range(count):
            d = Drone(55 + random.random() * 100,
                      25 + random.random() * (H - 50), start_idx + i)
            drones.append(d)
        if mode == 'kaal':
            slots_data = make_slots_v4(len(drones))
            for d, sd in zip(drones, slots_data):
                d.role = sd['role']; d.slot = sd['slot']; d.grid_pos = sd['grid']

    def setup_kaal():
        nonlocal mode, mesh_tgt, fc
        mode = 'kaal'
        mesh_tgt[:] = 0.0
        slots_data = make_slots_v4(len(drones))
        for d, sd in zip(drones, slots_data):
            d.role = sd['role']
            d.slot = sd['slot']
            d.grid_pos = sd['grid']
            d.status = 'SEARCH'
            d.seen_pos = None

        mesh_drones = [d for d in drones if d.role == 'mesh']
        if mesh_drones:
            avg_x = sum(d.pos[0] for d in mesh_drones) / len(mesh_drones)
            avg_y = sum(d.pos[1] for d in mesh_drones) / len(mesh_drones)
            fc[:] = [avg_x, avg_y]

    # ── Main loop ────────────────────────────────────────────
    while True:
        clock.tick(FPS)
        frame += 1
        mp  = np.array(pygame.mouse.get_pos(), dtype=float)
        mpi = (int(mp[0]), int(mp[1]))

        # ════ Events ════════════════════════════════════════
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # ── Start screen input ───────────────────────
            if scene == 'start':
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_BACKSPACE:
                        inp = inp[:-1]
                    elif ev.unicode.isdigit() and len(inp) < 3:
                        inp += ev.unicode
                    elif ev.key == pygame.K_RETURN:
                        deploy(int(inp) if inp else 40)

                if ev.type == pygame.MOUSEBUTTONDOWN:
                    deploy_rect = pygame.Rect(W // 2 - 95, H // 2 + 16, 190, 46)
                    if deploy_rect.collidepoint(ev.pos):
                        deploy(int(inp) if inp else 40)

            # ── Simulation input ─────────────────────────
            elif scene == 'sim':
                if ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_m:
                        if mode == 'normal': setup_kaal()
                        else: mode = 'normal'

                    elif k == pygame.K_n:
                        respawn(10)

                    elif k == pygame.K_j:
                        jammed = not jammed
                        if not jammed:
                            for d in drones:
                                d.knows = False

                    elif k == pygame.K_c:
                        walls = []

                    elif k in (pygame.K_r, pygame.K_ESCAPE):
                        scene = 'start'
                        inp   = '40'

                if ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1:          # left: check HUD or draw wall
                        # HUD Buttons Check
                        sw_rect = pygame.Rect(16, 48, 100, 20)
                        re_rect = pygame.Rect(124, 48, 100, 20)
                        if sw_rect.collidepoint(ev.pos):
                            if mode == 'normal': setup_kaal()
                            else: mode = 'normal'
                        elif re_rect.collidepoint(ev.pos):
                            respawn(10)
                        else:
                            wall_start = mpi
                    elif ev.button == 3:        # right: EMP blast
                        survivors = []
                        for d in drones:
                            if np.linalg.norm(d.pos - mp) < KILL_R:
                                ghosts.append(Ghost(d.pos))
                                casualties += 1
                            else:
                                survivors.append(d)
                        if len(survivors) < len(drones):
                            drones = survivors
                            if mode == 'kaal':
                                slots_data = make_slots_v4(len(drones))
                                for d, sd in zip(drones, slots_data):
                                    d.role = sd['role']; d.slot = sd['slot']; d.grid_pos = sd['grid']

                if ev.type == pygame.MOUSEBUTTONUP:
                    if ev.button == 1 and wall_start:
                        p1 = np.array(wall_start, dtype=float)
                        p2 = mp.copy()
                        if np.linalg.norm(p2 - p1) > 15:
                            walls.append(Wall(p1, p2))
                        wall_start = None

        # ════ Simulation tick ════════════════════════════════
        if scene == 'sim':

            # Target follows mouse (unless jammed)
            if not jammed:
                tgt = mp.copy()
                lk  = tgt.copy()

            # ── KAAL CHAIN PROPAGATION ──────────────────────────
            if mode == 'kaal':
                # 1. Reset awareness
                for d in drones:
                    d.knows = False
                
                # 2. Identify Sources: Anyone who can SEE the cursor directly
                # (Scouts in LEAD mode, or any mesh drone with LOS)
                sources = [d for d in drones if d.role == 'scout' and d.status == 'LEAD']
                for d in drones:
                    if d.role == 'mesh' and los_clear(d.pos, mp, walls) and np.linalg.norm(d.pos - mp) < PERC:
                        sources.append(d)
                
                for s in sources: 
                    s.knows = True
                
                # 3. Spread via BFS Chain (LOS + PERC)
                queue = list(sources)
                idx = 0
                while idx < len(queue):
                    curr = queue[idx]
                    idx += 1
                    for o in drones:
                        if not o.knows:
                            dist = np.linalg.norm(curr.pos - o.pos)
                            if dist < PERC and los_clear(curr.pos, o.pos, walls):
                                o.knows = True
                                queue.append(o)
                
                # 4. Check mesh awareness
                informed_mesh = [d for d in drones if d.role == 'mesh' and d.knows]
                if not informed_mesh:
                    # If mesh doesn't have an active leader, target is lost
                    mesh_tgt[:] = 0.0

            # Formation centre tracks mesh_tgt (with SMART wall sliding)
            if mode == 'kaal':
                if np.any(mesh_tgt != 0):
                    to_mt  = mesh_tgt - fc
                    dist_k = np.linalg.norm(to_mt)
                    
                    # ── TARGET VERIFICATION ──
                    # If we reached the spot but no one sees the target anymore, it's gone.
                    if dist_k < 20:
                        anyone_sees = any(los_clear(d.pos, mp, walls) and np.linalg.norm(d.pos - mp) < PERC for d in drones)
                        if not anyone_sees:
                            mesh_tgt[:] = 0.0 # Clear memory, go back to looking
                    
                    if dist_k > 2.0:
                        # ── LOOKAHEAD AVOIDANCE ──
                        # Steer away from walls before hitting them
                        steering = (to_mt / dist_k) * min(dist_k * 0.08, MAX_K * 1.6)
                        lookahead_pos = fc + steering * 10.0
                        for w in walls:
                            cp, dist = pt_seg(lookahead_pos, w.p1, w.p2)
                            if dist < WALL_R * 1.5:
                                # Push steering vector away from wall
                                away = lookahead_pos - cp
                                steering += (away / max(dist, 1.0)) * 0.5
                        
                        step = steering
                        new_fc = fc + step
                        collision_wall = None
                        for w in walls:
                            if seg_hit(fc, new_fc, w.p1, w.p2)[0]:
                                collision_wall = w; break
                        
                        if not collision_wall:
                            fc[:] = new_fc
                        else:
                            # ── ADVANCED SLIDING ──
                            wv = collision_wall.p2 - collision_wall.p1
                            u  = wv / (np.linalg.norm(wv) or 1.0)
                            s1 = np.dot(step, u) * u
                            s2 = -s1
                            best_s = s1 if np.linalg.norm(fc+s1-mesh_tgt) < np.linalg.norm(fc+s2-mesh_tgt) else s2
                            
                            n = np.array([-u[1], u[0]])
                            if np.dot(n, to_mt) < 0: n = -n
                            best_s += n * 1.2 # Stronger nudge away from sticky corner

                            test_fc = fc + best_s
                            blocked = False
                            for w in walls:
                                if seg_hit(fc, test_fc, w.p1, w.p2)[0]:
                                    blocked = True; break
                            if not blocked:
                                fc[:] = test_fc

            # Update each drone
            for d in drones:
                if mode == 'normal':
                    ai_normal(d, drones, walls, tgt if not jammed else lk)
                    integrate(d, MAX_N, walls)
                else:
                    ai_kaal(d, drones, walls, fc, mesh_tgt, mp, jammed)
                    integrate(d, MAX_K, walls)

                # Attack range check (normal mode)
                if mode == 'normal':
                    if los_clear(d.pos, tgt if not jammed else lk, walls):
                        dist_tgt = np.linalg.norm(d.pos - (tgt if not jammed else lk))
                        if dist_tgt < ATK_RANGE:
                            d.atk = max(d.atk, 12)

            # Ghost decay
            for g in ghosts:
                g.life -= 5
            ghosts = [g for g in ghosts if g.life > 0]

            # Telemetry to stdout (for Operator 3 / dashboard)
            status = 'JAMMED' if jammed else ('ENGAGING' if any(d.atk > 0 for d in drones) else 'NOMINAL')
            print(json.dumps({
                'status': status,
                'alive':  len(drones),
                'kia':    casualties,
                'mode':   mode.upper()
            }), flush=True)

        # ════ Render ═════════════════════════════════════════
        screen.fill(BG)

        if scene == 'start':
            start_screen(screen, inp, f_title, f_big, f_sm, frame)

        else:   # sim
            draw_grid(screen)

            if mode == 'kaal':
                draw_mesh_lines(screen, drones, walls)

            draw_walls(screen, walls, wall_start, mpi)

            effective_tgt = tgt if not jammed else lk
            draw_target(screen, effective_tgt, jammed, frame, mode)
            
            # Show attack lines in both modes (Normal uses effective target, KAAL uses live mp)
            if mode == 'normal':
                draw_attack_lines(screen, drones, effective_tgt, jammed)
            else:
                # In KAAL mode, attack the actual mouse cursor (mp)
                draw_attack_lines(screen, drones, mp, jammed)

            for d in drones:
                draw_drone(screen, d, mode, mesh_tgt)

            draw_ghosts(screen, ghosts)
            draw_emp_cursor(screen, mpi)
            draw_hud(screen, drones, casualties, mode, jammed, f_big, f_sm)

        pygame.display.flip()


if __name__ == '__main__':
    main()