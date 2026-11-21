// Sokoban solver: split the level into its hallway rooms, solve each room
// push-optimally with A* over pushes, then stitch the walks together.
// Original work. No clock, no randomness: every budget is counted in states,
// so two runs output byte-identical files.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions: up, down, left, right. Fixed order everywhere, for determinism.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const DL = ['u', 'd', 'l', 'r'];

// Work budgets, counted in expanded states (never wall-clock time).
const BUDGET = { used: 0, limit: 3_000_000 };
const ROOM_CAP = 400_000;
const FALLBACK_CAP = 600_000;

type Grid = {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
};

function parse(text: string): Grid {
  const rows = text.split('\n').map((r) => r.replace(/\r$/, '')).filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// Shortest walking distances from src over floor cells (walls only).
function bfsDist(w: number, h: number, floor: Uint8Array, src: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  const q: number[] = [src];
  dist[src] = 0;
  for (let i = 0; i < q.length; i++) {
    const c = q[i]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!floor[n] || dist[n] >= 0) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

// Cells the player reaches from `from` over floor cells, avoiding `blocked`.
// Returns the reached set as a mask plus its smallest cell (a canonical
// representative of the region: same boxes + same region => same future).
function regionOf(w: number, h: number, floor: Uint8Array, blocked: Set<number>, from: number): { mask: Uint8Array; min: number; cells: number[] } {
  const mask = new Uint8Array(w * h);
  const cells: number[] = [from];
  mask[from] = 1;
  let min = from;
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!floor[n] || mask[n] || blocked.has(n)) continue;
      mask[n] = 1;
      if (n < min) min = n;
      cells.push(n);
    }
  }
  return { mask, min, cells };
}

// Shortest walk from `from` to the first cell satisfying `isTarget`, over
// floor cells avoiding `blocked`. Returns the move letters, or null.
function walkPath(
  w: number,
  h: number,
  floor: Uint8Array,
  blocked: Set<number>,
  from: number,
  isTarget: (c: number) => boolean,
): string | null {
  if (isTarget(from)) return '';
  const prev = new Int32Array(w * h).fill(-1);
  const prevDir = new Int8Array(w * h).fill(-1);
  const q: number[] = [from];
  prev[from] = from;
  let found = -1;
  for (let i = 0; i < q.length && found < 0; i++) {
    const c = q[i]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d]!;
      const ny = cy + DY[d]!;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!floor[n] || prev[n] >= 0 || blocked.has(n)) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (isTarget(n)) {
        found = n;
        break;
      }
      q.push(n);
    }
  }
  if (found < 0) return null;
  let out = '';
  for (let c = found; c !== from; c = prev[c]!) out = DL[prevDir[c]!]! + out;
  return out;
}

// All permutations of 0..n-1 (n is 3 or 4 here).
function perms(n: number): number[][] {
  const out: number[][] = [];
  const a: number[] = [];
  for (let i = 0; i < n; i++) a.push(i);
  const rec = (k: number): void => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      const t = a[k]!;
      a[k] = a[i]!;
      a[i] = t;
      rec(k + 1);
      const t2 = a[k]!;
      a[k] = a[i]!;
      a[i] = t2;
    }
  };
  rec(0);
  return out;
}

type SearchOpts = {
  w: number;
  h: number;
  floor: Uint8Array; // cells the player may walk on
  boxCells: Uint8Array; // cells a box may occupy
  goals: number[];
  boxes0: number[];
  player0: number;
  needExit: boolean; // player must be able to reach the hallway afterwards
  hy: number; // hallway row (only used when needExit)
  cap: number; // max states expanded by this search
};

type Push = { from: number; dir: number };

// Push-optimal A* over push-states. A state is (sorted boxes, player region).
// Heuristic: min-cost bipartite matching of boxes to goals on box-blind
// shortest-path distances (admissible: one push moves one box one step).
// Returns the push list and the player's cell after the last push.
function pushSearch(o: SearchOpts): { pushes: Push[]; endPlayer: number } | null {
  const { w, h, floor, boxCells, goals, boxes0, player0, needExit, hy, cap } = o;
  const size = w * h;
  const n = boxes0.length;
  if (n === 0) return { pushes: [], endPlayer: player0 };
  if (goals.length !== n) return null;
  const goalSet = new Set(goals);
  const distFromGoal = goals.map((g) => bfsDist(w, h, floor, g));

  // Static dead squares: box positions from which no goal is reachable by
  // pushes (reverse-push BFS from the goals; a push needs a free cell for
  // the player on the opposite side). A box there can never be saved.
  const alive = new Uint8Array(size);
  const stack: number[] = [];
  for (const g of goals) {
    alive[g] = 1;
    stack.push(g);
  }
  while (stack.length > 0) {
    const c = stack.pop()!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const px = cx - DX[d]!;
      const py = cy - DY[d]!;
      const sx = cx - 2 * DX[d]!;
      const sy = cy - 2 * DY[d]!;
      if (px < 0 || py < 0 || px >= w || py >= h) continue;
      if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
      const p = py * w + px;
      const s = sy * w + sx;
      if (!boxCells[p] || !floor[s] || alive[p]) continue;
      alive[p] = 1;
      stack.push(p);
    }
  }
  for (const b of boxes0) if (!alive[b]) return null;

  const P = perms(n);
  const hCache = new Map<string, number>();
  const heuristic = (bs: number[]): number => {
    const key = bs.join(',');
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let best = Infinity;
    for (const pm of P) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = distFromGoal[pm[i]!]![bs[i]!]!;
        if (d < 0) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    hCache.set(key, best);
    return best;
  };

  const h0 = heuristic(boxes0);
  if (!isFinite(h0)) return null;

  // A* state storage (parallel arrays).
  const stBoxes: number[][] = [boxes0.slice()];
  const stPlayer: number[] = [player0];
  const stG: number[] = [0];
  const stParent: number[] = [-1];
  const stPush: number[] = [-1]; // from * 4 + dir
  const stKey: string[] = [''];
  const best = new Map<string, number>();
  const startRegion = regionOf(w, h, floor, new Set(boxes0), player0);
  stKey[0] = boxes0.join(',') + '|' + startRegion.min;
  best.set(stKey[0], 0);

  // Binary heap on (f, h, seq): deterministic tie-breaking.
  const heap: number[][] = [];
  let seq = 0;
  const less = (a: number[], b: number[]): boolean =>
    a[0]! < b[0]! || (a[0]! === b[0]! && (a[1]! < b[1]! || (a[1]! === b[1]! && a[2]! < b[2]!)));
  const heapPush = (e: number[]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): number[] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, h0, seq++, 0]);

  let expanded = 0;
  let goalState = -1;
  while (heap.length > 0) {
    const [, , , s] = heapPop();
    const bs = stBoxes[s]!;
    const g = stG[s]!;
    if (best.get(stKey[s]!) !== g) continue; // stale entry
    // Goal test on pop: with a consistent heuristic the first popped goal
    // state is push-optimal.
    let solved = true;
    for (const b of bs) if (!goalSet.has(b)) { solved = false; break; }
    if (solved) {
      if (!needExit) {
        goalState = s;
        break;
      }
      const reg = regionOf(w, h, floor, new Set(bs), stPlayer[s]!);
      for (const c of reg.cells) {
        if (((c / w) | 0) === hy) {
          goalState = s;
          break;
        }
      }
      if (goalState >= 0) break;
    }
    if (expanded >= cap || BUDGET.used >= BUDGET.limit) return null;
    expanded++;
    BUDGET.used++;
    const p = stPlayer[s]!;
    const blocked = new Set(bs);
    const reg = regionOf(w, h, floor, blocked, p);
    for (let bi = 0; bi < n; bi++) {
      const b = bs[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = bx + DX[d]!;
        const ny = by + DY[d]!;
        const px = bx - DX[d]!;
        const py = by - DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        if (px < 0 || py < 0 || px >= w || py >= h) continue;
        const dest = ny * w + nx;
        const pusher = py * w + px;
        if (!boxCells[dest] || blocked.has(dest)) continue;
        if (!reg.mask[pusher]) continue;
        if (!alive[dest]) continue; // static dead square
        const nb = bs.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        const h1 = heuristic(nb);
        if (!isFinite(h1)) continue;
        const reg2 = regionOf(w, h, floor, new Set(nb), b);
        const key = nb.join(',') + '|' + reg2.min;
        const old = best.get(key);
        if (old !== undefined && old <= g + 1) continue;
        best.set(key, g + 1);
        const t = stBoxes.length;
        stBoxes.push(nb);
        stPlayer.push(b);
        stG.push(g + 1);
        stParent.push(s);
        stPush.push(b * 4 + d);
        stKey.push(key);
        heapPush([g + 1 + h1, h1, seq++, t]);
      }
    }
  }
  if (goalState < 0) return null;
  const pushes: Push[] = [];
  for (let t = goalState; t > 0; t = stParent[t]!) {
    const e = stPush[t]!;
    pushes.push({ from: (e / 4) | 0, dir: e % 4 });
  }
  pushes.reverse();
  return { pushes, endPlayer: stPlayer[goalState]! };
}

// Turn a push list into moves: walk to each pushing square, then push.
// Afterwards, unless this is the last room, walk out to the hallway.
function buildMoves(
  w: number,
  h: number,
  floor: Uint8Array,
  boxes0: number[],
  player0: number,
  pushes: Push[],
  needExit: boolean,
  hy: number,
): { moves: string; endPlayer: number } | null {
  let moves = '';
  let player = player0;
  const boxes = boxes0.slice();
  for (const { from, dir } of pushes) {
    const bx = from % w;
    const by = (from / w) | 0;
    const pusher = (by - DY[dir]!) * w + (bx - DX[dir]!);
    const walk = walkPath(w, h, floor, new Set(boxes), player, (c) => c === pusher);
    if (walk === null) return null;
    moves += walk + DL[dir]!;
    boxes[boxes.indexOf(from)] = from + DY[dir]! * w + DX[dir]!;
    boxes.sort((a, b) => a - b);
    player = from;
  }
  if (needExit) {
    const walk = walkPath(w, h, floor, new Set(boxes), player, (c) => ((c / w) | 0) === hy);
    if (walk === null) return null;
    moves += walk;
    // Player ends on a hallway cell; recompute by replaying the walk.
    let p = player;
    for (const ch of walk) {
      const d = DL.indexOf(ch);
      p += DY[d]! * w + DX[d]!;
    }
    player = p;
  }
  return { moves, endPlayer: player };
}

function solveLevel(text: string): string {
  const g = parse(text);
  const { w, h, wall, goal, boxes, player } = g;
  const size = w * h;
  const floor = new Uint8Array(size);
  for (let i = 0; i < size; i++) floor[i] = wall[i] ? 0 : 1;
  const hy = (player / w) | 0;

  // Rooms: connected components of the floor once the hallway row is out.
  const seen = new Uint8Array(size);
  type Room = { mask: Uint8Array; cells: number[]; doors: number[]; boxes: number[]; goals: number[] };
  const rooms: Room[] = [];
  for (let i = 0; i < size; i++) {
    if (!floor[i] || ((i / w) | 0) === hy || seen[i]) continue;
    const cells: number[] = [i];
    seen[i] = 1;
    for (let k = 0; k < cells.length; k++) {
      const c = cells[k]!;
      const cx = c % w;
      const cy = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d]!;
        const ny = cy + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        if (ny === hy) continue;
        const n = ny * w + nx;
        if (!floor[n] || seen[n]) continue;
        seen[n] = 1;
        cells.push(n);
      }
    }
    const mask = new Uint8Array(size);
    for (const c of cells) mask[c] = 1;
    const doors: number[] = [];
    const rBoxes: number[] = [];
    const rGoals: number[] = [];
    for (const c of cells) {
      const cx = c % w;
      const cy = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const ny = cy + DY[d]!;
        if (ny !== hy) continue;
        const nx = cx + DX[d]!;
        if (nx < 0 || nx >= w) continue;
        const n = ny * w + nx;
        if (floor[n] && !doors.includes(c)) doors.push(c);
      }
      if (boxes.includes(c)) rBoxes.push(c);
      if (goal[c]) rGoals.push(c);
    }
    rBoxes.sort((a, b) => a - b);
    rGoals.sort((a, b) => a - b);
    rooms.push({ mask, cells, doors, boxes: rBoxes, goals: rGoals });
  }

  const hallway = new Uint8Array(size);
  for (let i = 0; i < size; i++) if (floor[i] && ((i / w) | 0) === hy) hallway[i] = 1;

  // Decomposition is usable only if every box and goal sits in a room,
  // each room holds as many goals as boxes, and each room has a door.
  let decomposed = rooms.length > 0;
  const assignedBoxes = rooms.reduce((a, r) => a + r.boxes.length, 0);
  const assignedGoals = rooms.reduce((a, r) => a + r.goals.length, 0);
  let totalGoals = 0;
  for (let i = 0; i < size; i++) if (goal[i]) totalGoals++;
  if (assignedBoxes !== boxes.length || assignedGoals !== totalGoals) decomposed = false;
  for (const r of rooms) {
    if (r.boxes.length !== r.goals.length || r.doors.length === 0) decomposed = false;
  }

  let moves = '';
  if (decomposed) {
    // Player's own room first (if the player starts inside one), then by door.
    rooms.sort((a, b) => {
      const ap = a.mask[player] ? 0 : 1;
      const bp = b.mask[player] ? 0 : 1;
      if (ap !== bp) return ap - bp;
      return Math.min(...a.doors) - Math.min(...b.doors);
    });
    let playerPos = player;
    let ok = true;
    for (let ri = 0; ri < rooms.length; ri++) {
      const r = rooms[ri]!;
      const last = ri === rooms.length - 1;
      if (r.boxes.length === 0) continue;
      const roomFloor = new Uint8Array(size);
      for (let i = 0; i < size; i++) roomFloor[i] = r.mask[i] || hallway[i] ? 1 : 0;
      const res = pushSearch({
        w,
        h,
        floor: roomFloor,
        boxCells: r.mask,
        goals: r.goals,
        boxes0: r.boxes,
        player0: playerPos,
        needExit: !last,
        hy,
        cap: ROOM_CAP,
      });
      if (res === null) {
        ok = false;
        break;
      }
      const built = buildMoves(w, h, roomFloor, r.boxes, playerPos, res.pushes, !last, hy);
      if (built === null) {
        ok = false;
        break;
      }
      moves += built.moves;
      playerPos = built.endPlayer;
    }
    if (!ok) moves = '';
  }

  if (moves === '') {
    // Fallback: one push-optimal A* over the whole level.
    const allGoals: number[] = [];
    for (let i = 0; i < size; i++) if (goal[i]) allGoals.push(i);
    const res = pushSearch({
      w,
      h,
      floor,
      boxCells: floor,
      goals: allGoals,
      boxes0: boxes,
      player0: player,
      needExit: false,
      hy,
      cap: FALLBACK_CAP,
    });
    if (res !== null) {
      const built = buildMoves(w, h, floor, boxes, player, res.pushes, false, hy);
      if (built !== null) moves = built.moves;
    }
  }

  if (moves === '') return '';
  // Validate by replaying: never output an illegal or unsolved answer.
  let p = player;
  const bset = new Set(boxes);
  for (const ch of moves) {
    const d = DL.indexOf(ch);
    if (d < 0) return '';
    const q = p + DY[d]! * w + DX[d]!;
    if (wall[q]) return '';
    if (bset.has(q)) {
      const beyond = q + DY[d]! * w + DX[d]!;
      if (wall[beyond] || bset.has(beyond)) return '';
      bset.delete(q);
      bset.add(beyond);
    }
    p = q;
  }
  for (const b of bset) if (!goal[b]) return '';
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
