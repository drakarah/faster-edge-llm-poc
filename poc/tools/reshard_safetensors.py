#!/usr/bin/env python3
"""Re-shard a safetensors checkpoint into ~N GB shards without mmap (plain file reads).
Why: the 49.9 GB shard cannot be mmap'd on a 31 GB-RAM box (heuristic overcommit refuses, 'Cannot allocate memory').
Usage: reshard_safetensors.py SRC_DIR DST_DIR [GB=2]"""
import json, struct, sys, os, shutil
src, dst = sys.argv[1], sys.argv[2]; lim = float(sys.argv[3] if len(sys.argv) > 3 else 2) * 2**30
os.makedirs(dst, exist_ok=True)
ix = json.load(open(f'{src}/model.safetensors.index.json'))
if all(os.path.getsize(f'{src}/{sh}') <= lim for sh in set(ix['weight_map'].values())):   # already small shards (a seeded copy): hard-link instead of copying 49 GB again
    for f in os.listdir(src):
        if f.endswith('.safetensors'): os.link(f'{src}/{f}', f'{dst}/{f}')
        else: shutil.copy(f'{src}/{f}', dst)
    print('RESHARD_DONE: shards already small, hard-linked'); sys.exit(0)
for f in os.listdir(src):
    if not f.endswith('.safetensors') and f != 'model.safetensors.index.json': shutil.copy(f'{src}/{f}', dst)
tens = []  # (name, shard, dtype, shape, off0, off1, base)
for sh in sorted(set(ix['weight_map'].values())):
    with open(f'{src}/{sh}', 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n)); h.pop('__metadata__', None)
    for k, v in h.items(): tens.append((k, sh, v['dtype'], v['shape'], v['data_offsets'][0], v['data_offsets'][1], 8 + n))
groups, cur, sz = [], [], 0
for t in tens:
    s = t[5] - t[4]
    if cur and sz + s > lim: groups.append(cur); cur, sz = [], 0
    cur.append(t); sz += s
if cur: groups.append(cur)
wm = {}
for i, g in enumerate(groups):
    name = f'model-{i+1:05d}-of-{len(groups):05d}.safetensors'; hdr, off = {'__metadata__': {'format': 'pt'}}, 0
    for k, sh, dt, shp, a, b, base in g: hdr[k] = {'dtype': dt, 'shape': shp, 'data_offsets': [off, off + b - a]}; off += b - a; wm[k] = name
    hb = json.dumps(hdr).encode(); hb += b' ' * (-len(hb) % 8)
    with open(f'{dst}/{name}', 'wb') as out:
        out.write(struct.pack('<Q', len(hb))); out.write(hb)
        for k, sh, dt, shp, a, b, base in g:
            with open(f'{src}/{sh}', 'rb') as fh:
                fh.seek(base + a); left = b - a
                while left: c = fh.read(min(left, 64 << 20)); out.write(c); left -= len(c)
    print(name, len(g), 'tensors', flush=True)
ix['weight_map'] = wm; json.dump(ix, open(f'{dst}/model.safetensors.index.json', 'w'))
print('RESHARD_DONE', len(tens), 'tensors', len(groups), 'shards')
