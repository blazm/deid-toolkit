import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from analyze_linkability import discover_techniques, load_embeddings

MAP_FILE = r"D:\dev\deid-toolkit\root_dir\datasets\original\celeba-test\aligned_stem_identity_map.csv"
idmap = {}
with open(MAP_FILE, encoding="utf-8") as f:
    next(f)
    for line in f:
        q = line.rstrip("\n").split("\t")
        if len(q) == 3 and q[2].strip():
            idmap[q[0]] = int(q[2])

for model, root in [("swinface", "D:/dev/deid-toolkit/root_dir/embeddings/SwinFace"),
                    ("transface", "D:/dev/deid-toolkit/root_dir/embeddings/TransFace")]:
    root = Path(root)
    aligned_emb, aligned_keys = load_embeddings(root / "aligned" / "celeba-test")
    aligned_mat = np.stack([aligned_emb[k] for k in aligned_keys])
    gal_idx = {k: i for i, k in enumerate(aligned_keys)}
    out = Path(f"D:/dev/deid-toolkit/root_dir/predictions/ranks_{model}_per_identity.csv")
    with open(out, "w") as f:
        f.write("technique,stem,rank\n")
        for tech, ct in discover_techniques(root):
            deid_emb, deid_keys = load_embeddings(ct)
            if not deid_emb:
                continue
            common = [k for k in deid_keys if k in gal_idx]
            q_mat = np.stack([deid_emb[k] for k in common])
            sim = q_mat @ aligned_mat[np.array([gal_idx[k] for k in common])].T
            idcols = {}
            for i, k in enumerate(common):
                idcols.setdefault(idmap.get(k), []).append(i)
            for i, k in enumerate(common):
                same = np.asarray(sorted(idcols[idmap[k]]))
                best = float(sim[i][same].max())
                r = int(np.sum(sim[i] > best)) + 1
                f.write(f"{tech},{k},{r}\n")
    print("wrote", out)
