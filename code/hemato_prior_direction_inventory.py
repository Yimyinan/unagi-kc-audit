"""hemato_prior_direction_inventory.py — 造血先验的方向信息盘点

把分析 A 的结论落到"我们真正用的那 12 个 TF"上：
TRRUST 里它们有多少条边、多少条带方向、多少条落在模型的基因宇宙内。

这决定了"注入方向信息"即使通路修好，实际上能注入多少料。
"""
import os, json, collections
import numpy as np

F = r"F:/UNAGI_Project"
TRRUST = os.path.join(F, "unagi_custom_data/trrust_raw_human.txt")
UNIV = os.path.join(F, "UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/0.h5ad")
ENCODE = os.path.join(F, "UNAGI/UNAGI_source/UNAGI/data/human_encode.txt")
OUT = r"F:/UNAGI_Project/outputs/positive_result"

import anndata as ad
GENES = [str(g).upper() for g in ad.read_h5ad(UNIV).var_names]
GSET = set(GENES)

trrust = collections.defaultdict(list)
with open(TRRUST, encoding="utf-8", errors="ignore") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            trrust[p[0].upper()].append((p[1].upper(), p[2], p[3]))

enc_tfs = set()
with open(ENCODE, encoding="utf-8", errors="ignore") as f:
    next(f, None)
    for line in f:
        p = line.split("\t")
        if p:
            enc_tfs.add(p[0].strip())

HEM = ["GATA1","GATA2","TAL1","KLF1","LMO2","RUNX1","SPI1","CEBPA","GFI1","GFI1B","PAX5","NFE2"]

print("=" * 108)
print("造血先验的方向信息盘点（模型基因宇宙 = 20,466 个人源基因）")
print("=" * 108)
print(f'{"TF":<8}{"在宇宙":>7}{"在ENCODE":>10}{"TRRUST边":>10}{"带方向":>8}'
      f'{"A(cov)":>12}{"R(cov)":>12}{"两类≥3":>9}{"可分析":>8}')
rows = []
for tf in HEM:
    in_univ = tf in GSET
    in_enc = tf in enc_tfs
    edges = trrust.get(tf, [])
    n_tot = len(edges)
    A = [g for g, d, _ in edges if d == "Activation"]
    R = [g for g, d, _ in edges if d == "Repression"]
    U = [g for g, d, _ in edges if d == "Unknown"]
    A_in = [g for g in A if g in GSET]
    R_in = [g for g in R if g in GSET]
    n_dir = len(A) + len(R)
    ok = (len(A_in) >= 3 and len(R_in) >= 3)
    print(f'{tf:<8}{("是" if in_univ else "否"):>7}{("是" if in_enc else "否"):>10}'
          f'{n_tot:>10}{n_dir:>8}{f"{len(A_in)}/{len(A)}":>12}{f"{len(R_in)}/{len(R)}":>12}'
          f'{("是" if ok else "否"):>9}{("是" if (ok and in_univ) else "否"):>8}')
    rows.append(dict(tf=tf, in_universe=in_univ, in_encode=in_enc, n_edges=n_tot,
                     n_directed=n_dir, n_unknown=len(U),
                     n_A=len(A), n_R=len(R), n_A_in_universe=len(A_in),
                     n_R_in_universe=len(R_in), analysable=bool(ok and in_univ)))

n_an = sum(1 for r in rows if r["analysable"])
n_uncov = sum(1 for r in rows if not r["in_universe"])
n_enc_miss = sum(1 for r in rows if not r["in_encode"])
n_A_frac = sum(r["n_A_in_universe"] for r in rows)
n_A_tot = sum(r["n_A"] for r in rows)
n_R_frac = sum(r["n_R_in_universe"] for r in rows)
n_R_tot = sum(r["n_R"] for r in rows)

print()
print(f'  12 个 TF 中：在基因宇宙内 {12-n_uncov}/12；在 ENCODE 中 {12-n_enc_miss}/12')
print(f'  TRRUST 带方向的边合计 {n_A_tot+n_R_tot} 条（A={n_A_tot}, R={n_R_tot}）')
print(f'  落在基因宇宙内: A={n_A_frac}/{n_A_tot} ({100*n_A_frac/max(n_A_tot,1):.1f}%), '
      f'R={n_R_frac}/{n_R_tot} ({100*n_R_frac/max(n_R_tot,1):.1f}%)')
print(f'  两类各≥3 靶（可做方向性检验）的 TF: {n_an}/12')

# 记录哪个 TF 因为什么原因不可分析
print()
print("不可分析的原因:")
for r in rows:
    if r["analysable"]:
        continue
    reasons = []
    if not r["in_universe"]:
        reasons.append("TF 基因本身不在模型基因宇宙")
    if r["n_A_in_universe"] < 3:
        reasons.append(f"宇宙内激活靶仅 {r['n_A_in_universe']} 个 (<3)")
    if r["n_R_in_universe"] < 3:
        reasons.append(f"宇宙内抑制靶仅 {r['n_R_in_universe']} 个 (<3)")
    if not r["in_encode"]:
        reasons.append("ENCODE 无此 TF（geneWeight 通路不可达）")
    print(f'  {r["tf"]:<8} {" ; ".join(reasons)}')

out = dict(gene_universe_size=len(GENES), tfs=rows,
           summary=dict(n_tf=12, n_in_universe=12-n_uncov, n_in_encode=12-n_enc_miss,
                        n_analysable=n_an,
                        n_directed_A=n_A_tot, n_directed_R=n_R_tot,
                        n_A_in_universe=n_A_frac, n_R_in_universe=n_R_frac))
p = os.path.join(OUT, "hemato_prior_direction_inventory.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n-> {p}')
