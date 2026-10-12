"""latent_directionality_refined.py — 分析 A 的精细化统计量

主检验的统计量 cos(W[:,TF], cA-cR) 有一个潜在混淆：
  cA 与 cR 的差异可能来自**载荷幅度**（A 靶基因解码方向更长/更对齐）
  而不是**符号方向**（A 往一边、R 往另一边）。
虽然"固定靶集合打乱标签"的置换对照能部分处理，但更干净的做法是
直接检验方向性本身。

本脚本用两个互补的、幅度无关的统计量：

  S1  投影差（最自然的"方向性"表述）
        proj_A = mean_{g∈A} cos(W[:,g], W[:,TF])
        proj_R = mean_{g∈R} cos(W[:,g], W[:,TF])
        预测: proj_A - proj_R > 0
      含义：激活靶是否比抑制靶更对齐 TF 自身的解码方向。
      由于分子是 cos（单位向量），与载荷长度无关。

  S2  单位化质心分离
        u_g = W[:,g] / ||W[:,g]||
        d_unit = mean_{g∈A} u_g - mean_{g∈R} u_g
        统计量 cos(W[:,TF], d_unit)
      同样幅度无关。

两者都对 latent 的轴置换/反号/正交旋转不变（cos 不变），
因此可跨 seed 比较。

同时输出 S1 的逐 TF 置换 p 值与 BH-FDR。
"""
import os, json, glob, collections
import numpy as np
from scipy import stats

F = r"F:/UNAGI_Project"
LOADING = os.path.join(F, "UNAGI/decoder_unified/loading")
TRRUST = os.path.join(F, "unagi_custom_data/trrust_raw_human.txt")
UNIV = os.path.join(F, "UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/0.h5ad")
OUT = r"F:/UNAGI_Project/outputs/positive_result"

B = 2000
MIN_PER_CLASS = 3
RNG = np.random.default_rng(20260916)

import anndata as ad
GENES = [str(g) for g in ad.read_h5ad(UNIV).var_names]
GENE_IDX = {g.upper(): i for i, g in enumerate(GENES)}

trrust = collections.defaultdict(lambda: {"A": [], "R": []})
with open(TRRUST, encoding="utf-8", errors="ignore") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        if p[2] == "Activation":
            trrust[p[0].upper()]["A"].append(p[1].upper())
        elif p[2] == "Repression":
            trrust[p[0].upper()]["R"].append(p[1].upper())

ELIG = {}
for tf, d in trrust.items():
    if tf not in GENE_IDX:
        continue
    A = sorted({GENE_IDX[g] for g in d["A"] if g in GENE_IDX})
    R = sorted({GENE_IDX[g] for g in d["R"] if g in GENE_IDX})
    if len(A) >= MIN_PER_CLASS and len(R) >= MIN_PER_CLASS:
        ELIG[tf] = (GENE_IDX[tf], np.array(A), np.array(R))

files = {}
for p in sorted(glob.glob(os.path.join(LOADING, "*.npy"))):
    W = np.load(p)
    if W.shape[1] == len(GENES):
        files[os.path.basename(p).replace(".npy", "")] = W
seen, keep = {}, []
for c, W in files.items():
    h = hash(np.ascontiguousarray(W).tobytes())
    if h in seen:
        continue
    seen[h] = c
    keep.append(c)
files = {c: files[c] for c in keep}

def unitize(W):
    n = np.linalg.norm(W, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return W / n

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan

print("=" * 94)
print("分析 A 精细化：幅度无关的方向性统计量")
print("=" * 94)

s1_obs, s2_obs = [], []
per_tf_s1 = collections.defaultdict(list)
per_tf_s2 = collections.defaultdict(list)
per_tf_s1_p = collections.defaultdict(list)

for cfg, W in files.items():
    U = unitize(W)
    for tf, (ti, A, R) in ELIG.items():
        u_tf = U[:, ti]
        # ---- S1: 投影差 ----
        projA = float(np.mean([float(np.dot(u_tf, U[:, g])) for g in A]))
        projR = float(np.mean([float(np.dot(u_tf, U[:, g])) for g in R]))
        s1 = projA - projR
        # ---- S2: 单位化质心分离 ----
        d = U[:, A].mean(axis=1) - U[:, R].mean(axis=1)
        s2 = cos(W[:, ti], d)
        # ---- S1 置换 p ----
        pooled = np.concatenate([A, R]); nA = len(A)
        proj_all = np.array([float(np.dot(u_tf, U[:, g])) for g in pooled])
        null = np.empty(512)
        for b in range(512):
            perm = RNG.permutation(proj_all)
            null[b] = perm[:nA].mean() - perm[nA:].mean()
        p1 = float((np.sum(np.abs(null) >= abs(s1)) + 1) / (512 + 1))

        s1_obs.append(s1); s2_obs.append(s2)
        per_tf_s1[tf].append(s1); per_tf_s2[tf].append(s2)
        per_tf_s1_p[tf].append(p1)

s1_obs = np.array(s1_obs); s2_obs = np.array(s2_obs)
print(f'配置数={len(files)}  可评估TF={len(ELIG)}  观测数={len(s1_obs)}')
print()
print(f'{"统计量":<50}{"mean":>10}{"median":>10}{"frac>0":>9}{"Wilcoxon p":>13}')
for name, arr in [("S1 投影差  proj_A - proj_R", s1_obs),
                  ("S2 单位化质心分離 cos(W[:,TF], d_unit)", s2_obs)]:
    w = stats.wilcoxon(arr)
    print(f'{name:<50}{arr.mean():>+10.5f}{np.median(arr):>+10.5f}'
          f'{(arr>0).mean()*100:>8.1f}%{w.pvalue:>13.3e}')

# ---- 逐 TF 聚合 + FDR ----
summary = []
for tf in ELIG:
    v1 = np.array(per_tf_s1[tf]); v2 = np.array(per_tf_s2[tf])
    ps = np.array(per_tf_s1_p[tf])
    # Fisher 合并（单侧）
    chi = -2 * np.log(np.clip(ps, 1e-12, 1)) if False else None
    # 用 Stouffer 在 z 尺度合并更稳：把单侧 p 转 z
    zs = stats.norm.ppf(1 - np.clip(ps, 1e-12, 1 - 1e-12))
    zc = float(zs.sum() / np.sqrt(len(zs)))
    summary.append(dict(tf=tf, nA=int(len(ELIG[tf][1])), nR=int(len(ELIG[tf][2])),
                        mean_s1=float(v1.mean()), frac_pos_s1=float((v1 > 0).mean()),
                        mean_s2=float(v2.mean()), frac_pos_s2=float((v2 > 0).mean()),
                        z_comb=zc, p_comb=float(1 - stats.norm.cdf(zc))))
summary.sort(key=lambda s: s["p_comb"])
m = len(summary); prev = 1.0
for i in range(m - 1, -1, -1):
    prev = min(prev, summary[i]["p_comb"] * m / (i + 1))
    summary[i]["q"] = float(prev)

print()
print(f'{"TF":<10}{"nA":>4}{"nR":>4}{"mean_s1":>10}{"frac>0":>8}{"mean_s2":>10}{"z_comb":>9}{"p_comb":>11}{"q":>8}')
for s in summary[:18]:
    print(f'{s["tf"]:<10}{s["nA"]:>4}{s["nR"]:>4}{s["mean_s1"]:>+10.5f}'
          f'{s["frac_pos_s1"]*100:>7.0f}%{s["mean_s2"]:>+10.5f}'
          f'{s["z_comb"]:>+9.2f}{s["p_comb"]:>11.3e}{s["q"]:>8.3f}')

n_q = int(sum(1 for s in summary if s["q"] < 0.05))
print(f'\n通过 BH-FDR q<0.05 的 TF: {n_q}/{len(summary)}')

m_s1 = np.array([s["mean_s1"] for s in summary])
print(f'按 TF 聚合 S1: mean={m_s1.mean():+.5f} frac>0={(m_s1>0).mean()*100:.1f}% '
      f'Wilcoxon p={stats.wilcoxon(m_s1).pvalue:.3e}')

HEM = ["GATA1","GATA2","TAL1","KLF1","LMO2","RUNX1","SPI1","CEBPA","GFI1","GFI1B","PAX5","NFE2"]
print()
print("12 个造血 TF:")
print(f'{"TF":<10}{"eligible":>9}{"nA":>5}{"nR":>5}{"mean_s1":>11}{"frac>0":>8}{"mean_s2":>11}{"p_comb":>11}')
hem_rows = []
for tf in HEM:
    s = next((x for x in summary if x["tf"] == tf), None)
    if s is None:
        print(f'{tf:<10}{"no":>9}')
        hem_rows.append(dict(tf=tf, eligible=False))
    else:
        print(f'{tf:<10}{"yes":>9}{s["nA"]:>5}{s["nR"]:>5}{s["mean_s1"]:>+11.5f}'
              f'{s["frac_pos_s1"]*100:>7.0f}%{s["mean_s2"]:>+11.5f}{s["p_comb"]:>11.3e}')
        hem_rows.append(dict(tf=tf, eligible=True, **{k: s[k] for k in
                          ("nA","nR","mean_s1","frac_pos_s1","mean_s2","z_comb","p_comb","q")}))

out = dict(meta=dict(n_configs=len(files), n_tf=len(ELIG), permutations=512,
                     stat_s1="proj_A - proj_R, proj_X = mean_{g in X} cos(W[:,g], W[:,TF])",
                     stat_s2="cos(W[:,TF], mean_A(u_g) - mean_R(u_g)), u_g = W[:,g]/||W[:,g]||"),
           global_stats=dict(
               s1=dict(mean=float(s1_obs.mean()), median=float(np.median(s1_obs)),
                       frac_pos=float((s1_obs > 0).mean()),
                       wilcoxon_p=float(stats.wilcoxon(s1_obs).pvalue)),
               s2=dict(mean=float(s2_obs.mean()), median=float(np.median(s2_obs)),
                       frac_pos=float((s2_obs > 0).mean()),
                       wilcoxon_p=float(stats.wilcoxon(s2_obs).pvalue)),
               per_tf_s1=dict(mean=float(m_s1.mean()), frac_pos=float((m_s1 > 0).mean()),
                              wilcoxon_p=float(stats.wilcoxon(m_s1).pvalue),
                              n_fdr_sig=n_q, n_tf=len(summary))),
           per_tf=summary, hematopoiesis=hem_rows)
p = os.path.join(OUT, "latent_directionality_refined.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(f'\n-> {p}')
