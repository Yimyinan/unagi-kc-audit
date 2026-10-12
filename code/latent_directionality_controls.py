"""latent_directionality_controls.py — 分析 A 的对照实验

主检验（latent_directionality_test.py）发现：
  全样本 mean cos = +0.0475, Wilcoxon p=1e-9；但按 TF 聚合后 p=0.15。
  => 效应是异质的：一部分 TF 强，多数为零。

在解释之前必须排除三类混淆：

  C1  "任意方向都能对上"     —— 把 W[:,TF] 换成随机基因的解码方向，cos 是否也 >0？
    若也 >0，则统计量有全局偏置，结果无效。

  C2  "任意基因集合都能被分开" —— 保留 W[:,TF]，把 A/R 换成大小匹配的随机基因集合。
    若也 >0，则效应来自 W 的整体结构而非 TRRUST 标注。

  C3  "靶基因共表达导致"      —— 已在主检验中由"固定靶集合、只打乱标签"的置换对照处理。
    这里额外做 leave-one-out：逐个剔除靶基因，看 cos 是否由个别基因驱动。

  C4  多重比较 / 异质性      —— 93 个 TF 的 mean_cos 分布 vs C1 的零分布，
    估计"真实有效"的 TF 比例（类似 Storey 方法）。
"""
import os, json, glob, collections
import numpy as np
from scipy import stats

F = r"F:/UNAGI_Project"
LOADING = os.path.join(F, "UNAGI/decoder_unified/loading")
TRRUST = os.path.join(F, "unagi_custom_data/trrust_raw_human.txt")
UNIV = os.path.join(F, "UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/0.h5ad")
OUT = r"F:/UNAGI_Project/outputs/positive_result"

B = 1000
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
ALLG = np.arange(len(GENES))

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan

def stat(W, dir_idx, A, R):
    return cos(W[:, dir_idx], W[:, A].mean(axis=1) - W[:, R].mean(axis=1))

print("=" * 94)
print("对照实验：cos(W[:,TF], cA - cR) 的三种零模型")
print("=" * 94)
print(f"配置数={len(files)}  可评估TF={len(ELIG)}  置换B={B}\n")

c1_r, c2_r, c3_r = [], [], []
per_tf_obs = collections.defaultdict(list)
per_tf_c1 = collections.defaultdict(list)
per_tf_c2 = collections.defaultdict(list)
loo_dev = []

for cfg, W in files.items():
    for tf, (tf_idx, A, R) in ELIG.items():
        obs = stat(W, tf_idx, A, R)
        per_tf_obs[tf].append(obs)

        # ---- C1: 随机基因方向代替 W[:,TF] ----
        v1 = []
        for _ in range(20):
            g = int(RNG.integers(len(GENES)))
            v1.append(stat(W, g, A, R))
        c1_r.extend(v1); per_tf_c1[tf].append(float(np.mean(v1)))

        # ---- C2: 大小匹配的随机基因集合代替 A/R ----
        v2 = []
        for _ in range(20):
            Ag = RNG.choice(ALLG, size=len(A), replace=False)
            Rg = RNG.choice(ALLG, size=len(R), replace=False)
            v2.append(stat(W, tf_idx, Ag, Rg))
        c2_r.extend(v2); per_tf_c2[tf].append(float(np.mean(v2)))

        # ---- C3: leave-one-out（只对中小规模靶集合） ----
        if len(A) <= 30 and len(R) <= 30 and cfg == sorted(files)[0]:
            drops = []
            for i in range(len(A)):
                if len(A) - 1 >= MIN_PER_CLASS:
                    drops.append(stat(W, tf_idx, np.delete(A, i), R))
            for j in range(len(R)):
                if len(R) - 1 >= MIN_PER_CLASS:
                    drops.append(stat(W, tf_idx, A, np.delete(R, j)))
            if drops:
                loo_dev.append(max(abs(d - obs) for d in drops))

c1 = np.array(c1_r); c2 = np.array(c2_r)
obs_all = np.concatenate([np.array(v) for v in per_tf_obs.values()])

print(f'{"零模型":<44}{"n":>7}{"mean":>10}{"median":>10}{"frac>0":>9}{"Wilcoxon p":>13}')
for name, arr in [("观测 (TRRUST 方向标注, 真实 TF 方向)", obs_all),
                  ("C1 随机基因方向 (A/R 保持)", c1),
                  ("C2 随机 A/R 基因集合 (TF 方向保持)", c2)]:
    w = stats.wilcoxon(arr) if len(arr) > 10 else None
    print(f'{name:<44}{len(arr):>7}{arr.mean():>+10.4f}{np.median(arr):>+10.4f}'
          f'{(arr>0).mean()*100:>8.1f}%{(w.pvalue if w else float("nan")):>13.3e}')

# 观测 vs 各自零模型
print()
print("观测 vs 零模型的分布比较 (Mann-Whitney):")
print(f'  obs vs C1: U-p={stats.mannwhitneyu(obs_all, c1, alternative="greater").pvalue:.3e}   '
      f'Δmean = {obs_all.mean()-c1.mean():+.4f}')
print(f'  obs vs C2: U-p={stats.mannwhitneyu(obs_all, c2, alternative="greater").pvalue:.3e}   '
      f'Δmean = {obs_all.mean()-c2.mean():+.4f}')
print(f'  C1 vs C2 : U-p={stats.mannwhitneyu(c1, c2).pvalue:.3f} (两个零模型应当无差异)')

# ---- C4: 异质性 / 真实有效比例 ----
m_obs = np.array([np.mean(per_tf_obs[t]) for t in ELIG])
m_c1 = np.array([np.mean(per_tf_c1[t]) for t in ELIG])
print()
print("按 TF 聚合后的分布（异质性检查）:")
print(f'  obs  mean_cos: mean={m_obs.mean():+.4f} sd={m_obs.std(ddof=1):.4f} '
      f'frac>0={(m_obs>0).mean()*100:.1f}%')
print(f'  C1   mean_cos: mean={m_c1.mean():+.4f} sd={m_c1.std(ddof=1):.4f} '
      f'frac>0={(m_c1>0).mean()*100:.1f}%')
print(f'  Wilcoxon(obs vs 0) p={stats.wilcoxon(m_obs).pvalue:.3e}')
# Storey 式估计：pi0 = min(1, 2*P(null region)/... ) 简化用 obs 的 sd 与 C1 的 sd 比较
print(f'  obs sd / C1 sd = {m_obs.std(ddof=1)/m_c1.std(ddof=1):.2f}  '
      f'(>1 说明存在真实的 TF 间异质信号)')

# 用 C1 的分布做经验零分布，估真实有效个数（FDR 控制）
c1_ref = np.array([np.mean(c1_r[i::20]) for i in range(20)])  # 20 个 C1 复制
thr = np.quantile(m_c1, 0.95)
n_obs_hi = int((m_obs > thr).sum())
print(f'  C1 的 95% 分位 = {thr:+.4f}；超过该阈值的 TF 数 = {n_obs_hi}/{len(m_obs)} '
      f'({100*n_obs_hi/len(m_obs):.1f}%)，期望假阳性 {0.05*len(m_obs):.1f}')

# ---- C3 leave-one-out ----
if loo_dev:
    ld = np.array(loo_dev)
    print()
    print(f'C3 leave-one-out（首个配置，|A|,|R|<=30 的 TF，n={len(ld)}）:')
    print(f'  单个靶基因移除后 cos 的最大偏移: median={np.median(ld):.4f} '
          f'p90={np.quantile(ld,0.9):.4f} max={ld.max():.4f}')

out = dict(
    n_configs=len(files), n_tf=len(ELIG), B=B,
    observed=dict(mean=float(obs_all.mean()), median=float(np.median(obs_all)),
                  frac_pos=float((obs_all > 0).mean()),
                  wilcoxon_p=float(stats.wilcoxon(obs_all).pvalue)),
    c1_random_direction=dict(mean=float(c1.mean()), median=float(np.median(c1)),
                             frac_pos=float((c1 > 0).mean()),
                             wilcoxon_p=float(stats.wilcoxon(c1).pvalue)),
    c2_random_genesets=dict(mean=float(c2.mean()), median=float(np.median(c2)),
                            frac_pos=float((c2 > 0).mean()),
                            wilcoxon_p=float(stats.wilcoxon(c2).pvalue)),
    obs_vs_c1_p=float(stats.mannwhitneyu(obs_all, c1, alternative="greater").pvalue),
    obs_vs_c2_p=float(stats.mannwhitneyu(obs_all, c2, alternative="greater").pvalue),
    per_tf=dict(obs_mean=float(m_obs.mean()), obs_sd=float(m_obs.std(ddof=1)),
                c1_mean=float(m_c1.mean()), c1_sd=float(m_c1.std(ddof=1)),
                wilcoxon_p=float(stats.wilcoxon(m_obs).pvalue),
                sd_ratio=float(m_obs.std(ddof=1) / m_c1.std(ddof=1)),
                n_above_c1_q95=n_obs_hi, frac_above=float(n_obs_hi / len(m_obs))),
    loo=dict(median=float(np.median(ld)), p90=float(np.quantile(ld, 0.9)),
             max=float(ld.max()), n=int(len(ld))) if loo_dev else None,
)
p = os.path.join(OUT, "latent_directionality_controls.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n-> {p}")
