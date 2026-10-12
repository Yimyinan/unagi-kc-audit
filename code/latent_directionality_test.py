"""latent_directionality_test.py — 分析 A：方向信息是否已经编码在 latent 里？

动机
----
前序结论（RETRAIN_NECESSITY_ANALYSIS.md §3）：TRRUST 的 activation/repression 方向
信息在 UNAGI 的数据流里没有任何入口（参考库硬编码 human_encode.txt，无 direction 列）。
因此"注入方向信息"这个实验在方法上无效。

但那留下一个更有意思的问题：**方向信息是否本来就已经在 latent 里了？**
如果 latent 自发编码了方向，那"注入"就是多余的——问题被回答，且是正向发现。

方法
----
解码器读出 W（由 ridge_readout 得到，形状 (n_latent, n_genes)），
x_hat_g = W[:,g] · z，即 W[:,g] ∈ R^64 是基因 g 的解码方向。

TRRUST 给出每个 TF 的激活靶 A_t 与抑制靶 R_t。若方向信息被编码，则
"TF 上调时 A_t 上调、R_t 下调"这一关系应当体现在
    d_t = mean_{g∈A_t} W[:,g] - mean_{g∈R_t} W[:,g]
应向 W[:,TF]（TF 自身的解码方向）对齐：

    预测：cos( W[:,TF_t] , d_t ) > 0

为什么这个统计量是对的
--------------------
latent 空间存在规范不确定性：轴可置换、可反号、可正交旋转。
  轴置换 / 反号：W 的行被置换或反号 -> 两个列向量的 cos 不变。
  正交旋转 z -> Qz：W -> QW，cos(Qa,Qb) = cos(a,b) 不变。
所以 cos 对 latent 的规范不确定性完全免疫，可以跨 seed / 跨配置比较。
这是本分析能成立的关键。

对照（null）
-----------
关键混淆：一个 TF 的靶基因本来就共表达，它们的解码方向本来就可能与 TF 自身同向，
**这与方向标注无关**。因此对照必须固定靶集合、只打乱方向标签：
  在 A_t ∪ R_t 内部随机重排 |A_t| / |R_t| 的分组，B 次，得到 cos 的零分布。
这检验的是"方向标注本身是否携带信息"，而非"靶集合是否与 TF 相关"。

输出
----
outputs/positive_result/LATENT_DIRECTIONALITY.md + latent_directionality.json
"""
import os, json, glob, collections
import numpy as np

F = r"F:/UNAGI_Project"
LOADING = os.path.join(F, "UNAGI/decoder_unified/loading")
TRRUST = os.path.join(F, "unagi_custom_data/trrust_raw_human.txt")
UNIV_HUMAN = os.path.join(F, "UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/0.h5ad")
OUT = r"F:/UNAGI_Project/outputs/positive_result"

B = 2000            # 置换次数
MIN_PER_CLASS = 3   # 每类最少靶基因数
RNG = np.random.default_rng(20260916)

# ---------------------------------------------------------------- 基因宇宙
import anndata as ad
adata = ad.read_h5ad(UNIV_HUMAN)
GENES = [str(g) for g in adata.var_names]
GENE_IDX = {g.upper(): i for i, g in enumerate(GENES)}
print(f"human gene universe: {len(GENES):,} genes (from R11 stagedata)")

# ---------------------------------------------------------------- TRRUST
trrust = collections.defaultdict(lambda: {"A": [], "R": []})
n_edge = 0
with open(TRRUST, encoding="utf-8", errors="ignore") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        tf, tgt, direction = p[0].upper(), p[1].upper(), p[2]
        n_edge += 1
        if direction == "Activation":
            trrust[tf]["A"].append(tgt)
        elif direction == "Repression":
            trrust[tf]["R"].append(tgt)

print(f"TRRUST: {n_edge:,} edges, {len(trrust):,} TFs with a direction label")

# 只保留：TF 自身在宇宙内 + 两类靶基因各 >= MIN_PER_CLASS 且都在宇宙内
ELIG = {}
for tf, d in trrust.items():
    if tf not in GENE_IDX:
        continue
    A = sorted({GENE_IDX[g] for g in d["A"] if g in GENE_IDX})
    R = sorted({GENE_IDX[g] for g in d["R"] if g in GENE_IDX})
    if len(A) >= MIN_PER_CLASS and len(R) >= MIN_PER_CLASS:
        ELIG[tf] = (GENE_IDX[tf], np.array(A), np.array(R))

print(f"eligible TFs (both classes >= {MIN_PER_CLASS} targets in universe): {len(ELIG)}")
sizes = [(len(a), len(r)) for _, a, r in ELIG.values()]
if sizes:
    aa = np.array([s[0] for s in sizes]); rr = np.array([s[1] for s in sizes])
    print(f"  |A|: median={int(np.median(aa))} range={aa.min()}-{aa.max()}")
    print(f"  |R|: median={int(np.median(rr))} range={rr.min()}-{rr.max()}")

# ---------------------------------------------------------------- 载入 loadings
files = {}
for p in sorted(glob.glob(os.path.join(LOADING, "*.npy"))):
    cfg = os.path.basename(p).replace(".npy", "")
    W = np.load(p)
    if W.shape[1] != len(GENES):     # 只保留人源宇宙（20466）
        continue
    files[cfg] = W
# 去掉已知的完全重复（R14_seed* == R14_s*）
seen, keep = {}, []
for cfg, W in files.items():
    h = hash(np.ascontiguousarray(W).tobytes())
    if h in seen:
        continue
    seen[h] = cfg
    keep.append(cfg)
files = {c: files[c] for c in keep}
print(f"\nusable human configs: {len(files)}  ({', '.join(sorted(files))})")

# ---------------------------------------------------------------- 主统计
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

per_cfg = {}          # cfg -> {tf: {cos, p, dnorm, sep}}
tf_pool = collections.defaultdict(list)

for cfg, W in files.items():
    res = {}
    Wt = W  # (64, G); W[:, g] = gene g 的 decoder 方向
    for tf, (tf_idx, A, R) in ELIG.items():
        cA = Wt[:, A].mean(axis=1)
        cR = Wt[:, R].mean(axis=1)
        d = cA - cR
        obs = cos(Wt[:, tf_idx], d)

        # ---- 置换对照：固定靶集合，只打乱方向标签 ----
        pooled = np.concatenate([A, R])
        nA = len(A)
        null = np.empty(B)
        for b in range(B):
            perm = RNG.permutation(pooled)
            a_i, r_i = perm[:nA], perm[nA:]
            db = Wt[:, a_i].mean(axis=1) - Wt[:, r_i].mean(axis=1)
            null[b] = cos(Wt[:, tf_idx], db)

        # 双侧 p（置换分布未必对称，用 |null| >= |obs| 的秩式定义）
        p_perm = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1))

        res[tf] = dict(cos=obs, p=p_perm, dnorm=float(np.linalg.norm(d)),
                       nA=int(nA), nR=int(len(R)),
                       null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)))
        tf_pool[tf].append(obs)
    per_cfg[cfg] = res

np.save(os.path.join(OUT, "_tmp_latent_dir_percfg.npy"),
        np.array([1]))  # 占位，实际结果走 json

# ---------------------------------------------------------------- 聚合
# 每个 TF：跨配置的 cos 均值 + Stouffer 合并 p
summary = []
for tf, vals in tf_pool.items():
    vals = np.array(vals)
    # 用置换零分布把 cos 标准化为 z，再 Stouffer 合并
    zs, ps = [], []
    for cfg, res in per_cfg.items():
        if tf not in res:
            continue
        r = res[tf]
        if r["null_sd"] > 0:
            zs.append((r["cos"] - r["null_mean"]) / r["null_sd"])
        ps.append(r["p"])
    ps = np.array(ps); zs = np.array(zs)
    # Stouffer（单侧：预测 cos>0）
    z_comb = float(zs.sum() / np.sqrt(len(zs))) if len(zs) else np.nan
    from scipy import stats
    p_comb = float(1 - stats.norm.cdf(z_comb)) if len(zs) else np.nan
    summary.append(dict(
        tf=tf, n_cfg=len(vals), mean_cos=float(vals.mean()),
        median_cos=float(np.median(vals)),
        frac_pos=float((vals > 0).mean()),
        min_cos=float(vals.min()), max_cos=float(vals.max()),
        z_comb=z_comb, p_comb=p_comb,
        nA=per_cfg[sorted(files)[0]][tf]["nA"],
        nR=per_cfg[sorted(files)[0]][tf]["nR"],
    ))

# BH-FDR
summary.sort(key=lambda s: (s["p_comb"] if np.isfinite(s["p_comb"]) else 1))
m = len(summary)
prev = 1.0
for i in range(m - 1, -1, -1):
    p = summary[i]["p_comb"]
    if not np.isfinite(p):
        summary[i]["q"] = np.nan
        continue
    prev = min(prev, p * m / (i + 1))
    summary[i]["q"] = float(prev)

# ---------------------------------------------------------------- 报告
all_cos = np.concatenate([np.array(v) for v in tf_pool.values()])
from scipy import stats as st

print()
print("=" * 92)
print(f"分析 A：方向信息是否已编码在 latent 中？  cos(W[:,TF], cA - cR)  预测 > 0")
print("=" * 92)
print(f"  配置数            : {len(files)}")
print(f"  可评估 TF 数      : {len(tf_pool)}")
print(f"  总观测数 (cfg×TF) : {len(all_cos)}")
print(f"  全样本 mean cos   : {all_cos.mean():+.4f}   median {np.median(all_cos):+.4f}")
print(f"  cos > 0 的比例    : {(all_cos > 0).mean()*100:.1f}%")
w = st.wilcoxon(all_cos)
print(f"  Wilcoxon (cos vs 0): statistic={w.statistic:.0f}  p={w.pvalue:.3e}")

mean_cos_per_tf = np.array([s["mean_cos"] for s in summary])
w2 = st.wilcoxon(mean_cos_per_tf)
print(f"  按 TF 聚合后 mean cos: {mean_cos_per_tf.mean():+.4f}  "
      f"frac>0={(mean_cos_per_tf>0).mean()*100:.1f}%  Wilcoxon p={w2.pvalue:.3e}")

n_sig = int(np.sum(np.array([s["q"] for s in summary], dtype=float) < 0.05))
print(f"  通过 BH-FDR q<0.05 的 TF 数: {n_sig}/{len(summary)}")

print()
print("  Top 15 TF（按合并 p）:")
print(f'    {"TF":<10}{"nA":>4}{"nR":>4}{"mean_cos":>10}{"frac>0":>8}{"z_comb":>9}{"p_comb":>11}{"q":>10}')
for s in summary[:15]:
    print(f'    {s["tf"]:<10}{s["nA"]:>4}{s["nR"]:>4}{s["mean_cos"]:>+10.4f}'
          f'{s["frac_pos"]*100:>7.0f}%{s["z_comb"]:>+9.2f}{s["p_comb"]:>11.3e}{s["q"]:>10.3f}')

# 造血 TF 特别关注
HEM = ["GATA1", "GATA2", "TAL1", "KLF1", "LMO2", "RUNX1", "SPI1", "CEBPA",
       "GFI1", "GFI1B", "PAX5", "NFE2"]
print()
print("  12 个造血 TF 的表现:")
print(f'    {"TF":<10}{"eligible":>9}{"nA":>5}{"nR":>5}{"mean_cos":>10}{"frac>0":>8}{"p_comb":>11}')
hem_rows = []
for tf in HEM:
    s = next((x for x in summary if x["tf"] == tf), None)
    if s is None:
        why = "TF不在宇宙" if tf not in GENE_IDX else f"每类<{MIN_PER_CLASS}靶"
        print(f'    {tf:<10}{"no":>9}   ({why})')
        hem_rows.append(dict(tf=tf, eligible=False, reason=why))
    else:
        print(f'    {tf:<10}{"yes":>9}{s["nA"]:>5}{s["nR"]:>5}{s["mean_cos"]:>+10.4f}'
              f'{s["frac_pos"]*100:>7.0f}%{s["p_comb"]:>11.3e}')
        hem_rows.append(dict(tf=tf, eligible=True, **{k: s[k] for k in
                          ("nA", "nR", "mean_cos", "frac_pos", "z_comb", "p_comb", "q")}))

# 跨 seed 可复现性（同一配置的 3 个 seed）
print()
print("=" * 92)
print("跨 seed 可复现性：同一配置 3 个 seed 的 cos 是否同号？")
print("=" * 92)
groups = {"R11 (KLF1+TAL1)": ["R11_s42", "R11_s123", "R11_s456"],
          "R12 (GATA2/GFI1B/PAX5)": ["R12_s42", "R12_s123", "R12_s456"],
          "R13 (pan-hema 11TF)": ["R13_s42", "R13_s123", "R13_s456"],
          "R14/R15 (vanilla ref)": ["R14_s42", "R14_s123", "R14_s456"]}
rep_rows = []
for gname, cfgs in groups.items():
    cfgs = [c for c in cfgs if c in per_cfg]
    if len(cfgs) < 2:
        continue
    tfs = sorted(set.intersection(*[set(per_cfg[c]) for c in cfgs]))
    if not tfs:
        continue
    M = np.array([[per_cfg[c][t]["cos"] for c in cfgs] for t in tfs])   # (n_tf, n_seed)
    # 同号一致率
    all_same = np.all(M > 0, axis=1) | np.all(M < 0, axis=1)
    # 组内相关（seed 间）
    cors = []
    for i in range(len(cfgs)):
        for j in range(i + 1, len(cfgs)):
            a, b = M[:, i], M[:, j]
            if a.std() > 0 and b.std() > 0:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
    print(f'  {gname:<26} n_TF={len(tfs):>3}  同号率={all_same.mean()*100:>5.1f}%  '
          f'seed间 cos 相关={np.mean(cors):+.3f} (n={len(cors)})')
    rep_rows.append(dict(group=gname, n_tf=len(tfs), sign_consistency=float(all_same.mean()),
                         mean_inter_seed_corr=float(np.mean(cors)) if cors else None))

# ---------------------------------------------------------------- 落盘
out = dict(
    meta=dict(n_configs=len(files), configs=sorted(files), n_eligible_tf=len(ELIG),
              n_evaluated_tf=len(tf_pool), n_obs=int(len(all_cos)),
              permutations=B, min_per_class=MIN_PER_CLASS,
              gene_universe=len(GENES), gene_universe_source=UNIV_HUMAN,
              test="cos(W[:,TF], mean_A(W[:,g]) - mean_R(W[:,g]))  predicted > 0",
              null="direction-label permutation within the TF's own target set"),
    global_stats=dict(mean_cos=float(all_cos.mean()), median_cos=float(np.median(all_cos)),
                      frac_pos=float((all_cos > 0).mean()),
                      wilcoxon_all=dict(stat=float(w.statistic), p=float(w.pvalue)),
                      mean_cos_per_tf=float(mean_cos_per_tf.mean()),
                      wilcoxon_per_tf=dict(stat=float(w2.statistic), p=float(w2.pvalue)),
                      n_fdr_sig=n_sig),
    per_tf=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in s.items()} for s in summary],
    hematopoiesis=hem_rows,
    cross_seed_reproducibility=rep_rows,
)
p = os.path.join(OUT, "latent_directionality.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(f"\n-> {p}")
