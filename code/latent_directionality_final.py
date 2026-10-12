"""latent_directionality_final.py — 分析 A 最终版（修正 + 表达量混淆控制）

本脚本修正并扩展前两版：

  [修正] z_comb 的符号错误。前一版把**双侧**置换 p 用 `ppf(1-p)` 转成**单侧** z，
        导致强负效应的 TF（如 HDAC1、FOS、PPARA）被误判为强正效应并进入 FDR 显著集。
        正确做法：z = sign(stat) * ppf(1 - p_two_sided/2)。

  [新增对照 C6] 表达量混淆。靶基因集合的检出率若在 A/R 之间系统性不同，
        cos 可能反映的是"高表达基因更对齐"而非方向。用两种方式检验：
        (a) 表达标签对照：在 A∪R 内按检出率高低伪造 A/R 标签，重算统计量；
        (b) 检出率匹配零模型：为每个 A/R 基因抽取检出率相近的随机替身。

  [新增] 逐 TF 的 A/R 检出率差，用于解释符号异质性。

输出：outputs/positive_result/LATENT_DIRECTIONALITY.md + latent_directionality_final.json
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
NG = len(GENES)
DET = np.load(os.path.join(OUT, "_tmp_det.npy"))
assert len(DET) == NG, f"detection vector {len(DET)} != {NG}"

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
    if W.shape[1] == NG:
        files[os.path.basename(p).replace(".npy", "")] = W
seen, keep = {}, []
for c, W in files.items():
    h = hash(np.ascontiguousarray(W).tobytes())
    if h in seen:
        continue
    seen[h] = c
    keep.append(c)
files = {c: files[c] for c in keep}

def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan

# 检出率排序索引，用于"近邻替身"抽样
order = np.argsort(DET)
rank_of = np.empty(NG, dtype=int)
rank_of[order] = np.arange(NG)

# 校验基因顺序一致（var_names 可能有重复，用第一个出现）
# GENE_IDX 已按 upper() 建，与 DET 同序

print("=" * 100)
print("分析 A 最终版：方向信息是否编码在 latent 中（含表达量混淆控制）")
print("=" * 100)
print(f"配置数={len(files)}  可评估TF={len(ELIG)}  基因宇宙={NG}")

# ============================================================ 主统计（幅度无关 S1）
per_tf = collections.defaultdict(lambda: collections.defaultdict(list))
detA_all, detR_all = {}, {}

for cfg, W in files.items():
    n = np.linalg.norm(W, axis=0, keepdims=True); n[n == 0] = 1.0
    U = W / n
    for tf, (ti, A, R) in ELIG.items():
        u_tf = U[:, ti]
        pA = u_tf @ U[:, A]      # 与 TF 方向的余弦
        pR = u_tf @ U[:, R]
        s1 = float(pA.mean() - pR.mean())
        per_tf[tf]["s1"].append(s1)

        pooled = np.concatenate([A, R]); nA = len(A)
        pall = np.concatenate([pA, pR])
        # --- 置换零模型（固定靶集合、打乱标签）
        null = np.empty(B)
        for b in range(B):
            q = RNG.permutation(pall)
            null[b] = q[:nA].mean() - q[nA:].mean()
        p_two = float((np.sum(np.abs(null) >= abs(s1)) + 1) / (B + 1))
        per_tf[tf]["p"].append(p_two)

        # --- C6a 表达标签对照：在 A∪R 内按检出率伪造 A/R
        dd = DET[pooled]
        hi = np.argsort(dd)[::-1][:nA]
        pl = np.zeros(len(pooled), dtype=bool); pl[hi] = True
        per_tf[tf]["s1_detlabel"].append(float(pall[pl].mean() - pall[~pl].mean()))

        # --- C6b 检出率匹配零模型
        others = np.setdiff1d(np.arange(NG), pooled, assume_unique=False)
        if len(others) >= 2:
            oa = others[np.argsort(np.abs(rank_of[others] - rank_of[A].mean()))[:len(A)]] \
                if len(others) >= len(A) else others[:len(A)]
            # 更稳的做法：按 rank 最近邻逐个抽替身（有放回，避免越界）
            simA = order[np.clip(rank_of[A][:, None] + RNG.integers(-50, 51, (len(A), 8)), 0, NG - 1)]
            simR = order[np.clip(rank_of[R][:, None] + RNG.integers(-50, 51, (len(R), 8)), 0, NG - 1)]
            qa = u_tf @ U[:, simA.ravel()]
            qr = u_tf @ U[:, simR.ravel()]
            per_tf[tf]["s1_detmatched"].append(float(qa.mean() - qr.mean()))

        detA_all.setdefault(tf, []).append(float(DET[A].mean()))
        detR_all.setdefault(tf, []).append(float(DET[R].mean()))

# ============================================================ 聚合（修正符号）
summary = []
for tf, v in per_tf.items():
    s1 = np.array(v["s1"]); ps = np.array(v["p"])
    # 修正：双侧 p -> 带符号的单侧 z
    z_each = np.sign(s1) * stats.norm.ppf(1 - np.clip(ps, 1e-12, 1 - 1e-12) / 2)
    zc = float(z_each.sum() / np.sqrt(len(z_each)))
    summary.append(dict(
        tf=tf, nA=int(len(ELIG[tf][1])), nR=int(len(ELIG[tf][2])),
        mean_s1=float(s1.mean()), sd_s1=float(s1.std(ddof=1)),
        frac_pos=float((s1 > 0).mean()),
        mean_s1_detlabel=float(np.mean(v["s1_detlabel"])),
        mean_s1_detmatched=float(np.mean(v["s1_detmatched"])),
        det_A=float(np.mean(detA_all[tf])), det_R=float(np.mean(detR_all[tf])),
        det_diff=float(np.mean(detA_all[tf]) - np.mean(detR_all[tf])),
        z_comb=zc, p_comb=float(1 - stats.norm.cdf(zc)),
        sign_consistent=bool(abs(s1 > 0).mean() in (0.0, 1.0)),
    ))
summary.sort(key=lambda s: s["p_comb"])
m = len(summary); prev = 1.0
for i in range(m - 1, -1, -1):
    prev = min(prev, summary[i]["p_comb"] * m / (i + 1))
    summary[i]["q"] = float(prev)

# ============================================================ 报告
all_s1 = np.concatenate([np.array(v["s1"]) for v in per_tf.values()])
all_dl = np.concatenate([np.array(v["s1_detlabel"]) for v in per_tf.values()])
all_dm = np.concatenate([np.array(v["s1_detmatched"]) for v in per_tf.values()])
m_obs = np.array([s["mean_s1"] for s in summary])
m_dm = np.array([s["mean_s1_detmatched"] for s in summary])
m_dl = np.array([s["mean_s1_detlabel"] for s in summary])

print()
print(f'{"统计量":<46}{"mean":>10}{"median":>10}{"frac>0":>9}{"Wilcoxon p":>13}')
for name, arr in [("观测 S1 = proj_A - proj_R", all_s1),
                  ("C6a 表达标签伪造 A/R", all_dl),
                  ("C6b 检出率匹配替身", all_dm)]:
    w = stats.wilcoxon(arr)
    print(f'{name:<46}{arr.mean():>+10.5f}{np.median(arr):>+10.5f}'
          f'{(arr>0).mean()*100:>8.1f}%{w.pvalue:>13.3e}')

print()
print("观测 vs 对照 (Mann-Whitney, 单侧):")
print(f'  obs vs C6a 表达标签: p={stats.mannwhitneyu(all_s1, all_dl, alternative="greater").pvalue:.3e}  '
      f'Δmean={all_s1.mean()-all_dl.mean():+.5f}')
print(f'  obs vs C6b 匹配替身: p={stats.mannwhitneyu(all_s1, all_dm, alternative="greater").pvalue:.3e}  '
      f'Δmean={all_s1.mean()-all_dm.mean():+.5f}')

print()
print("符号异质性（这是本分析最重要的发现之一）:")
n_pos = int((m_obs > 0).sum()); n_neg = int((m_obs < 0).sum())
n_cons = int(sum(1 for s in summary if s["sign_consistent"]))
n_cons_pos = int(sum(1 for s in summary if s["sign_consistent"] and s["mean_s1"] > 0))
n_cons_neg = int(sum(1 for s in summary if s["sign_consistent"] and s["mean_s1"] < 0))
print(f'  平均 S1 > 0 的 TF: {n_pos}/{len(summary)}    < 0 的: {n_neg}/{len(summary)}')
print(f'  跨 {len(files)} 个配置符号完全一致（100% 或 0%）的 TF: {n_cons}/{len(summary)}')
print(f'    其中一致为正: {n_cons_pos}    一致为负: {n_cons_neg}   <-- 后者与 TRRUST 约定相反')
print(f'  按 TF 聚合 S1 的 Wilcoxon p={stats.wilcoxon(m_obs).pvalue:.3e} (整体系数不代表每个TF)')

n_q = int(sum(1 for s in summary if s["q"] < 0.05))
n_qpos = int(sum(1 for s in summary if s["q"] < 0.05 and s["mean_s1"] > 0))
n_qneg = int(sum(1 for s in summary if s["q"] < 0.05 and s["mean_s1"] < 0))
print(f'  修正后 BH-FDR q<0.05 的 TF: {n_q}/{len(summary)}  (正 {n_qpos} / 负 {n_qneg})')

print()
print(f'{"TF":<9}{"nA":>4}{"nR":>4}{"S1":>10}{"符号一致":>9}{"det_A":>8}{"det_R":>8}'
      f'{"S1_detmatch":>13}{"p":>11}{"q":>7}')
for s in summary[:20]:
    print(f'{s["tf"]:<9}{s["nA"]:>4}{s["nR"]:>4}{s["mean_s1"]:>+10.5f}'
          f'{("是" if s["sign_consistent"] else "否"):>9}{s["det_A"]:>8.4f}{s["det_R"]:>8.4f}'
          f'{s["mean_s1_detmatched"]:>+13.5f}{s["p_comb"]:>11.2e}{s["q"]:>7.3f}')

# 检出率差与符号的关系
dd = np.array([s["det_diff"] for s in summary])
r, pr = stats.pearsonr(dd, m_obs)
rs, prs = stats.spearmanr(dd, m_obs)
print()
print(f'检出率差 (det_A - det_R) 与 S1 的相关: Pearson r={r:+.3f} (p={pr:.3e})  '
      f'Spearman rho={rs:+.3f} (p={prs:.3e})')
print(f'  -> {"存在表达量混淆" if pr < 0.05 else "无显著表达量混淆"}')

print()
print("  负向 TF 的检出率画像（检验是否由低表达造成）:")
neg = [s for s in summary if s["mean_s1"] < 0]
pos = [s for s in summary if s["mean_s1"] > 0]
if neg and pos:
    print(f'    负向 TF (n={len(neg)}): det_A={np.mean([s["det_A"] for s in neg]):.4f} '
          f'det_R={np.mean([s["det_R"] for s in neg]):.4f}')
    print(f'    正向 TF (n={len(pos)}): det_A={np.mean([s["det_A"] for s in pos]):.4f} '
          f'det_R={np.mean([s["det_R"] for s in pos]):.4f}')
neg_c = [s for s in summary if s["mean_s1"] < 0 and s["sign_consistent"]]
print(f'  符号一致的负向 TF 清单 ({len(neg_c)}): {[s["tf"] for s in neg_c]}')
print(f'  符号一致的正向 TF 清单 ({n_cons_pos}): '
      f'{[s["tf"] for s in summary if s["sign_consistent"] and s["mean_s1"]>0]}')

HEM = ["GATA1","GATA2","TAL1","KLF1","LMO2","RUNX1","SPI1","CEBPA","GFI1","GFI1B","PAX5","NFE2"]
print()
print("12 个造血 TF:")
print(f'{"TF":<8}{"可评估":>8}{"nA":>5}{"nR":>5}{"S1":>11}{"符号一致":>10}{"S1_detmatch":>13}{"p":>11}')
hem_rows = []
for tf in HEM:
    s = next((x for x in summary if x["tf"] == tf), None)
    if s is None:
        print(f'{tf:<8}{"否":>8}')
        hem_rows.append(dict(tf=tf, eligible=False))
    else:
        print(f'{tf:<8}{"是":>8}{s["nA"]:>5}{s["nR"]:>5}{s["mean_s1"]:>+11.5f}'
              f'{("是" if s["sign_consistent"] else "否"):>10}'
              f'{s["mean_s1_detmatched"]:>+13.5f}{s["p_comb"]:>11.2e}')
        hem_rows.append(dict(tf=tf, eligible=True, **{k: s[k] for k in
                          ("nA","nR","mean_s1","frac_pos","mean_s1_detmatched","z_comb","p_comb","q")}))

out = dict(meta=dict(n_configs=len(files), n_tf=len(ELIG), permutations=B,
                     stat="S1 = mean_{g in A} cos(W[:,g],W[:,TF]) - mean_{g in R} cos(W[:,g],W[:,TF])",
                     note="magnitude-independent (unit-normalised decoder directions); "
                          "invariant to latent axis permutation/sign/rotation",
                     z_fix="z = sign(S1) * ppf(1 - p_two_sided/2)"),
           global_stats=dict(
               obs=dict(mean=float(all_s1.mean()), median=float(np.median(all_s1)),
                        frac_pos=float((all_s1 > 0).mean()),
                        wilcoxon_p=float(stats.wilcoxon(all_s1).pvalue)),
               c6a_detection_label=dict(mean=float(all_dl.mean()),
                                        frac_pos=float((all_dl > 0).mean()),
                                        wilcoxon_p=float(stats.wilcoxon(all_dl).pvalue)),
               c6b_detection_matched=dict(mean=float(all_dm.mean()),
                                          frac_pos=float((all_dm > 0).mean()),
                                          wilcoxon_p=float(stats.wilcoxon(all_dm).pvalue)),
               obs_vs_c6a_p=float(stats.mannwhitneyu(all_s1, all_dl, alternative="greater").pvalue),
               obs_vs_c6b_p=float(stats.mannwhitneyu(all_s1, all_dm, alternative="greater").pvalue),
               per_tf=dict(n_pos=n_pos, n_neg=n_neg, n_sign_consistent=n_cons,
                           n_consistent_pos=n_cons_pos, n_consistent_neg=n_cons_neg,
                           wilcoxon_p=float(stats.wilcoxon(m_obs).pvalue),
                           n_fdr_sig=n_q, n_fdr_sig_pos=n_qpos, n_fdr_sig_neg=n_qneg),
               detection_corr=dict(pearson_r=float(r), pearson_p=float(pr),
                                   spearman_rho=float(rs), spearman_p=float(prs))),
           per_tf=summary, hematopoiesis=hem_rows)
p = os.path.join(OUT, "latent_directionality_final.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(f'\n-> {p}')
