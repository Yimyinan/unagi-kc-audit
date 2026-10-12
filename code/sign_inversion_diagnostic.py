"""sign_inversion_diagnostic.py — 符号反号的 17 个 TF 是生物学信号还是数据伪影？

背景：latent_directionality_final.py 发现 40/93 个 TF 跨 23 配置符号完全锁定，
其中 23 正 / 17 负。当时按"异质性"处理，未追问负号的成因。

目视印象：负号集里 NFKB1、RELA、FOS、JUND、HIF1A、ATF4 都是应激/诱导型 TF
（NF-κB / AP-1 / 缺氧 / ISR），而正号集里多是谱系决定型与组成型因子。
若成立，这是一个可解释的生物学发现：分化轨迹上训练出的 VAE 无法正确表征
"瞬时诱导型"TF 与靶基因的关系。

本脚本检验三个竞争解释：
  H1 生物学：负号与 TF 的"诱导型/应激型"属性相关
  H2 数据质量：负号 TF 在数据里低表达/低变异，其解码方向是噪声
  H3 靶集合结构：负号 TF 的靶集合规模或表达量分布不同

只用已有产物，零训练成本。
"""
import os, json, re, collections
import numpy as np
from scipy import stats

OUT = r"F:/UNAGI_Project/outputs/positive_result"
UNIV = r"F:/UNAGI_Project/UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/0.h5ad"
TRRUST = r"F:/UNAGI_Project/unagi_custom_data/trrust_raw_human.txt"

J = json.load(open(os.path.join(OUT, "latent_directionality_final.json"), encoding="utf-8"))
per_tf = {s["tf"]: s for s in J["per_tf"]}

locked_pos = [t for t, s in per_tf.items() if s["sign_consistent"] and s["mean_s1"] > 0]
locked_neg = [t for t, s in per_tf.items() if s["sign_consistent"] and s["mean_s1"] < 0]
print(f"符号锁定 TF：正 {len(locked_pos)} / 负 {len(locked_neg)}")

# ---------------- 基因层面观测量 ----------------
import anndata as ad
adata = ad.read_h5ad(UNIV)
GENES = [str(g).upper() for g in adata.var_names]
GI = {g: i for i, g in enumerate(GENES)}
DET = np.load(os.path.join(OUT, "_tmp_det.npy"))
MEAN = np.load(os.path.join(OUT, "_tmp_mean.npy"))

# 变异系数（在检出基因上），用 dataset.h5ad 已算的 mean/det 近似
def tf_features(tf):
    if tf not in GI:
        return None
    i = GI[tf]
    return dict(det=float(DET[i]), mean=float(MEAN[i]))

rows = []
for lab, group in [("pos", locked_pos), ("neg", locked_neg)]:
    for tf in group:
        f = tf_features(tf)
        s = per_tf[tf]
        if f is None:
            rows.append(dict(tf=tf, group=lab, in_univ=False))
            continue
        rows.append(dict(tf=tf, group=lab, in_univ=True, **f,
                         mean_s1=s["mean_s1"], nA=s["nA"], nR=s["nR"],
                         det_A=s["det_A"], det_R=s["det_R"],
                         s1_detmatch=s["mean_s1_detmatched"]))

df = [r for r in rows if r.get("in_univ")]
pos = [r for r in df if r["group"] == "pos"]
neg = [r for r in df if r["group"] == "neg"]

print()
print("=" * 90)
print("H2 数据质量：反号 TF 是否低表达/低检出？")
print("=" * 90)
print(f'{"组":<6}{"n":>4}{"检出率中位":>12}{"均值表达中位":>14}{"靶集合|A|中位":>14}{"靶集合|R|中位":>14}')
for lab, g in [("正号", pos), ("负号", neg)]:
    print(f'{lab:<6}{len(g):>4}{np.median([r["det"] for r in g]):>12.4f}'
          f'{np.median([r["mean"] for r in g]):>14.4f}'
          f'{np.median([r["nA"] for r in g]):>14.0f}'
          f'{np.median([r["nR"] for r in g]):>14.0f}')

for key, label in [("det", "TF 自身检出率"), ("mean", "TF 自身均值表达"),
                   ("nA", "激活靶数"), ("nR", "抑制靶数")]:
    a = np.array([r[key] for r in pos]); b = np.array([r[key] for r in neg])
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    print(f'  {label:<14} 正号 {a.mean():>8.4f} vs 负号 {b.mean():>8.4f}   MW p={u.pvalue:.4f}')

# ---------------- H1 生物学：诱导型 TF 注释 ----------------
# 用公认的应激/诱导型 TF 名录（预注册式，不事后挑选）
INDUCIBLE = {
    "NFKB1", "RELA", "RELB", "NFKB2", "NFKBIA", "NFKBIZ",     # NF-kB
    "FOS", "FOSB", "FOSL1", "FOSL2", "JUN", "JUNB", "JUND",   # AP-1
    "ATF2", "ATF3", "ATF4", "ATF6", "XBP1", "DDIT3",          # ISR / UPR
    "HIF1A", "EPAS1", "ARNT",                                  # hypoxia
    "STAT1", "STAT2", "STAT3", "STAT5A", "STAT5B", "STAT6",   # cytokine
    "IRF1", "IRF3", "IRF4", "IRF5", "IRF7", "IRF8", "IRF9",   # interferon
    "EGR1", "EGR2", "EGR3", "EGR4",                            # immediate-early
    "MYC", "MYCN", "FOSL1",                                    # growth-response
}
LINEAGE = {
    "GATA1", "GATA2", "GATA3", "KLF1", "TAL1", "LMO2", "RUNX1", "RUNX2", "RUNX3",
    "SPI1", "CEBPA", "CEBPB", "CEBPE", "GFI1", "GFI1B", "PAX5", "NFE2", "MYB",
    "TCF3", "TCF7", "EBF1", "IKZF1", "IKZF3", "PU.1",
    "FOXO1", "PPARA", "PPARG", "NR3C1", "RARA", "RARB", "VDR", "ESR1",
    "SOX2", "SOX9", "SOX10", "MITF", "PAX3", "PAX6", "PAX7", "NEUROG2", "ASCL1",
    "HOXA10", "HOXB4", "HOXC6", "HOXD9", "TBX1", "TBX3",
}

def classify(tf):
    if tf in INDUCIBLE:
        return "inducible"
    if tf in LINEAGE:
        return "lineage"
    return "other"

for lab, g in [("正号", pos), ("负号", neg)]:
    cls = collections.Counter(classify(r["tf"]) for r in g)
    print(f'  {lab} 分组: {dict(cls)}')

print()
print("=" * 90)
print("H1 生物学：诱导型 vs 谱系型 与符号的关联")
print("=" * 90)
tab = np.zeros((2, 2), dtype=int)
for r in df:
    c = classify(r["tf"])
    if c == "other":
        continue
    row = 0 if r["group"] == "pos" else 1
    col = 0 if c == "inducible" else 1
    tab[row, col] += 1
print(f'{"":<10}{"诱导型":>8}{"谱系型":>8}')
print(f'{"正号":<10}{tab[0,0]:>8}{tab[0,1]:>8}')
print(f'{"负号":<10}{tab[1,0]:>8}{tab[1,1]:>8}')
if tab.sum() > 0 and tab.min() >= 0:
    odds, p = stats.fisher_exact(tab)
    print(f'\nFisher exact: OR={odds:.2f}  p={p:.4f}')
    print(f'  -> {"诱导型 TF 显著偏向负号" if p < 0.05 and odds > 1 else "无显著关联（或方向相反）"}')

print()
print("两个锁定组的成员：")
print(f'  正号({len(locked_pos)}): {sorted(locked_pos)}')
print(f'  负号({len(locked_neg)}): {sorted(locked_neg)}')
print()
print("诱导型名录在两组的落点：")
for lab, g in [("正号", locked_pos), ("负号", locked_neg)]:
    hit = sorted([t for t in g if t in INDUCIBLE])
    print(f'  {lab} 中的诱导型: {hit}  ({len(hit)}/{len(g)})')

out = dict(n_pos=len(locked_pos), n_neg=len(locked_neg),
           contingency=tab.tolist(),
           fisher=dict(odds=float(odds), p=float(p)),
           locked_pos=sorted(locked_pos), locked_neg=sorted(locked_neg),
           inducible_in_pos=sorted([t for t in locked_pos if t in INDUCIBLE]),
           inducible_in_neg=sorted([t for t in locked_neg if t in INDUCIBLE]),
           per_tf=df)
p_ = os.path.join(OUT, "sign_inversion_diagnostic.json")
with open(p_, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(f'\n-> {p_}')
