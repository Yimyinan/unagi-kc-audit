"""database_coverage_audit.py — curated GRN 数据库是否系统性欠覆盖谱系决定因子？

起点（本次会话发现）：KLF1 —— 红系主控因子 —— 在 TRRUST 里 0 条**有向**边、
在 ENCODE 里**完全缺失**。这是特例还是系统性问题？

可检验命题：
  谱系决定因子（lineage-determining TF）的 curated 覆盖度系统性差于管家型/泛表达 TF，
  因为 TRRUST 与 ENCODE 主要由细胞系与特定刺激条件下的 ChIP/扰动实验构建，
  而谱系决定因子恰恰在那些系统里不活跃或不被研究。

两条独立检验：
  主检验（数据驱动，无手选名录）：以基因在造血发育各阶段的表达特异度 tau
    度量"谱系决定程度"，检验 tau 与覆盖度是否负相关。
  副检验（预注册名录）：一组公认的谱系决定因子 vs 其余 TF 的覆盖度分布比较。

只用磁盘上已有的 TRRUST / ENCODE / 表达数据。零训练成本。
"""
import os, json, collections
import numpy as np
import scipy.sparse as sp
from scipy import stats

OUT = r"F:/UNAGI_Project/outputs/positive_result"
DATA = r"F:/UNAGI_Project/UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/dataset.h5ad"
TRRUST = r"F:/UNAGI_Project/unagi_custom_data/trrust_raw_human.txt"
ENCODE = r"F:/UNAGI_Project/UNAGI/UNAGI_source/UNAGI/data/human_encode.txt"

import anndata as ad
adata = ad.read_h5ad(DATA)
GENES = [str(g).upper() for g in adata.var_names]
GI = {g: i for i, g in enumerate(GENES)}
print(f"数据集: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

# ---------------- 覆盖度指标 ----------------
tr_total, tr_dir = collections.Counter(), collections.Counter()
TFSET = set()
for line in open(TRRUST, encoding="utf-8", errors="ignore"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 4:
        continue
    tf, d = p[0].upper(), p[2]
    tr_total[tf] += 1
    TFSET.add(tf)
    if d in ("Activation", "Repression"):
        tr_dir[tf] += 1

enc_edges, enc_tfs = collections.Counter(), set()
with open(ENCODE, encoding="utf-8", errors="ignore") as f:
    next(f, None)
    for line in f:
        p = line.split("\t")
        if len(p) >= 3:
            enc_edges[p[0].strip()] += 1
            enc_tfs.add(p[0].strip())

print(f"TRRUST: {len(TFSET)} TFs / {sum(tr_total.values()):,} edges")
print(f"ENCODE: {len(enc_tfs)} TFs / {sum(enc_edges.values()):,} edges")

# ---------------- 表达特异度 tau ----------------
stage = adata.obs["stage"].astype(str).values
stages = sorted(set(stage))
print(f"阶段数: {len(stages)} -> {stages}")
X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)

means = np.zeros((len(stages), X.shape[1]))
for i, s in enumerate(stages):
    m = (stage == s)
    means[i] = np.asarray(X[m].mean(axis=0)).ravel()

mx = means.max(axis=0)
tau = np.zeros(X.shape[1])
nz = mx > 0
tau[nz] = (1.0 - means[:, nz] / mx[nz]).sum(axis=0) / (len(stages) - 1)
print(f"tau 范围: {tau[nz].min():.3f} - {tau.max():.3f}, 中位 {np.median(tau[nz]):.3f}")

# ---------------- 主检验：tau vs 覆盖度 ----------------
tfs_in_univ = sorted([t for t in TFSET if t in GI])
print(f"\nTRRUST TF 中落在基因宇宙内的: {len(tfs_in_univ)}")

rows = []
for tf in tfs_in_univ:
    i = GI[tf]
    rows.append(dict(tf=tf, tau=float(tau[i]),
                     tr_total=int(tr_total[tf]), tr_dir=int(tr_dir[tf]),
                     tr_dir_frac=float(tr_dir[tf] / tr_total[tf]) if tr_total[tf] else 0.0,
                     in_encode=bool(tf in enc_tfs),
                     enc_edges=int(enc_edges.get(tf, 0))))

t = np.array([r["tau"] for r in rows])
tot = np.array([r["tr_total"] for r in rows])
dr = np.array([r["tr_dir"] for r in rows])
ine = np.array([r["in_encode"] for r in rows])
ene = np.array([r["enc_edges"] for r in rows])

print()
print("=" * 92)
print("主检验：表达特异度 tau 与 curated 覆盖度的关系")
print("=" * 92)
tests = {}
for label, y, yname in [("TRRUST 总边数", tot, "tr_total"),
                        ("TRRUST 有向边数", dr, "tr_dir"),
                        ("ENCODE 边数", ene, "enc_edges")]:
    rho, p = stats.spearmanr(t, y)
    tests[yname] = dict(spearman_rho=float(rho), p=float(p))
    print(f'  tau vs {label:<18} Spearman ρ={rho:+.3f}  p={p:.2e}  '
          f'{"← 负相关，符合命题" if rho < 0 and p < 0.05 else ""}')
# ENCODE 命中（二值）
u = stats.mannwhitneyu(t[ine], t[~ine], alternative="two-sided")
print(f'  命中 ENCODE 的 TF 平均 tau={t[ine].mean():.3f} vs 未命中 {t[~ine].mean():.3f}  '
      f'MW p={u.pvalue:.2e}  (命中率 {ine.mean()*100:.1f}%)')
tests["encode_hit"] = dict(mean_tau_in=float(t[ine].mean()), mean_tau_out=float(t[~ine].mean()),
                           p=float(u.pvalue), hit_rate=float(ine.mean()))

# ---------------- 副检验：预注册的谱系决定因子名录 ----------------
MASTER = {
    # 造血 / 红系 / 髓系 / 淋系 / HSC
    "GATA1", "GATA2", "KLF1", "TAL1", "LMO2", "RUNX1", "SPI1", "CEBPA", "CEBPB",
    "CEBPE", "GFI1", "GFI1B", "PAX5", "NFE2", "MYB", "TCF3", "EBF1", "IKZF1",
    "LYL1", "MEF2C", "HHEX", "MAFB", "IRF8", "MEIS1",
    # 神经嵴 / 神经
    "SOX10", "FOXD3", "SNAI2", "TWIST1", "MITF", "PAX3", "PAX7", "TFAP2A",
    "MSX1", "DLX5", "ZEB2", "SOX9", "SOX2", "POU5F1", "NANOG", "NEUROG2",
    "ASCL1", "OLIG2",
    # 肺
    "NKX2-1", "FOXJ1", "FOXA2", "GATA6", "ELF5", "ETV5", "HOPX",
    # 泛发育 / 器官发生
    "GATA4", "NKX2-5", "TBX5", "ISL1", "HAND1", "HAND2", "MESP1", "PAX6",
    "OTX2", "LHX2", "SIX1", "PITX2",
}
is_master = np.array([r["tf"] in MASTER for r in rows])
print()
print("=" * 92)
print("副检验：预注册的谱系决定因子名录 vs 其余 TF")
print("=" * 92)
print(f'  名录命中 {is_master.sum()} / {len(rows)} 个（名录共 {len(MASTER)} 个，'
      f'{len(MASTER)-is_master.sum()} 个不在宇宙内或不在 TRRUST）')
print()
print(f'  {"指标":<22}{"谱系决定":>12}{"其余":>12}{"MW p":>12}')
sub = {}
for label, y, yname in [("TRRUST 总边数", tot, "tr_total"),
                        ("TRRUST 有向边数", dr, "tr_dir"),
                        ("ENCODE 边数", ene, "enc_edges")]:
    a, b = y[is_master], y[~is_master]
    uu = stats.mannwhitneyu(a, b, alternative="less")
    print(f'  {label:<22}{a.mean():>12.2f}{b.mean():>12.2f}{uu.pvalue:>12.3f}')
    sub[yname] = dict(master_mean=float(a.mean()), other_mean=float(b.mean()),
                      p_greater=float(uu.pvalue))
a, b = ine[is_master], ine[~is_master]
# 命中率比较用 Fisher
tab = np.array([[int(a.sum()), int(len(a)-a.sum())], [int(b.sum()), int(len(b)-b.sum())]])
odds, pf = stats.fisher_exact(tab)
print(f'  {"命中 ENCODE 比例":<22}{a.mean()*100:>11.1f}%{b.mean()*100:>11.1f}%{pf:>12.3f}')
sub["encode_hit"] = dict(master_rate=float(a.mean()), other_rate=float(b.mean()),
                         fisher_odds=float(odds), fisher_p=float(pf))

# 关键案例：>=3 有向靶（可做方向性检验）的比例
mask_testable = dr >= 6   # 两类各>=3 的近似下限
tm = mask_testable[is_master].mean(); to = mask_testable[~is_master].mean()
print(f'  {"有向边>=6 的比例":<22}{tm*100:>11.1f}%{to*100:>11.1f}%')

print()
print("名录中在本文数据里可用性最差的 TF（有向边 < 3）:")
bad = [r for r in rows if r["tf"] in MASTER and r["tr_dir"] < 3]
for r in sorted(bad, key=lambda x: (x["tr_dir"], -x["tau"])):
    print(f'  {r["tf"]:<9} TRRUST 总边={r["tr_total"]:>3}  有向={r["tr_dir"]:>2}  '
          f'ENCODE={"有" if r["in_encode"] else "无":>3}  tau={r["tau"]:.3f}')

out = dict(n_tf=len(rows), n_master_in=MASTER,
           trrust_total_tfs=len(TFSET), encode_total_tfs=len(enc_tfs),
           primary_tests=tests, secondary_tests=sub,
           n_master_matched=int(is_master.sum()),
           master_with_lt3_directed=[r["tf"] for r in bad],
           per_tf=rows)
p_ = os.path.join(OUT, "database_coverage_audit.json")
with open(p_, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(f'\n-> {p_}')
