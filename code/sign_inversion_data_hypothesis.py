"""sign_inversion_data_hypothesis.py — 符号反号：模型跟随数据，还是跟随数据库？

背景（sign_inversion_diagnostic.py 已排除两个解释）：
  H1 生物学（诱导型 TF 成类）: Fisher OR=1.39, p=1.00  -> 否
  H2 数据质量（低表达噪声）  : MW p=0.85 / 1.00         -> 否
符号反号的成因仍然未知。

本脚本检验 H3（新假说）：
  模型解码器几何的符号反映的是**数据里的共表达结构**，而不是 TRRUST 的调控标注。
  若成立，则"反号"不是模型出错，而是 TRRUST 标注与该数据集的语境不符 ——
  与"ENCODE 缺 6/12 造血 TF"是同一个故事的另一面。

设计：
  对每个 TF t：
    S_model = mean_{g in A} cos(W[:,g], W[:,t]) - mean_{g in R} cos(W[:,g], W[:,t])
      （已有，来自 latent_directionality_final.json）
    S_data  = mean_{g in A} corr(x_t, x_g)   - mean_{g in R} corr(x_t, x_g)
      （在数据集里直接算 TF 与靶基因的经验共表达；corr 是尺度不变的）
    TRRUST 隐含符号恒为 +（A 应为正、R 应为负）

  预测（H3）：sign(S_model) 与 sign(S_data) 高度一致，
              而与 TRRUST 隐含符号（恒 +）不一致。

零训练成本。
"""
import os, json
import numpy as np
import scipy.sparse as sp
from scipy import stats

OUT = r"F:/UNAGI_Project/outputs/positive_result"
UNIV = r"F:/UNAGI_Project/UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/0.h5ad"
DATA = r"F:/UNAGI_Project/UNAGI/result/R11/R11_KLF1_TAL1_seed42/iter_0/stagedata/dataset.h5ad"
TRRUST = r"F:/UNAGI_Project/unagi_custom_data/trrust_raw_human.txt"

import anndata as ad
a0 = ad.read_h5ad(UNIV)
GENES = [str(g).upper() for g in a0.var_names]
GI = {g: i for i, g in enumerate(GENES)}

J = json.load(open(os.path.join(OUT, "latent_directionality_final.json"), encoding="utf-8"))
per_tf_json = {s["tf"]: s for s in J["per_tf"]}

# TRRUST 靶集合（与主分析同口径：只保留宇宙内基因，每类 >= 3）
trrust = {}
for line in open(TRRUST, encoding="utf-8", errors="ignore"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 4:
        continue
    tf, tgt, d = p[0].upper(), p[1].upper(), p[2]
    if d not in ("Activation", "Repression"):
        continue
    trrust.setdefault(tf, {"A": set(), "R": set()})["A" if d == "Activation" else "R"].add(tgt)

print("载入表达矩阵 ...")
adata = ad.read_h5ad(DATA)
X = adata.X
if not sp.issparse(X):
    X = sp.csr_matrix(X)
GENES_DATA = [str(g).upper() for g in adata.var_names]
assert GENES_DATA == GENES, "基因顺序不一致"

def col_corr(idx_a, idx_b):
    """单基因 vs 多基因的 Pearson 相关（稀疏安全的 densify 子集）"""
    A = X[:, idx_a].toarray().ravel().astype(np.float64)
    B = X[:, idx_b].toarray().astype(np.float64)
    A -= A.mean()
    B -= B.mean(axis=0, keepdims=True)
    na = np.sqrt((A * A).sum())
    nb = np.sqrt((B * B).sum(axis=0))
    nb[nb == 0] = 1.0
    denom = na * nb
    denom[denom == 0] = 1.0
    return (A @ B) / denom

rows = []
for tf in per_tf_json:
    if tf not in GI or tf not in trrust:
        continue
    A = sorted([GI[g] for g in trrust[tf]["A"] if g in GI])
    R = sorted([GI[g] for g in trrust[tf]["R"] if g in GI])
    if len(A) < 3 or len(R) < 3:
        continue
    cA = col_corr(GI[tf], A)
    cR = col_corr(GI[tf], R)
    s_data = float(cA.mean() - cR.mean())
    s_model = per_tf_json[tf]["mean_s1"]
    cons = per_tf_json[tf]["sign_consistent"]
    rows.append(dict(tf=tf, s_model=s_model, s_data=s_data, sign_consistent=bool(cons),
                     nA=len(A), nR=len(R),
                     data_A=float(cA.mean()), data_R=float(cR.mean())))

print(f"\n可评估 TF: {len(rows)}")

s_model = np.array([r["s_model"] for r in rows])
s_data = np.array([r["s_data"] for r in rows])
locked = np.array([r["sign_consistent"] for r in rows])

def report(mask, label):
    m, d = s_model[mask], s_data[mask]
    if len(m) < 3:
        print(f"  {label}: n={len(m)} 太少"); return None
    agree = (np.sign(m) == np.sign(d))
    r, p = stats.pearsonr(m, d)
    rho, prho = stats.spearmanr(m, d)
    print(f'  {label:<26} n={len(m):>3}  符号一致率={agree.mean()*100:>5.1f}%'
          f'  Pearson r={r:+.3f} (p={p:.2e})  Spearman ρ={rho:+.3f}')
    return dict(n=int(len(m)), agree=float(agree.mean()), pearson_r=float(r),
                pearson_p=float(p), spearman_rho=float(rho), spearman_p=float(prho))

print()
print("=" * 92)
print("H3 检验：模型符号 vs 数据经验共表达符号")
print("=" * 92)
allr = report(np.ones(len(rows), bool), "全部可评估 TF")
lockr = report(locked, "符号锁定 TF（主分析关注集）")
unlockr = report(~locked, "非锁定 TF（对照）")

# 分正负号看
posm = locked & (s_model > 0)
negm = locked & (s_model < 0)
print()
print(f'  锁定的正号 TF (n={int(posm.sum())}): 数据符号为正的占 {100*(s_data[posm]>0).mean():.0f}%')
print(f'  锁定的负号 TF (n={int(negm.sum())}): 数据符号为负的占 {100*(s_data[negm]<0).mean():.0f}%')

print()
print("逐 TF（按 |S_data| 排序，前 20）:")
print(f'  {"TF":<9}{"锁定":>5}{"S_model":>10}{"S_data":>10}{"符号一致":>9}{"data_corr_A":>12}{"data_corr_R":>12}')
for r in sorted(rows, key=lambda x: -abs(x["s_data"]))[:20]:
    ok = np.sign(r["s_model"]) == np.sign(r["s_data"])
    print(f'  {r["tf"]:<9}{("是" if r["sign_consistent"] else ""):>5}{r["s_model"]:>+10.4f}'
          f'{r["s_data"]:>+10.4f}{("✓" if ok else "✗"):>9}{r["data_A"]:>+12.4f}{r["data_R"]:>+12.4f}')

print()
print("重点：17 个符号锁定的负号 TF，其数据符号如何？")
for r in sorted([r for r in rows if r["sign_consistent"] and r["s_model"] < 0],
                key=lambda x: x["s_data"]):
    ok = r["s_data"] < 0
    print(f'  {r["tf"]:<9}S_model={r["s_model"]:+.4f}  S_data={r["s_data"]:+.4f}  '
          f'{"数据也负 ✓" if ok else "数据为正 ✗"}')

out = dict(all=allr, locked=lockr, unlocked=unlockr,
           frac_data_same_sign_when_locked_neg=float((s_data[negm] < 0).mean()),
           frac_data_same_sign_when_locked_pos=float((s_data[posm] > 0).mean()),
           per_tf=rows)
p_ = os.path.join(OUT, "sign_inversion_data_hypothesis.json")
with open(p_, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print(f'\n-> {p_}')
