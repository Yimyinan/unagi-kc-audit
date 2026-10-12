# -*- coding: utf-8 -*-
"""prior_degree_matched_control.py —— 度匹配（保度重布线）对照。

问题
----
先验注入的收益，究竟来自先验**具体的接线内容**，还是只来自它带来的**稀疏/度结构**？
这正是 Caranzano 等人（arXiv:2505.04300）在通路先验上提出的对照的核心问题。
本脚本给出**不需要重新训练**的网络层版本：把先验的 TF→靶基因二部图做
**保度重布线**（固定每个 TF 的出度、每个靶基因的入度），再比较观测量与零分布。

统计量
------
① 靶集两两 Jaccard 的均值（11 个 TF 共 C(11,2)=55 对）—— 共调控一致性；
② 被 ≥3 个 TF 共同靶向的基因数 —— 模块核心规模。

★ 非循环性说明
---------------
curated CORE_HEMATO 是**人工整理**的 TF→TF 边，且它本身被并入先验；若把它算进统计量
就是自证。故本对照**只用 TRRUST 来源的 TF→非TF 边**（排除 TF→TF 边），保持问题开放：
"这 11 个造血 TF 在 TRRUST 中的靶集，是否比保度随机接线更重叠？"

输入
----
unagi_custom_data/trrust_raw_human.txt  （TRRUST v2 human，原始 9,396 条 / 795 TF；
                                          去除 27 个自环后 **9,369 条 / 794 TF / 2,491 靶**）
code/reconstruct_prior_configs.py       （HEMATOPOIETIC_TFS 定义）

输出
----
data/prior_degree_matched_control.json
"""
import io, json, os, random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRRUST = os.path.join(BASE, "unagi_custom_data", "trrust_raw_human.txt")
OUT = os.path.join(BASE, "data", "prior_degree_matched_control.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

HEMATOPOIETIC_TFS = ["GATA2", "GATA1", "KLF1", "SPI1", "CEBPA", "PAX5",
                     "RUNX1", "TAL1", "LMO2", "GFI1", "GFI1B"]
TF_SET = set(HEMATOPOIETIC_TFS)
N_NULL, SWAP_FACTOR, SEED = 200, 10, 20260922


# ---------------------------------------------------------------- 读 TRRUST
edges, all_tfs, all_genes = [], set(), set()
for ln in io.open(TRRUST, encoding="utf-8", errors="replace"):
    p = ln.rstrip("\n").split("\t")
    if len(p) < 3:
        continue
    tf, tgt = p[0].strip(), p[1].strip()
    if not tf or not tgt or tf == tgt:
        continue
    edges.append((tf, tgt))
    all_tfs.add(tf); all_genes.add(tgt)
print("TRRUST: %d 条边 | %d TF | %d 靶基因" % (len(edges), len(all_tfs), len(all_genes)))

# 先验的 TRRUST 部分：11 个造血 TF 的靶集，**排除 TF→TF 边**（避免与 CORE_HEMATO 自证）
prior = {tf: set() for tf in HEMATOPOIETIC_TFS}
for tf, tgt in edges:
    if tf in TF_SET and tgt not in TF_SET:
        prior[tf].add(tgt)
E = [(tf, g) for tf in HEMATOPOIETIC_TFS for g in sorted(prior[tf])]
print("先验（TRRUST 来源、非 TF→TF）: %d 条边 | %d 个靶基因"
      % (len(E), len({g for _t, g in E})))
for tf in HEMATOPOIETIC_TFS:
    print("   %-6s %3d 个靶基因" % (tf, len(prior[tf])))


# ---------------------------------------------------------------- 统计量
def stats(esets):
    js = []
    for i in range(len(HEMATOPOIETIC_TFS)):
        for j in range(i + 1, len(HEMATOPOIETIC_TFS)):
            a, b = esets[HEMATOPOIETIC_TFS[i]], esets[HEMATOPOIETIC_TFS[j]]
            u = len(a | b)
            js.append(len(a & b) / u if u else 0.0)
    from collections import Counter
    c = Counter()
    for tf in HEMATOPOIETIC_TFS:
        for g in esets[tf]:
            c[g] += 1
    core3 = sum(1 for g, k in c.items() if k >= 3)
    return sum(js) / len(js), core3


def rewired(edges, rng):
    """保度重布线：保留每个 TF 的出度与每个靶基因的入度。"""
    es = set(edges)
    tfs = sorted({t for t, _ in es}); gl = sorted({g for _, g in es})
    for _ in range(SWAP_FACTOR * len(es)):
        (t1, g1), (t2, g2) = rng.sample(sorted(es), 2)
        if t1 == t2 or g1 == g2:
            continue
        if (t1, g2) in es or (t2, g1) in es:
            continue
        es.discard((t1, g1)); es.discard((t2, g2))
        es.add((t1, g2)); es.add((t2, g1))
    out = {t: set() for t in tfs}
    for t, g in es:
        out[t].add(g)
    return out


obs_j, obs_core = stats(prior)
rng = random.Random(SEED)
null_j, null_core = [], []
for _ in range(N_NULL):
    r = rewired(E, rng)
    j, c = stats({**{t: set() for t in HEMATOPOIETIC_TFS}, **r})
    null_j.append(j); null_core.append(c)

def summ(obs, null):
    m = sum(null) / len(null)
    sd = (sum((x - m) ** 2 for x in null) / (len(null) - 1)) ** 0.5
    z = (obs - m) / sd if sd else float("nan")
    ge = sum(1 for x in null if x >= obs)
    return {"observed": obs, "null_mean": m, "null_sd": sd, "z": z,
            "p_empirical": (ge + 1) / (len(null) + 1), "n_null": len(null)}

res = {"trrust_edges_total": len(edges), "n_tfs": len(HEMATOPOIETIC_TFS),
       "prior_edges_trrust_only": len(E),
       "prior_targets_trrust_only": len({g for _t, g in E}),
       "stat_jaccard": summ(obs_j, null_j),
       "stat_core_ge3": summ(obs_core, null_core),
       "note": "保度重布线保留每个 TF 的出度与每个靶基因的入度；统计量只用 TRRUST 来源的非 TF→TF 边，"
               "故不与 curated CORE_HEMATO 自证。p 为经验值（含 +1 平滑）。"}
io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(res, ensure_ascii=False, indent=2))

print()
print("=" * 62)
print("保度重布线对照（N_null=%d）" % N_NULL)
print("=" * 62)
for k, lab in (("stat_jaccard", "靶集两两 Jaccard 均值"), ("stat_core_ge3", "被≥3个TF共同靶向的基因数")):
    s = res[k]
    print("%-22s 观测 %.4f | 零假设 %.4f ± %.4f | z=%.2f | p=%.4f"
          % (lab, s["observed"], s["null_mean"], s["null_sd"], s["z"], s["p_empirical"]))
print()
print("wrote", OUT)
