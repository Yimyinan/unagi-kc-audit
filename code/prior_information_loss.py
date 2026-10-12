"""prior_information_loss.py — 量化先验信息在 UNAGI 数据流中的损失

回答：TRRUST 里带方向/置信度的先验，有多少信息真正进入 geneWeight？

数据流（已从源码走通）：
  构建期:  build_tf_database(HEMATOPOIETIC_TFS) -> tf_db {TF: [targets]}   [方向被丢弃]
  注入期:  build_prior_matrix(tf_db) -> per-gene scalar vector             [拓扑被丢弃]
  运行期:  runner.run() -> run_IDREM() -> getTFs(idremResults)
                          -> matchTFandTGWithFoldChange(..., human_encode.txt, ...)
                          -> updataGeneTablesWithDecay() -> geneWeight      [换成 ENCODE]
  损失期:  trainer.train_model(geneWeights=...) -> loss_function(gene_weights=...)
  但:      geneWeight nnz = 0 (40/40 文件) => 该通路实际未激活

输出：每一步的信息保留率
"""
import os, re, json, pickle, collections

F = r"F:/UNAGI_Project"
TRRUST = os.path.join(F, "unagi_custom_data/trrust_raw_human.txt")
ENCODE = os.path.join(F, "UNAGI/UNAGI_source/UNAGI/data/human_encode.txt")
PKL = os.path.join(F, "unagi_custom_data/tf_prior_database_human.pkl")
OUT = r"F:/UNAGI_Project/outputs/positive_result"

# ---------- 1. TRRUST 原始信息量 ----------
trrust = collections.defaultdict(list)
n_trrust = 0
dir_count = collections.Counter()
with open(TRRUST, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        tf, target, direction, pmid = parts[0], parts[1], parts[2], parts[3]
        trrust[tf].append((target, direction, pmid))
        dir_count[direction] += 1
        n_trrust += 1

print("=" * 90)
print("第 1 步：TRRUST 原始库")
print("=" * 90)
print(f"  边数 total        : {n_trrust:,}")
print(f"  唯一 TF 数        : {len(trrust):,}")
print(f"  方向分布          : {dict(dir_count)}")
n_dir_known = dir_count.get("Activation", 0) + dir_count.get("Repression", 0)
print(f"  带方向信息的边     : {n_dir_known:,}  ({100*n_dir_known/n_trrust:.1f}%)")
print(f"  带 PMID 的边       : 全量（每条边都有 PMID）")

# ---------- 2. CORE_HEMATO 子集 ----------
HEMATO_TFS = None
cfg = os.path.join(F, "UNAGI/pg_unagi/config.py")
if os.path.exists(cfg):
    txt = open(cfg, encoding="utf-8", errors="ignore").read()
    m = re.search(r"HEMATOPOIETIC_TFS\s*=\s*\[(.*?)\]", txt, re.S)
    if m:
        HEMATO_TFS = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
print()
print("=" * 90)
print("第 2 步：注入用的 TF 子集（CORE_HEMATO）")
print("=" * 90)
if HEMATO_TFS:
    print(f"  TF 列表 ({len(HEMATO_TFS)}): {HEMATO_TFS}")
    sub_edges = sum(len(trrust.get(t, [])) for t in HEMATO_TFS)
    sub_dir = sum(1 for t in HEMATO_TFS for (_, d, _) in trrust.get(t, [])
                  if d in ("Activation", "Repression"))
    print(f"  该子集在 TRRUST 中的边数 : {sub_edges:,}")
    print(f"  其中带方向               : {sub_dir:,}  ({100*sub_dir/max(sub_edges,1):.1f}%)")
else:
    print("  (未能解析 config.py)")

# ---------- 3. 实际注入的信息量 ----------
# 从 R13 日志已知：11 TF, 325 edges, boost 203/20466 genes (1.0%)
print()
print("=" * 90)
print("第 3 步：实际进入模型的量（R13 实测，来自日志）")
print("=" * 90)
print("  日志: 'TF prior database: 11 TFs, 325 edges'")
print("  日志: 'Prior boost: 203/20466 genes (1.0%)'")
print(f"  注入边数 325 -> 覆盖基因 203 个（去重后）")
print(f"  进入 geneWeight 的 TF 数 = 0/11（geneWeight 全零，40/40 文件 nnz=0）")

# ---------- 4. 被替换掉的参考库 ----------
print()
print("=" * 90)
print("第 4 步：运行期的参考库替换（这是方向信息消失的环节）")
print("=" * 90)
n_encode = 0
encode_tfs = set()
with open(ENCODE, encoding="utf-8", errors="ignore") as f:
    next(f, None)
    for line in f:
        parts = line.split("\t")
        if len(parts) >= 3:
            encode_tfs.add(parts[0])
            n_encode += 1
print(f"  human_encode.txt 边数 : {n_encode:,}   (TRRUST 的 {n_encode/n_trrust:.1f}x)")
print(f"  ENCODE 唯一 TF 数     : {len(encode_tfs):,}")
print(f"  ENCODE 列             : TF, Gene, Input  -> 无 direction 列，无 PMID 列")
if HEMATO_TFS:
    hit = [t for t in HEMATO_TFS if t in encode_tfs]
    print(f"  CORE_HEMATO 的 {len(HEMATO_TFS)} 个 TF 在 ENCODE 中命中 : {len(hit)}/{len(HEMATO_TFS)}")
    print(f"    -> {hit}")

# ---------- 5. 信息保留率总结 ----------
print()
print("=" * 90)
print("信息保留率（逐环节）")
print("=" * 90)
rows = [
    ("TRRUST 原始",              f"{n_trrust:,} 边 / {len(trrust)} TF", "方向 + PMID + 拓扑", "100%"),
    ("build_tf_database()",      "325 边 / 11 TF",                      "仅 target 列表",     f"{100*325/n_trrust:.1f}% 边 / {100*11/len(trrust):.1f}% TF"),
    ("build_prior_matrix()",     "203 基因标量",                        "无方向/无拓扑",       "塌缩为 per-gene 标量"),
    ("run_IDREM() + getTFs()",   "branch-point 富集",                   "先验仅影响富集",      "间接"),
    ("matchTFandTG(ENCODE)",     f"{n_encode:,} 边",                    "换成 ENCODE 参考",    "TRRUST 方向在此彻底丢失"),
    ("geneWeight (实测)",        "nnz = 0",                             "—",                   "0%"),
    ("loss_function(gene_weights)", "分支存在但实例化 0 次",             "—",                   "0%"),
]
print(f'{"环节":<32}{"规模":<28}{"保留的信息":<24}{"保留率"}')
for r in rows:
    print(f"{r[0]:<32}{r[1]:<28}{r[2]:<24}{r[3]}")

# ---------- 输出 ----------
out = {
    "trrust": {"n_edges": n_trrust, "n_tfs": len(trrust), "direction_counts": dict(dir_count),
               "n_direction_known": n_dir_known},
    "core_hemato": {"tfs": HEMATO_TFS,
                    "n_edges_in_trrust": sub_edges if HEMATO_TFS else None,
                    "n_edges_with_direction": sub_dir if HEMATO_TFS else None},
    "r13_measured": {"tfs": 11, "edges": 325, "genes_boosted": 203, "n_genes": 20466,
                     "geneWeight_nnz": 0},
    "encode": {"n_edges": n_encode, "n_tfs": len(encode_tfs),
               "has_direction_column": False, "has_pmid_column": False},
    "retention": [
        {"stage": r[0], "scale": r[1], "retained": r[2], "retention": r[3]} for r in rows
    ],
}
p = os.path.join(OUT, "prior_information_loss.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n-> {p}")
