"""arm_provenance_table.py — 把每个审计臂映射到它的训练日志与先验配置

目的（回应用户信息：r14_hemato_seed42/123/456 其实是 R15，后命名）：
  审计矩阵 R14_s*.npy / R15_s*.npy 到底由哪一次运行产出？
  各臂的先验配置（TF 数 / boost 基因数）分别是什么？

做法：扫描 UNAGI/result 下所有含 "TFs in prior database" 或 "Prior boost" 的日志，
      抽取 (日志路径, 臂名, TF 数, boost 基因数, TF 索引匹配, 训练开始时间)，
      再与 decoder_unified/loading 下的矩阵名对照。
"""
import os, re, glob, json
import collections

RES = r"F:/UNAGI_Project/UNAGI/result"
LOAD = r"F:/UNAGI_Project/UNAGI/decoder_unified/loading"

rows = []
for lg in glob.glob(os.path.join(RES, "**", "*.log"), recursive=True):
    try:
        txt = open(lg, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    if "prior" not in txt.lower():
        continue
    tf_db = re.search(r"(\d+)\s+TFs in prior database", txt)
    tf_db2 = re.search(r"TF prior database:\s*(\d+)\s*TFs,\s*(\d+)\s*edges", txt)
    boost = re.search(r"Prior boost:\s*([\d,]+)/(\d+)\s*genes\s*\(([\d.]+)%\)", txt)
    idx = re.search(r"TF indices:\s*(\d+)/(\d+)\s*matched", txt)
    start = re.search(r"Training started at ([\d\- :]+)", txt)
    named = re.search(r"(TruePrior|WrongPrior|R11/R12 Subset|Subset)[^\n]*", txt)
    rows.append(dict(
        log=os.path.relpath(lg, RES).replace("\\", "/"),
        tf_db=int(tf_db.group(1)) if tf_db else (int(tf_db2.group(1)) if tf_db2 else None),
        edges=int(tf_db2.group(2)) if tf_db2 else None,
        boost_genes=int(boost.group(1).replace(",", "")) if boost else None,
        boost_pct=float(boost.group(3)) if boost else None,
        tf_idx=f"{idx.group(1)}/{idx.group(2)}" if idx else None,
        start=start.group(1).strip() if start else None,
        tag=named.group(1) if named else None,
    ))

# 去重（同一日志有多份副本）
key = lambda r: (r["tf_db"], r["edges"], r["boost_genes"], r["tf_idx"], r["start"])
byk = collections.defaultdict(list)
for r in rows:
    byk[key(r)].append(r)

print("=" * 118)
print("各臂的先验配置（按签名去重；列出所有副本路径）")
print("=" * 118)
print(f'{"TF数":>5}{"边数":>7}{"boost基因":>11}{"占比":>8}{"TF索引":>9}  {"开始时间":<21} 副本')
sigs = sorted(byk.items(), key=lambda kv: (kv[0][2] is None, kv[0][2] or 0), reverse=True)
for (tf, ed, bg, ti, st), lst in sigs:
    pct = next((r["boost_pct"] for r in lst if r["boost_pct"] is not None), None)
    print(f'{str(tf):>5}{str(ed):>7}{str(bg):>11}{(f"{pct}%" if pct else "-"):>8}{str(ti):>9}  {str(st):<21} {len(lst)}')
    for r in lst[:6]:
        print(f'{"":>52}└ {r["log"]}')
    if len(lst) > 6:
        print(f'{"":>52}└ … 另 {len(lst)-6} 份')

# 矩阵名 → 可能的臂
print()
print("=" * 118)
print("decoder_unified/loading 下的矩阵（审计用）")
print("=" * 118)
mats = sorted(os.path.basename(p).replace(".npy", "") for p in glob.glob(os.path.join(LOAD, "*.npy")))
print("  " + ", ".join(mats))

print()
print("=" * 118)
print("关键问题：R14 家族的矩阵对应哪一行？")
print("=" * 118)
print("  已知（用户确认）：r14_hemato_seed42/123/456 这三份运行 = R15（后命名）")
print("  已知（registry）：R15_s* 与 R14_s* 字节等同/同行空间")
print("  => 若 R14_s* 与 R15_s* 是同一读出的两个名字，则审计所谓的 'R14 空先验' 臂")
print("     实际继承的是上表中 boost=1054 基因那份运行的产物。")
print("  => 上表中唯一 boost 较小的臂是 WrongPrior（88 基因），那对应 R16，不是 R14。")

out = dict(signatures=[dict(tf_db=k[0], edges=k[1], boost_genes=k[2], tf_idx=k[3], start=k[4],
                            n_copies=len(v), logs=[r["log"] for r in v]) for k, v in sigs],
           matrices=mats)
p = r"F:/UNAGI_Project/outputs/positive_result/arm_provenance_table.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n-> {p}")
