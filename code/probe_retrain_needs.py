"""probe_retrain_needs.py — 回答"是否必须重训全量 PG-UNAGI"

三个可判定问题：
  Q1 现有 33 个审计配置中，哪些是"全量三增强"臂？  (区分 arm 类型)
  Q2 全量 PG-UNAGI 单次训练成本实测是多少？           (从 run_info + 时间戳反推)
  Q3 是否存在"部分训练即可回答方向/强度问题"的替代设计？
"""
import os, glob, re, json, time, datetime

UNAGI_RES = r"F:/UNAGI_Project/UNAGI/result"
SRC = r"F:/UNAGI_Project/UNAGI/pg_unagi"

# ---------- Q2: 从 run_info + log 反推成本 ----------
RUNS = {
    "R0_vanilla (max_iter=8)":      "R0/R0_vanilla_unagi_20260530_223153",
    "R1_neural (21k x 11.9k)":      "R1/R1_neural_seed123",
    "R11 (2 TF, 30k x 20.5k)":      "R11/R11_KLF1_TAL1_seed42",
    "R11 seed123":                  "R11/R11_KLF1_TAL1_seed123",
    "R11 seed456":                  "R11/R11_KLF1_TAL1_seed456",
    "R12 (3 TF) seed42":            "R12/R12_GATA2_GFI1B_PAX5_seed42",
    "R13 (11 TF, full-arm) seed42": "R13/R13_hemato_trueprior_seed42",
    "R13 seed123":                  "R13/R13_hemato_trueprior_seed123",
    "R13 seed456":                  "R13/R13_hemato_trueprior_seed456",
}

def probe(d):
    full = os.path.join(UNAGI_RES, d)
    start = cells = None
    for lg in glob.glob(full + "/*.log"):
        try:
            for line in open(lg, errors="ignore"):
                m = re.search(r"Training started at ([\d\- :]+)", line)
                if m:
                    start = m.group(1).strip()
                m2 = re.search(r"Cells: ([\d,]+), Genes: ([\d,]+)", line)
                if m2:
                    cells = (m2.group(1), m2.group(2))
        except Exception:
            pass
    n_iter = len(glob.glob(full + "/iter_*"))
    # epoch 总数 -> 训练量代理
    n_epoch = 0
    for lg in glob.glob(full + "/*.log"):
        try:
            for line in open(lg, errors="ignore"):
                if line.startswith("[epoch ") and "average training loss" in line:
                    n_epoch += 1
        except Exception:
            pass
    last = max((os.path.getmtime(p) for p in
                glob.glob(full + "/iter_*/*") + glob.glob(full + "/iter_*/*/*")
                + glob.glob(full + "/*.log")), default=None)
    dur = None
    if start and last:
        try:
            t0 = time.mktime(time.strptime(start, "%Y-%m-%d %H:%M:%S"))
            dur = (last - t0) / 3600.0
        except Exception:
            pass
    return dict(dir=d, start=start, cells=cells, iters=n_iter,
                epochs=n_epoch, end=(datetime.datetime.fromtimestamp(last)
                                     .strftime("%Y-%m-%d %H:%M") if last else None),
                hours=(round(dur, 2) if dur is not None else None))

print("=" * 100)
print("Q2  全量 PG-UNAGI 单次训练成本（实测，从 run_info 起始时间戳到末次产物写入）")
print("=" * 100)
print(f'{"run":<30}{"cells":>9}{"genes":>8}{"iter":>6}{"epoch":>7}{"start":>18}{"end":>17}{"hours":>8}')
costs = {}
for label, d in RUNS.items():
    r = probe(d)
    costs[label] = r
    cells, genes = (r["cells"] if r["cells"] else ("-", "-"))
    print(f'{label:<30}{cells:>9}{genes:>8}{r["iters"]:>6}{r["epochs"]:>7}'
          f'{str(r["start"]):>18}{str(r["end"]):>17}{str(r["hours"]):>8}')

val = [c["hours"] for c in costs.values() if c["hours"]]
print(f'\n单次训练墙钟: min={min(val):.2f}h  median={sorted(val)[len(val)//2]:.2f}h  max={max(val):.2f}h  n={len(val)}')

# ---------- Q3: 是否有 partial 路径 ----------
print()
print("=" * 100)
print("Q3  成本构成拆分：iDREM(Java 子进程) vs VAE 训练")
print("=" * 100)
for label in ["R11 (2 TF, 30k x 20.5k)", "R13 (11 TF, full-arm) seed42"]:
    d = os.path.join(UNAGI_RES, RUNS[label])
    logs = glob.glob(d + "/*.log")
    if not logs:
        continue
    idrem_marks = []
    for lg in logs:
        for i, line in enumerate(open(lg, errors="ignore")):
            if "idrem Done" in line:
                idrem_marks.append(i)
    print(f'{label:<30} iDREM 完成标记次数 = {len(idrem_marks)}  (每 iter 一次 => {len(idrem_marks)} 次 Java 调用)')

# idrem.jar 是否存在
jar = glob.glob(r"F:/UNAGI_Project/**/idrem.jar", recursive=True)
print(f'\nidrem.jar 在本机: {"存在 " + jar[0] if jar else "不存在"}')
pyrep = r"F:/UNAGI_Project/UNAGI/UNAGI_source/UNAGI/dynamic_regulatory_networks/idrem_python.py"
print(f'纯 Python 替代实现: {"存在" if os.path.exists(pyrep) else "不存在"}  ({pyrep})')
print(f'Java 运行时: ', end="")
os.system("java -version 2>&1 | head -1")

# ---------- 输出 JSON ----------
out = {"costs": costs,
       "single_run_hours": {"min": min(val), "median": sorted(val)[len(val)//2], "max": max(val), "n": len(val)},
       "idrem_jar_present": bool(jar),
       "python_idrem_replacement_present": os.path.exists(pyrep)}
p = r"F:/UNAGI_Project/outputs/positive_result/retrain_cost_probe.json"
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n-> {p}')
