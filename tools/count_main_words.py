"""count_main_words.py — main.tex 正文字数统计（用于投稿字数预算）

口径：
  - 剥掉 preamble（\\begin{document} 之前）
  - 剥掉注释行
  - 剥掉数学环境内容（按投稿惯例，公式不计词）
  - 剥掉 figure/figure* 环境（图注单独统计）
  - 剥掉 LaTeX 命令与花括号
统计：正文词数 / 图注词数 / 章节明细
"""
import re, sys, os

ROOT = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic"
path = os.path.join(ROOT, "main.tex")
src = open(path, encoding="utf-8", errors="ignore").read()

# 取 document 环境
m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", src, re.S)
body = m.group(1) if m else src

# 去掉注释（保留转义 \%)
body = re.sub(r"(?<!\\)%.*", "", body)

# 抽出图/表环境，单独统计
figs = re.findall(r"\\begin\{(figure\*?|table\*?|table)\}(.*?)\\end\{\1\}", body, re.S)
fig_text = " ".join(f[1] for f in figs)
body_nofig = body
for f in figs:
    body_nofig = body_nofig.replace(f[1], " ", 1)

# 去掉 tabular 内容（表格数字不计词）
body_nofig = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body_nofig, flags=re.S)
body_nofig = re.sub(r"\\begin\{tabularx\}.*?\\end\{tabularx\}", " ", body_nofig, flags=re.S)

# 去掉独立数学环境
def strip_display_math(t):
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(equation|align|eqnarray|gather)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{boxednote\}.*?\n", " ", t)   # 保留 boxednote 文字，仅去环境标记
    return t
body_nofig = strip_display_math(body_nofig)
fig_text = strip_display_math(fig_text)

def clean(t):
    t = re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\end\{[^}]*\}", " ", t)
    t = re.sub(r"\\item\b", " ", t)
    t = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", t)   # 命令
    t = re.sub(r"\\[^a-zA-Z]", " ", t)                      # 转义符号
    t = re.sub(r"[{}~]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def wc(t):
    t = clean(t)
    # 英文按空格分词；中日韩字符逐个计为 1 词
    latin = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-\.,]*", t))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
    return latin + cjk

print("=" * 72)
print("main.tex 字数统计")
print("=" * 72)
print(f"正文（不含图/表/公式）      : {wc(body_nofig):>6} 词")
print(f"图注 + 表格文字             : {wc(fig_text):>6} 词")
print(f"合计（含图表）              : {wc(body_nofig) + wc(fig_text):>6} 词")

print()
print("章节明细：")
# 按 \section 切分
parts = re.split(r"(\\section\{[^}]*\})", body_nofig)
cur = "（前言/摘要）"
rows = []
for p in parts:
    mm = re.match(r"\\section\{([^}]*)\}", p)
    if mm:
        cur = mm.group(1)
        continue
    n = wc(p)
    if n:
        rows.append((cur, n))
# 合并同名
agg = {}
for k, v in rows:
    agg[k] = agg.get(k, 0) + v
for k, v in agg.items():
    kk = re.sub(r"\\(cite|ref|label)\{[^}]*\}", "", k)
    kk = clean(kk)
    print(f"  {kk:<52} {v:>6}")
print(f"  {'— 正文合计':<52} {sum(agg.values()):>6}")

abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
if abstract:
    print(f"\n摘要单独计: {wc(abstract.group(1))} 词")
