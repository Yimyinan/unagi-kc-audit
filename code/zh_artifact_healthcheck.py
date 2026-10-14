# -*- coding: utf-8 -*-
"""中文产物体检：docx / pdf 的文本抽取 + LaTeX 残留 + 统计。

用法: python code/zh_artifact_healthcheck.py <file> [...]
支持 .docx (zip/word/document.xml) 与 .md/.tex (纯文本)。
"""
import io, os, re, sys, zipfile

CJK = re.compile(r"[\u4e00-\u9fff]")
# LaTeX 残留：命令、环境、数学定界符、转义符。
# ★ 一律用非捕获组 (?:...)：带捕获组的模式会让 re.findall 返回「组内容」而不是整个匹配，
#   计数看着对、样例却全是空串——这个坑已经踩过一次。
RESIDUE = {
    r"\\[a-zA-Z]+(?:\*)?(?:\{\})?": "LaTeX 命令 (\\cmd)",
    r"\\begin\{[^}]*\}": "\\begin{env}",
    r"\\end\{[^}]*\}": "\\end{env}",
    r"\\(?:cite|citep|citet|ref|eqref|label|autoref|Cref)\{[^}]*\}": "cite/ref/label",
    r"\\(?:input|include|includegraphics|includepdf)\b": "input/includegraphics",
    r"\\%": "转义百分号",
    r"\\_": "转义下划线",
    r"\$\$[^$]{2,}\$\$": "数学定界符 $$...$$",
    r"(?<![\\$])\$[^$\n]{2,}\$(?!\$)": "数学定界符 $...$",
}
HTMLESC = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#39;": "'"}
PLACEHOLDER = re.compile(r"\[(?:TODO|TBD|XXX|REPOSITORY[\\_]?URL|FIGSHARE[\\_]?DOI|"
                         r"AUTHOR NAME|AFFILIATION|待填|占位)[^\]]*\]", re.I)
# Word 自带的域代码/修订痕迹
FIELDS = {"MERGEFORMAT": "Word 域残留", "ADDIN": "Word 域残留",
          "PAGEREF": "页码域(正常)", "STYLEREF": "样式引用域(正常)", "TOC": "目录域(正常)",
          "HYPERLINK": "超链接域(正常)", "SEQ": "SEQ 域(正常)",
          "REF ": "REF 域(正常)", "w:ins ": "修订插入标记", "w:del ": "修订删除标记"}


def docx_text(path):
    """返回 (正文文本, zip, 域代码计数)。

    ★ 必须先剥掉 `w:instrText`（Word 域代码）再扫残留。域开关写成 `\\h` / `\\l` / `\\o` / `\\u`
      以及 WPS 生成的 `\\tdkey <rand>`，它们**不是 LaTeX**；不剥离就会把 68 个超链接域报成
      "68 处 LaTeX 残留"（这个误报类真的发生过，且样例打印还因捕获组 bug 显示成空串）。
    """
    z = zipfile.ZipFile(path)
    raw = z.read("word/document.xml").decode("utf-8", "replace")
    instr = re.findall(r"<w:instrText[^>]*>(.*?)</w:instrText>", raw, re.S)
    instr += re.findall(r"<w:fldSimple[^>]*w:instr=\"([^\"]*)\"", raw)
    body = re.sub(r"<w:instrText[^>]*>.*?</w:instrText>", "", raw, flags=re.S)
    body = re.sub(r"<w:fldSimple[^>]*>", "", body)
    # 段落边界保留换行语义
    body = re.sub(r"</w:p>", "\n", body)
    # 只取文本节点，避免把标签/属性当正文。
    # ★ 必须写作 <w:t> 或 <w:t ...>：裸的 <w:t[^>]*> 会匹配 <w:tabs> / <w:tab>，
    #   把 XML 片段当正文吞进来。
    txt = "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", body, re.S)) \
        or re.sub(r"<[^>]+>", "", body)
    for k, v in HTMLESC.items():
        txt = txt.replace(k, v)
    return txt, z, instr


def plain_text(path):
    return io.open(path, encoding="utf-8", errors="replace").read(), None, []


def report(path):
    print("=" * 78)
    print(os.path.basename(path))
    print("=" * 78)
    if path.endswith(".docx"):
        try:
            txt, z, instr = docx_text(path)
        except Exception as e:
            print("  读取失败:", e)
            return
    else:
        txt, z, instr = plain_text(path)
    cjk = len(CJK.findall(txt))
    print("  汉字数        : %d" % cjk)
    print("  正文字符数    : %d" % len(txt))
    print()
    print("  -- LaTeX 残留 --")
    total = 0
    for pat, label in RESIDUE.items():
        hits = re.findall(pat, txt)
        if hits:
            total += len(hits)
            from collections import Counter
            top = Counter(hits).most_common(5)
            print("    %-26s %5d" % (label, len(hits)))
            for h, c in top:
                print("        %4d x  %s" % (c, h.replace("\n", " ")[:70]))
    print("    合计残留                  %5d  %s" % (total, "OK" if total == 0 else "*** 需处理 ***"))
    print()
    print("  -- 占位符 --")
    ph = PLACEHOLDER.findall(txt)
    print("    未填占位            %5d  %s" % (len(ph), sorted(set(ph))[:6] if ph else "OK"))
    print()
    print("  -- Word 域（域代码已从正文剥离，仅作结构清点）--")
    if instr:
        from collections import Counter
        kinds = Counter(re.match(r"\s*([A-Z]+)", s).group(1) if re.match(r"\s*([A-Z]+)", s) else "?"
                        for s in instr)
        print("    域代码条数            %5d  %s" % (len(instr), dict(kinds)))
    else:
        print("    域代码条数            %5d" % 0)
    if z is not None:
        print()
        print("  -- docx 结构 --")
        print("    条目数              %5d" % len(z.namelist()))
        for n in ("word/comments.xml", "word/trackChanges.xml"):
            print("    %-20s %s" % (n, "存在" if n in z.namelist() else "无"))
        media = [n for n in z.namelist() if n.startswith("word/media/")]
        print("    嵌入图片            %5d" % len(media))
    print()


for p in sys.argv[1:]:
    if os.path.isfile(p):
        report(p)
    else:
        print("!! 不存在:", p)
