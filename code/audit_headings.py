import io, os, re, sys
D = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic"
Q = re.compile(r"(Why|How|What|Is|Are|Does|Do|Can|Should|Which|When)\b")
SKIP = ("/archive/", "/snapshots/", "/releases/", "/build/", "/_removed_/", "/zh_tex/", "/submission/")
bad = []
for root, _dd, fs in os.walk(D):
    rp = root.replace("\\", "/") + "/"
    if any(x in rp for x in SKIP):
        continue
    for f in fs:
        if not f.endswith(".tex"):
            continue
        p = os.path.join(root, f)
        s = io.open(p, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\\(section|subsection|paragraph)\*?\{([^}]*)\}", s):
            t = m.group(2).strip()
            if t.endswith("?") or (Q.match(t) and not t.lower().startswith("whether")):
                bad.append((p.replace(D + os.sep, ""), s[:m.start()].count("\n") + 1, t))
print("  [audit_headings] 疑问式标题 %d 处" % len(bad))
for b in bad:
    print("     %s L%d  %s" % b)
print("  VERDICT: %s" % ("CLEAN -- no interrogative heading" if not bad else "FAIL"))
sys.exit(1 if bad else 0)
