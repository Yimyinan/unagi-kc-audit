# -*- coding: utf-8 -*-
"""Unit-test the filename-collision guard in figures/generate_main_figures.py.

The script cannot be imported (it executes a full plotting pipeline that needs data files), so
extract just `OWNED_ELSEWHERE` + `save()` with ast and exercise them against a stub figure.
A file-existence check in a temp dir proves nothing when the script dies before reaching save().
"""
import ast
import io
import os

SRC = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/figures/generate_main_figures.py"
tree = ast.parse(io.open(SRC, encoding="utf-8").read())

ns = {"os": os, "BASE": r"C:/tmp/figs", "plt": type("P", (), {"close": staticmethod(lambda *a: None)})}
# `save()` calls fp.fit_to_width(...); figure_palette cannot be imported here (it needs matplotlib
# state), so stub the one method the guard path touches.  Without this the test dies with
# NameError before it can assert anything.
ns["fp"] = type("FP", (), {"fit_to_width": staticmethod(lambda *a, **k: None)})
got = []
class StubFig:
    def savefig(self, path, **kw):
        got.append(os.path.basename(path))
ns["__builtins__"] = __builtins__

for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("OWNED_ELSEWHERE", "_GEN_W"):
        exec(compile(ast.Module([node], []), "<guard>", "exec"), ns)
    if isinstance(node, ast.FunctionDef) and node.name == "save":
        exec(compile(ast.Module([node], []), "<guard>", "exec"), ns)

print("guard table:", ns.get("OWNED_ELSEWHERE"))
print()

CASES = [("fig2_kc_phase_diagram", True), ("fig4_prior_activity", True),
         ("fig1_coadaptation", False), ("fig3_purity_falsification", False)]
ok = True
for name, should_redirect in CASES:
    got.clear()
    ns["save"](StubFig(), name)
    written = [g for g in got if g.endswith(".pdf")][0].replace(".pdf", "")
    redirected = written.endswith("__UMBRELLA_LEGACY_DO_NOT_USE")
    good = (redirected == should_redirect)
    ok &= good
    print("%-4s %-26s -> %-44s %s" % ("PASS" if good else "FAIL", name, written,
          "redirected as required" if should_redirect else "passed through as required"))

print()
print("UNIT TEST:", "PASS -- guard redirects only the collision names" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
