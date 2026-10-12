"""Tests for the injection-path check and its external-facing CLI.

These build two synthetic pipelines in a temp directory -- one that delivers its prior
and one that does not -- so the check is exercised in BOTH directions. A check that has
only ever been seen failing is not evidence that it can pass.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

h5py = pytest.importorskip("h5py")
import numpy as np                                              # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(HERE, "examples", "check_injection_path.py")

sys.path.insert(0, HERE)
from kappa_c_audit.checks import injection_path_check           # noqa: E402


# --------------------------------------------------------------------------- fixtures
def _write_h5ad(path, data_values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as h:
        g = h.create_group("layers")
        wg = g.create_group("geneWeight")
        arr = np.array(data_values, dtype=np.float64)
        wg.create_dataset("data", data=arr)
        if len(arr):
            wg.create_dataset("indptr", data=np.array([0, len(arr)]))
            wg.create_dataset("indices", data=np.arange(len(arr)))


@pytest.fixture()
def delivering(tmp_path):
    """A pipeline whose weight vector is populated AND whose branch is instantiated."""
    art = tmp_path / "artifacts"
    src = tmp_path / "source"
    (src / "training").mkdir(parents=True)
    _write_h5ad(str(art / "benchmark" / "runA" / "iter_0" / "stagedata" / "cell.h5ad"),
                [0.7, 0.2, 0.9])
    (src / "training" / "model.py").write_text(
        "import torch\n"
        "class Model:\n"
        "    def __init__(self, geneWeight=None):\n"
        "        self.geneWeight = geneWeight\n"
        "    def loss(self):\n"
        "        return GVAELoss(geneWeight=self.geneWeight)\n", encoding="utf-8")
    return str(art), str(src)


@pytest.fixture()
def blocking(tmp_path):
    """A pipeline whose layer is present but EMPTY and whose branch is never built."""
    art = tmp_path / "artifacts"
    src = tmp_path / "source"
    (src / "nofollow").mkdir(parents=True)
    _write_h5ad(str(art / "result" / "runB" / "iter_0" / "stagedata" / "cell.h5ad"), [])
    (src / "nofollow" / "model.py").write_text(
        "class Model:\n"
        "    def __init__(self, geneWeight=None):\n"
        "        self.geneWeight = geneWeight\n", encoding="utf-8")
    return str(art), str(src)


# --------------------------------------------------------------------- the function
def test_delivering_pipeline_is_not_blocked(delivering):
    art, src = delivering
    res = injection_path_check(art, src, pattern="**/stagedata/*.h5ad",
                              roots=["benchmark"], layer_key="layers/geneWeight/data")
    assert res["weight_layer_totals"]["populated"] == 1
    assert res["branch"]["n_instantiations"] == 1
    assert res["verdict"]["delivery"] != "BLOCKED"


def test_blocking_pipeline_is_blocked(blocking):
    art, src = blocking
    res = injection_path_check(art, src, pattern="**/stagedata/*.h5ad",
                              roots=["result"])
    assert res["weight_layer_totals"]["populated"] == 0
    assert res["weight_layer_totals"]["empty"] == 1
    assert res["branch"]["verdict"].startswith("DEAD")
    assert res["verdict"]["delivery"] == "BLOCKED"


def test_roots_are_counted_separately(blocking, delivering):
    """A subset count must never be silently merged with a full-tree count."""
    art, src = delivering
    _write_h5ad(str(os.path.join(art, "result", "runB", "iter_0", "stagedata",
                                 "cell.h5ad")), [])
    res = injection_path_check(art, src, pattern="**/stagedata/*.h5ad",
                              roots=["benchmark", "result"])
    assert res["per_root"]["benchmark"]["populated_layer"] == 1
    assert res["per_root"]["result"]["populated_layer"] == 0
    assert res["weight_layer_totals"]["populated"] == 1


def test_weight_name_is_parameterised(delivering):
    """Grepping for a name that does not exist must yield DEAD, not a crash."""
    art, src = delivering
    res = injection_path_check(art, src, pattern="**/stagedata/*.h5ad",
                              roots=["benchmark"], weight_name="priorVec")
    assert res["readers"]["n_assignment_sites"] == 0
    assert res["branch"]["verdict"].startswith("DEAD")
    # D1 is unaffected: it reads bytes, not source text
    assert res["weight_layer_totals"]["populated"] == 1


def test_layer_key_is_parameterised(delivering):
    """A wrong layer key must report no layer, not a false pass."""
    art, src = delivering
    res = injection_path_check(art, src, pattern="**/stagedata/*.h5ad",
                              roots=["benchmark"], layer_key="layers/nonexistent/data")
    assert res["weight_layer_totals"]["with_layer"] == 0
    assert res["weight_layer_verdict"] == "undetermined"


# --------------------------------------------------------------------------- the CLI
def _cli(args):
    return subprocess.run([sys.executable, CLI] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=HERE)


def test_cli_exit_codes_differ_by_outcome(delivering, blocking):
    art, src = delivering
    good = _cli(["--artifacts", art, "--source", src, "--roots", "benchmark"])
    art2, src2 = blocking
    bad = _cli(["--artifacts", art2, "--source", src2, "--roots", "result"])
    assert good.returncode == 0, good.stdout + good.stderr
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "not blocked" in good.stdout
    assert "BLOCKED" in bad.stdout


def test_cli_writes_json(delivering, tmp_path):
    art, src = delivering
    dest = tmp_path / "out.json"
    r = _cli(["--artifacts", art, "--source", src, "--roots", "benchmark",
              "--json", str(dest)])
    assert r.returncode == 0
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["check"] == "injection-path"
    assert "per_root" in doc and "verdict" in doc


def test_cli_rejects_bad_paths(tmp_path):
    r = _cli(["--artifacts", str(tmp_path / "nope"), "--source", str(tmp_path)])
    assert r.returncode == 2
    assert "not a directory" in (r.stdout + r.stderr)
