# -*- coding: utf-8 -*-
"""Apply the COMPUTED vanilla-equivalence band to Star Methods.

Computation of record: outputs/positive_result/s1k_gvae_band.json -- both statistics evaluated
on the SAME 15 G-VAE hematopoietic vanilla seeds, 210 ordered seed-pair comparisons:

    trace form      tr(W^T W)      mean 1.0062 +/- 0.1123  -> +/-2s = [0.781, 1.231]
    corrected form  sum_g R^2_g    mean 1.0005 +/- 0.0319  -> +/-2s = [0.9367, 1.0644]

The manuscript's Results/captions/tables all use the corrected band [0.937, 1.064] (reported as
1.001 +/- 0.032). Star Methods still carried the superseded trace-form value [0.78, 1.23]
(1.01 +/- 0.11), so the two halves of the paper disagreed. This replaces the Methods numbers
with the corrected ones. Nothing but the numerals changes.

C = (1 - kappa) * 100%  ->  1.0005 gives -0.05%, and sd 0.0319 gives 3.2 percentage points.
"""
import io

P = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/star_methods.tex"
raw = io.open(P, "rb").read()
s = raw.decode("utf-8").replace("\r\n", "\n")
orig = s

REPL = [
    (r"$\kappa \in [0.78, 1.23]$", r"$\kappa \in [0.937, 1.064]$",
     "L64 region definition"),
    (r"the null is $\kappa=1.01\pm0.11$ (mean $\pm$ SD; 210 ordered seed-pair comparisons), "
     r"i.e.\ $C=-0.6\%\pm11.2$ percentage points",
     r"the null is $\kappa=1.001\pm0.032$ (mean $\pm$ SD; 210 ordered seed-pair comparisons), "
     r"i.e.\ $C=-0.05\%\pm3.2$ percentage points",
     "L64 null mean/sd + C"),
    (r"yields $\kappa=1.01\pm0.11$ and $C=-0.6\%\pm11.2$ percentage points",
     r"yields $\kappa=1.001\pm0.032$ and $C=-0.05\%\pm3.2$ percentage points",
     "L143 null mean/sd + C"),
    (r"region $\kappa\in[0.78,1.23]$ ($\pm2\sigma$)",
     r"region $\kappa\in[0.937,1.064]$ ($\pm2\sigma$)",
     "L143 region definition"),
    (r"$\kappa>1.23$ confirms", r"$\kappa>1.064$ confirms", "L143 expansion threshold"),
    (r"while $\kappa<0.78$ indicates", r"while $\kappa<0.937$ indicates", "L143 compression threshold"),
    (r"the interval $[0.78,1.23]$ was measured", r"the interval $[0.937,1.064]$ was measured",
     "L143 provenance sentence"),
    (r"the cross-family $[0.78,1.23]$ interval", r"the cross-family $[0.937,1.064]$ interval",
     "L143 caveat sentence"),
]

print("%-34s %s" % ("site", "result"))
for old, new, label in REPL:
    n = s.count(old)
    if n == 0:
        print("%-34s NOT FOUND" % label)
        continue
    s = s.replace(old, new)
    print("%-34s %d replacement(s)" % (label, n))

assert s != orig, "nothing changed"
io.open(P, "wb").write(s.replace("\n", "\r\n").encode("utf-8"))
chk = io.open(P, "rb").read()
print()
print("CRLF=%d lone_lf=%d" % (chk.count(b"\r\n"), chk.count(b"\n") - chk.count(b"\r\n")))
left = chk.count(b"0.78") + chk.count(b"1.23")
print("residual 0.78/1.23 in star_methods.tex:", left)
