"""Regenerate a per-detector overview figure with explicit detection markers.
Usage: python replot_run_overview.py [run_prefix]   (default 0005_alpha_0.40)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import exprim.anchor as anchor

prefix = sys.argv[1] if len(sys.argv) > 1 else "0005_alpha_0.40"
batch = sorted([d for d in Path("runs").iterdir()
                if d.is_dir() and d.name.startswith("batch_")])[-1]


def find_run_dir(batch_dir: Path, pref: str) -> Path:
    for cand in sorted(batch_dir.glob(pref + "*")):
        if not cand.is_dir():
            continue
        if (cand / "artifacts").is_dir():
            return cand
        for sub in sorted(cand.iterdir()):
            if sub.is_dir() and (sub / "artifacts").is_dir():
                return sub
    raise FileNotFoundError(f"no run for prefix {pref!r} in {batch_dir}")


run_dir = find_run_dir(batch, prefix)
healthy_dir = find_run_dir(batch, "0001_alpha_1.00")
cfgd = anchor.load_yaml(run_dir / "config" / "config.yaml")
det = cfgd["detection"]
settle, tsig = det["settle_time"], det["threshold_sigma"]
lam, L = det["ewma_lambda"], det["ewma_L"]
t_fault = cfgd["fault"]["t_fault"]

sig = pd.read_csv(run_dir / "artifacts" / "detection_signals.csv")
res = json.loads((run_dir / "artifacts" / "run_results.json").read_text())
tel = np.load(run_dir / "artifacts" / "telemetry.npz")
hsig = pd.read_csv(healthy_dir / "artifacts" / "detection_signals.csv")
h = hsig[hsig["t"] >= settle]

base_mu, base_sd = h["model_residual"].mean(), h["model_residual"].std()
thr = {
    "tracking_error": h["tracking_error"].mean() + tsig * h["tracking_error"].std(),
    "model_residual": base_mu + tsig * base_sd,
    "cusum": tsig * base_sd,
    "ewma": base_mu + L * base_sd * np.sqrt(lam / (2.0 - lam)),
    "ssm": h["ssm"].mean() + tsig * h["ssm"].std(),
}
cross = {r["detector"]: r["cross_time"] for r in res}
names = ["tracking_error", "model_residual", "cusum", "ewma", "ssm"]

fig, axs = plt.subplots(len(names) + 1, 1, figsize=(11, 3 * (len(names) + 1)), sharex=True)
axs[0].plot(tel["t"], tel["margin"], "k-", label="authority margin")
axs[0].axhline(1.0, color="r", ls="--", label="AMS boundary")
axs[0].axvline(t_fault, color="gray", ls=":", label="fault onset")
axs[0].set_ylabel("margin")
axs[0].legend(fontsize=8)
axs[0].grid(True)

for ax, name in zip(axs[1:], names):
    ax.plot(sig["t"], sig[name], label=name)
    ax.axhline(thr[name], color="k", ls=":", label=f"threshold={thr[name]:.3g}")
    ax.axvline(t_fault, color="gray", ls=":")
    ct = cross.get(name)
    if ct is not None:
        ax.axvline(ct, color="r", ls="-", label=f"detect t={ct:.2f}s")
    ax.set_ylabel(name)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)

axs[-1].set_xlabel("t (s)")
fig.suptitle(f"Per-detector signals and detection times ({run_dir.name})")
fig.tight_layout()
out = run_dir / "artifacts" / "overview_corrected.png"
fig.savefig(out, dpi=150)
plt.close(fig)

print(f"saved: {out}")
for name in names:
    print(f"  {name:15s} threshold={thr[name]:.3g}  cross_time={cross.get(name)}")
