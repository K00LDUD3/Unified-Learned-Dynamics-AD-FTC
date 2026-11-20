"""Early-detection benchmark orchestrator with ledger batching and echo logging."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import exprim.anchor as anchor
import exprim.ledger as ledger
from exprim.echo.logger import Logger
from exprim.echo.sinks import ConsoleSink, FileSink
from exprim.echo.event import Level

from early_detect_config import ExperimentConfig, validate_or_raise
from plant import simulate
from detectors import build_detectors, first_crossing
from ssm import SSM, train_ssm


def next_batch_dir(runs_root: Path) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in runs_root.iterdir():
        if p.is_dir() and p.name.startswith("batch_"):
            try:
                nums.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    n = (max(nums) + 1) if nums else 1
    d = runs_root / f"batch_{n:03d}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def artifact_dir(run) -> Path:
    d = ledger.paths.artifacts_dir(run.run_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_logger(run) -> Logger:
    sinks = [
        ConsoleSink(min_level=Level.INFO),
        FileSink(path=run.logs_path(), min_level=Level.DEBUG),
        FileSink(path=run.metrics_path(), min_level=Level.INFO),
    ]
    logger = Logger(run_id=run.run_id, module="benchmark", sinks=sinks, min_level=Level.DEBUG)
    run.attach_logger(logger)
    return logger


def slice_from(tel: dict, idx: int) -> dict:
    n = len(tel["t"])
    out = {}
    for k, v in tel.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[idx:]
        else:
            out[k] = v
    return out


def save_run_arrays(run, tel: dict, signals: dict) -> None:
    ad = artifact_dir(run)
    np.savez(
        ad / "telemetry.npz",
        t=tel["t"], x=tel["x"], x_ref=tel["x_ref"],
        v_cmd=tel["v_cmd"], v_act=tel["v_act"],
        margin=tel["margin"], alpha_traj=tel["alpha_traj"],
    )
    np.save(ad / "latent_signals.npy", signals)
    keys = sorted(signals.keys())
    header = "t," + ",".join(keys)
    data = np.column_stack([tel["t"]] + [signals[k] for k in keys])
    np.savetxt(ad / "detection_signals.csv", data, delimiter=",", header=header, comments="")


def plot_run(run, tel: dict, signals: dict, thresholds: dict, t_fault: float) -> None:
    ad = artifact_dir(run)
    fig, axs = plt.subplots(3, 1, figsize=(11, 12))

    axs[0].plot(tel["x_ref"][:, 0], tel["x_ref"][:, 1], "k--", alpha=0.5, label="reference")
    axs[0].plot(tel["x"][:, 0], tel["x"][:, 1], "b-", alpha=0.7, label="state")
    axs[0].set_title("Phase portrait")
    axs[0].set_aspect("equal")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(tel["t"], tel["margin"], "r-", label="authority margin")
    axs[1].axhline(1.0, color="k", linestyle=":", label="AMS boundary (margin=1)")
    axs[1].axvline(t_fault, color="gray", linestyle="--", label="fault onset")
    axs[1].set_title("Authority margin over time")
    axs[1].set_xlabel("t")
    axs[1].legend()
    axs[1].grid(True)

    for k in sorted(signals.keys()):
        thr = thresholds.get(k, np.nan)
        axs[2].plot(tel["t"], signals[k], label=k)
        if np.isfinite(thr):
            axs[2].axhline(thr, linestyle=":")
    axs[2].axvline(t_fault, color="gray", linestyle="--")
    axs[2].set_title("Detector signals (dotted = threshold)")
    axs[2].set_xlabel("t")
    axs[2].legend()
    axs[2].grid(True)

    fig.tight_layout()
    fig.savefig(ad / "overview.png", dpi=150)
    plt.close(fig)


def evaluate_run(cfg, tel, detectors, device, logger, idx_settle: int):
    t = tel["t"]
    t_fault = tel["t_fault"]
    persist = cfg.detection.min_persist
    bidx = np.where(tel["margin"][idx_settle:] >= 1.0)[0]
    boundary_time = float(t[idx_settle + bidx[0]]) if bidx.size else None

    rows = []
    signals = {}
    thresholds = {}
    ctx = {"warmup": idx_settle}
    for det in detectors:
        sig = det.score(tel, ctx)
        signals[det.name] = sig
        thresholds[det.name] = det.thr
        idx = first_crossing(sig, det.thr, warmup=idx_settle, persist=persist)
        cross_time = float(t[idx]) if idx is not None else None
        fired_after_fault = (cross_time is not None) and (cross_time >= t_fault - 1e-9)
        latency = (cross_time - t_fault) if fired_after_fault else None
        false_alarm = (cross_time is not None) and (cross_time < t_fault - 1e-9)
        margin_at_cross = float(tel["margin"][idx]) if idx is not None else None
        detected_before_boundary = fired_after_fault and (
            boundary_time is None or cross_time < boundary_time
        )
        rows.append({
            "detector": det.name,
            "cross_time": cross_time,
            "fired_after_fault": bool(fired_after_fault),
            "latency": latency,
            "false_alarm": bool(false_alarm),
            "margin_at_cross": margin_at_cross,
            "boundary_time": boundary_time,
            "detected_before_boundary": bool(detected_before_boundary),
        })
        if logger is not None:
            tag = (det.name,)
            if latency is not None:
                logger.metric(f"detect/{det.name}/latency", latency, tags=tag)
            if margin_at_cross is not None:
                logger.metric(f"detect/{det.name}/margin_at_cross", margin_at_cross, tags=tag)
            logger.metric(f"detect/{det.name}/fired_after_fault",
                          1.0 if fired_after_fault else 0.0, tags=tag)
            logger.metric(f"detect/{det.name}/false_alarm",
                          1.0 if false_alarm else 0.0, tags=tag)
    return rows, signals, thresholds, boundary_time


def main() -> None:
    cfg = ExperimentConfig()
    validate_or_raise(cfg)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(
        cfg.model.device if (cfg.model.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    runs_root = anchor.paths.from_root(cfg.output_dir)
    batch_dir = next_batch_dir(runs_root)
    results: list[dict] = []

    with ledger.ExperimentManager(runs_dir=batch_dir) as manager:
        train_tel = simulate(cfg, alpha_scalar=None)
        idx_settle_train = int(np.searchsorted(train_tel["t"], cfg.detection.settle_time))

        run_train = manager.create_run(cfg, name="0000_train_ssm",
                                       tags=["train", "ssm"], auto_snapshot=False)
        log_train = make_logger(run_train)
        log_train.info(f"batch={batch_dir.name} device={device} R={train_tel['R']:.4f}")

        model = SSM(state_dim=2, input_dim=2,
                    latent_dim=cfg.model.latent_dim,
                    hidden=cfg.model.hidden_dim,
                    n_hidden=cfg.model.n_hidden).to(device)
        model = train_ssm(model, train_tel["x"], train_tel["v_cmd"], cfg, device, log_train)

        ad_train = artifact_dir(run_train)
        torch.save(model.state_dict(), ad_train / "ssm.pt")
        np.savez(ad_train / "train_telemetry.npz",
                 t=train_tel["t"], x=train_tel["x"], v_cmd=train_tel["v_cmd"])
        run_train.mark_complete()
        log_train.info("SSM training complete")
        log_train.close()

        detectors = build_detectors(cfg, model=model, device=device)
        cal_tel = slice_from(train_tel, idx_settle_train)
        ctx = {"warmup": 0}
        for det in detectors:
            sig_cal = det.fit_signal(cal_tel, ctx)
            det.fit(sig_cal)

        for i, alpha in enumerate(cfg.actuators.alpha_sweep, start=1):
            label = f"{i:04d}_alpha_{alpha:.2f}"
            run = manager.create_run(cfg.replace(seed=cfg.seed + i), name=label,
                                     tags=["eval", f"alpha_{alpha:.2f}"], auto_snapshot=False)
            log = make_logger(run)
            
            alpha_arg = None if alpha == 1.0 else alpha
            tel = simulate(cfg, alpha_scalar=alpha_arg)
            
            idx_settle = int(np.searchsorted(tel["t"], cfg.detection.settle_time))
            rows, signals, thresholds, boundary_time = evaluate_run(
                cfg, tel, detectors, device, log, idx_settle)

            for r in rows:
                r["alpha"] = alpha
                r["run"] = label
                results.append(r)

            save_run_arrays(run, tel, signals)
            plot_run(run, tel, signals, thresholds, tel["t_fault"])

            ad = artifact_dir(run)
            with open(ad / "run_results.json", "w") as f:
                json.dump(rows, f, indent=2)

            log.info(f"alpha={alpha:.2f} boundary_time={boundary_time}")
            for r in rows:
                log.info(
                    f"  {r['detector']:15s} fired_after_fault={r['fired_after_fault']} "
                    f"latency={r['latency']} margin_at_cross={r['margin_at_cross']} "
                    f"before_boundary={r['detected_before_boundary']}"
                )
            run.mark_complete()
            log.close()

    with open(batch_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if results:
        cols = ["run", "alpha", "detector", "cross_time", "fired_after_fault",
                "latency", "false_alarm", "margin_at_cross", "boundary_time",
                "detected_before_boundary"]
        with open(batch_dir / "results.csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for r in results:
                f.write(",".join(str(r.get(c)) for c in cols) + "\n")

    summary = {}
    for det in cfg.detection.detectors:
        sub = [r for r in results if r["detector"] == det]
        degraded = [r for r in sub if r["alpha"] < 1.0]
        healthy = [r for r in sub if r["alpha"] >= 1.0]
        lat = [r["latency"] for r in degraded if r["latency"] is not None]
        summary[det] = {
            "n_degraded": len(degraded),
            "detected": sum(1 for r in degraded if r["fired_after_fault"]),
            "mean_latency": float(np.mean(lat)) if lat else None,
            "false_alarms_healthy": sum(
                1 for r in healthy if r["fired_after_fault"] or r["false_alarm"]),
            "detected_before_boundary": sum(1 for r in degraded if r["detected_before_boundary"]),
        }

    with open(batch_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PER-DETECTOR SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== FULL RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nbatch dir: {batch_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
