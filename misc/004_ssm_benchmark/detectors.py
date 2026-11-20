"""Early-detection baseline detectors. Causal, observable-telemetry only."""
from __future__ import annotations

import numpy as np


def rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty_like(y)
    for i in range(len(y)):
        a = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[a]) / (i - a + 1)
    return out


def causal_derivative(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    d = np.empty_like(X)
    dt = t[1:] - t[:-1]
    d[1:] = (X[1:] - X[:-1]) / dt[:, None]
    d[0] = d[1]
    return d


def plant_drift_vec(X: np.ndarray, mu: float, omega: float) -> np.ndarray:
    r2 = (X ** 2).sum(axis=1)
    return np.stack([
        mu * X[:, 0] - omega * X[:, 1] - X[:, 0] * r2,
        omega * X[:, 0] + mu * X[:, 1] - X[:, 1] * r2,
    ], axis=1)


def model_residual_signal(tel: dict, cfg) -> np.ndarray:
    dx = causal_derivative(tel["x"], tel["t"])
    implied = dx - plant_drift_vec(tel["x"], cfg.plant.mu, cfg.plant.omega)
    r = implied - tel["v_cmd"]
    return np.linalg.norm(r, axis=1)


def first_crossing(signal: np.ndarray, thr: float, warmup: int = 0,
                   persist: int = 1) -> int | None:
    n = len(signal)
    i = warmup
    while i <= n - persist:
        if np.all(signal[i:i + persist] > thr):
            return i
        i += 1
    return None


class _BaseDetector:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.thr = float("inf")

    def fit_signal(self, tel: dict, ctx: dict) -> np.ndarray:
        return self.score(tel, ctx)

    def fit(self, healthy_signal: np.ndarray) -> None:
        mu = float(np.mean(healthy_signal))
        sd = float(np.std(healthy_signal)) + 1e-12
        self.mu = mu
        self.sd = sd
        self.thr = mu + self.cfg.detection.threshold_sigma * sd

    def score(self, tel: dict, ctx: dict) -> np.ndarray:
        raise NotImplementedError


class TrackingErrorDetector(_BaseDetector):
    name = "tracking_error"

    def score(self, tel, ctx):
        e = np.linalg.norm(tel["x"] - tel["x_ref"], axis=1)
        return rolling_mean(e, self.cfg.detection.window)


class ModelResidualDetector(_BaseDetector):
    name = "model_residual"

    def score(self, tel, ctx):
        return model_residual_signal(tel, self.cfg)


class CUSUMDetector(_BaseDetector):
    name = "cusum"

    def fit_signal(self, tel, ctx):
        return model_residual_signal(tel, self.cfg)

    def fit(self, healthy_signal):
        self.mu = float(np.mean(healthy_signal))
        self.sd = float(np.std(healthy_signal)) + 1e-12
        self.k = self.cfg.detection.cusum_drift_k * self.sd
        self.thr = self.cfg.detection.threshold_sigma * self.sd

    def score(self, tel, ctx):
        y = model_residual_signal(tel, self.cfg)
        warmup = ctx.get("warmup", 0)
        s = 0.0
        out = np.zeros_like(y)
        for i in range(warmup, len(y)):
            s = max(0.0, s + (y[i] - self.mu) - self.k)
            out[i] = s
        return out


class EWMADetector(_BaseDetector):
    name = "ewma"

    def fit_signal(self, tel, ctx):
        return model_residual_signal(tel, self.cfg)

    def fit(self, healthy_signal):
        self.mu = float(np.mean(healthy_signal))
        self.sd = float(np.std(healthy_signal)) + 1e-12
        lam = self.cfg.detection.ewma_lambda
        self.thr = self.mu + self.cfg.detection.ewma_L * self.sd * np.sqrt(lam / (2.0 - lam))

    def score(self, tel, ctx):
        y = model_residual_signal(tel, self.cfg)
        warmup = ctx.get("warmup", 0)
        lam = self.cfg.detection.ewma_lambda
        z = np.zeros_like(y)
        if warmup < len(y):
            z[warmup] = y[warmup]
            for i in range(warmup + 1, len(y)):
                z[i] = lam * y[i] + (1.0 - lam) * z[i - 1]
        return z


class SSMDetector(_BaseDetector):
    name = "ssm"

    def __init__(self, cfg, model, device):
        super().__init__(cfg)
        self.model = model
        self.device = device

    def score(self, tel, ctx):
        from ssm import predict_error
        return predict_error(self.model, tel["x"], tel["v_cmd"], self.device)


def build_detectors(cfg, model=None, device=None) -> list:
    out = []
    names = cfg.detection.detectors
    if "tracking_error" in names:
        out.append(TrackingErrorDetector(cfg))
    if "model_residual" in names:
        out.append(ModelResidualDetector(cfg))
    if "cusum" in names:
        out.append(CUSUMDetector(cfg))
    if "ewma" in names:
        out.append(EWMADetector(cfg))
    if "ssm" in names and model is not None:
        out.append(SSMDetector(cfg, model, device))
    return out
