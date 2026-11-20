"""Hopf normal form plant with partially redundant actuators.
Uses a single continuous ODE integration with time-varying alpha(t) to avoid
solver-restart discontinuities that corrupt derivative-based residuals."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def build_effectiveness_matrix(angles_deg: list[float]) -> np.ndarray:
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.vstack([np.cos(th), np.sin(th)])


def ams_support(B: np.ndarray, alpha: np.ndarray, u_max: float, u_hat: np.ndarray) -> float:
    cols = B * np.asarray(alpha).reshape(1, -1)
    return float(u_max * np.sum(np.abs(u_hat @ cols)))


def authority_margin(v: np.ndarray, B: np.ndarray, alpha: np.ndarray, u_max: float) -> float:
    vnorm = float(np.linalg.norm(v))
    if vnorm < 1e-12:
        return 0.0
    u_hat = v / vnorm
    h = ams_support(B, alpha, u_max, u_hat)
    if h < 1e-12:
        return float("inf")
    return vnorm / h


def reference(t: float, R: float, W: float) -> np.ndarray:
    return np.array([R * np.cos(W * t), R * np.sin(W * t)])


def solve_reference_radius(cfg) -> float:
    if cfg.reference.radius_override is not None:
        return float(cfg.reference.radius_override)
    B = build_effectiveness_matrix(cfg.actuators.angles_deg)
    alpha_h = np.ones(B.shape[1])
    thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radii = []
    for th in thetas:
        u = np.array([np.cos(th), np.sin(th)])
        radii.append(ams_support(B, alpha_h, cfg.actuators.u_max, u))
    r_worst = float(min(radii))
    target = cfg.reference.demand_fraction_of_healthy_ams * r_worst
    mu = cfg.plant.mu
    best_R, best_err = 1.0, float("inf")
    for R in np.linspace(0.05, 5.0, 5000):
        d = abs((R * R - mu) * R)
        err = abs(d - target)
        if err < best_err:
            best_err, best_R = err, R
    return float(best_R)


def simulate(cfg, alpha_scalar: float | None) -> dict:
    B = build_effectiveness_matrix(cfg.actuators.angles_deg)
    n_act = B.shape[1]
    alpha_h = np.ones(n_act)
    
    if alpha_scalar is None:
        alpha_deg = alpha_h.copy()
        t_fault = cfg.plant.t_end + 1.0
    else:
        alpha_deg = alpha_h.copy()
        alpha_deg[cfg.actuators.degrade_channel] = float(alpha_scalar)
        t_fault = cfg.fault.t_fault
        
    ramp = cfg.fault.ramp_time

    def alpha_of_t(t: float) -> np.ndarray:
        if t < t_fault:
            return alpha_h
        if ramp <= 0.0:
            return alpha_deg
        s = min(1.0, (t - t_fault) / ramp)
        return alpha_h + (alpha_deg - alpha_h) * s

    R = solve_reference_radius(cfg)
    W = cfg.reference.freq
    mu, omega = cfg.plant.mu, cfg.plant.omega
    u_max = cfg.actuators.u_max
    kp = cfg.controller.kp
    B_nom_pinv = np.linalg.pinv(B)

    def rhs(t, x):
        alpha_t = alpha_of_t(t)
        x_r = reference(t, R, W)
        dref = np.array([-R * W * np.sin(W * t), R * W * np.cos(W * t)])
        fx = mu * x_r[0] - omega * x_r[1] - x_r[0] * (R * R)
        fy = omega * x_r[0] + mu * x_r[1] - x_r[1] * (R * R)
        v_ff = dref - np.array([fx, fy])
        v_cmd = v_ff - kp * (x - x_r)
        
        u_cmd = np.clip(B_nom_pinv @ v_cmd, -u_max, u_max)
        v_act = (B * alpha_t.reshape(1, -1)) @ u_cmd
        
        r2 = x[0]**2 + x[1]**2
        drift = np.array([
            mu * x[0] - omega * x[1] - x[0] * r2,
            omega * x[0] + mu * x[1] - x[1] * r2,
        ])
        return drift + v_act

    x0 = np.asarray(cfg.plant.x0, dtype=float)
    t_eval = np.arange(0.0, cfg.plant.t_end, cfg.plant.dt)
    
    sol = solve_ivp(rhs, (0.0, cfg.plant.t_end), x0, t_eval=t_eval, 
                    method="RK45", rtol=1e-6, atol=1e-8)
    t, X = sol.t, sol.y.T

    T = len(t)
    V_cmd = np.zeros((T, 2))
    V_act = np.zeros((T, 2))
    X_ref = np.zeros((T, 2))
    margin = np.zeros(T)
    alpha_traj = np.zeros((T, n_act))
    
    for i in range(T):
        alpha_i = alpha_of_t(t[i])
        x_r = reference(t[i], R, W)
        dref = np.array([-R * W * np.sin(W * t[i]), R * W * np.cos(W * t[i])])
        fx = mu * x_r[0] - omega * x_r[1] - x_r[0] * (R * R)
        fy = omega * x_r[0] + mu * x_r[1] - x_r[1] * (R * R)
        v_ff = dref - np.array([fx, fy])
        v_cmd = v_ff - kp * (X[i] - x_r)
        
        u_cmd = np.clip(B_nom_pinv @ v_cmd, -u_max, u_max)
        v_act = (B * alpha_i.reshape(1, -1)) @ u_cmd
        
        V_cmd[i] = v_cmd
        V_act[i] = v_act
        X_ref[i] = x_r
        margin[i] = authority_margin(v_cmd, B, alpha_i, u_max)
        alpha_traj[i] = alpha_i

    return {
        "t": t, "x": X, "x_ref": X_ref, "v_cmd": V_cmd, "v_act": V_act,
        "margin": margin, "alpha_traj": alpha_traj,
        "t_fault": t_fault, "R": R,
        "alpha_healthy": alpha_h, "alpha_degraded": alpha_deg,
    }
