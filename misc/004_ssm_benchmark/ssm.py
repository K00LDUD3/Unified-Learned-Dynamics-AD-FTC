"""Learned nonlinear state-space model (autoencoder + latent transition)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _mlp(in_dim: int, out_dim: int, hidden: int, n_hidden: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class SSM(nn.Module):
    def __init__(self, state_dim: int = 2, input_dim: int = 2,
                 latent_dim: int = 2, hidden: int = 64, n_hidden: int = 2):
        super().__init__()
        self.enc = _mlp(state_dim, latent_dim, hidden, n_hidden)
        self.dec = _mlp(latent_dim, state_dim, hidden, n_hidden)
        self.dyn = _mlp(latent_dim + input_dim, latent_dim, hidden, n_hidden)

    def set_stats(self, X: np.ndarray, U: np.ndarray, device) -> None:
        self.register_buffer("x_mean", torch.tensor(X.mean(0), dtype=torch.float32).to(device))
        self.register_buffer("x_std", torch.tensor(X.std(0) + 1e-6, dtype=torch.float32).to(device))
        self.register_buffer("u_mean", torch.tensor(U.mean(0), dtype=torch.float32).to(device))
        self.register_buffer("u_std", torch.tensor(U.std(0) + 1e-6, dtype=torch.float32).to(device))

    def _nx(self, x):
        return (x - self.x_mean) / self.x_std

    def _nu(self, u):
        return (u - self.u_mean) / self.u_std

    def encode_raw(self, x):
        return self.enc(self._nx(x))

    def decode_raw(self, z):
        return self.dec(z) * self.x_std + self.x_mean

    def step_raw(self, z, u):
        return self.dyn(torch.cat([z, self._nu(u)], dim=-1))


def train_ssm(model: SSM, X: np.ndarray, U: np.ndarray, cfg, device, logger=None) -> SSM:
    model.set_stats(X, U, device)
    Xt = torch.tensor(X[:-1], dtype=torch.float32).to(device)
    Ut = torch.tensor(U[:-1], dtype=torch.float32).to(device)
    Xn = torch.tensor(X[1:], dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    mse = nn.MSELoss()
    ds = TensorDataset(Xt, Ut, Xn)
    dl = DataLoader(ds, batch_size=cfg.model.batch_size, shuffle=True)

    for epoch in range(cfg.model.epochs):
        tot = 0.0
        for bx, bu, bxn in dl:
            opt.zero_grad()
            z = model.encode_raw(bx)
            zn = model.encode_raw(bxn)
            zpred = model.step_raw(z, bu)
            xrec = model.decode_raw(z)
            loss = mse(xrec, bx) + mse(zpred, zn)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bx)
        avg = tot / len(ds)
        if logger is not None and (epoch % 10 == 0 or epoch == cfg.model.epochs - 1):
            logger.metric("ssm/train_loss", avg, step=epoch)
    return model


def predict_error(model: SSM, X: np.ndarray, U: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        Ut = torch.tensor(U, dtype=torch.float32).to(device)
        z = model.encode_raw(Xt)
        zpred = model.step_raw(z[:-1], Ut[:-1])
        znext = z[1:]
        err = torch.norm(zpred - znext, dim=1).cpu().numpy()
    if err.size == 0:
        return np.zeros(len(X))
    return np.concatenate([[err[0]], err])
