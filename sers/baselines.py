"""Final-run fixed-background MCR-ALS and prior neural comparator implementations."""
from __future__ import annotations
import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import lsq_linear

def normalize_level(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span

def fit_nnls(task: dict[str, Any]) -> dict[str, np.ndarray | float]:
    x_mix = np.asarray(task["x_mix"], dtype=float)
    concentrations = normalize_level(np.asarray(task["concentrations"], dtype=float))
    backgrounds = np.vstack([np.asarray(background, dtype=float) for background in task["bg"]])
    coeffs = []
    bg_recon = []
    for row in x_mix:
        fit = lsq_linear(backgrounds.T, row, bounds=(0.0, np.inf), lsmr_tol="auto", max_iter=200)
        coeff = np.asarray(fit.x, dtype=float)
        coeffs.append(coeff)
        bg_recon.append(coeff @ backgrounds)
    coeffs_array = np.vstack(coeffs)
    residuals = x_mix - np.vstack(bg_recon)
    denom = float(np.dot(concentrations, concentrations))
    curve = residuals.mean(axis=0) if denom <= 1e-12 else (concentrations[:, None] * residuals).sum(axis=0) / denom
    analyte_coeff = concentrations.copy()
    reconstruction = np.vstack(bg_recon) + analyte_coeff[:, None] * curve[None, :]
    return {
        "curve": curve,
        "background_coefficients": coeffs_array,
        "analyte_coefficients": analyte_coeff,
        "reconstruction": reconstruction,
        "residual_norm_ratio": float(np.linalg.norm(x_mix - reconstruction) / max(np.linalg.norm(x_mix), 1e-12)),
    }

def solve_fixed_coefficients(x_mix: np.ndarray, curve: np.ndarray, backgrounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = np.vstack([curve[None, :], backgrounds])
    coeffs = []
    recon = []
    for row in x_mix:
        fit = lsq_linear(basis.T, row, bounds=(0.0, np.inf), lsmr_tol="auto", max_iter=200)
        coeff = np.asarray(fit.x, dtype=float)
        coeffs.append(coeff)
        recon.append(coeff @ basis)
    return np.vstack(coeffs), np.vstack(recon)

def fit_fixed_bg_mcr(task: dict[str, Any], max_iter: int = 250, tol: float = 1e-7) -> dict[str, Any]:
    x_mix = np.asarray(task["x_mix"], dtype=float)
    backgrounds = np.vstack([np.asarray(background, dtype=float) for background in task["bg"]])
    initial = fit_nnls(task)
    curve = np.clip(np.asarray(initial["curve"], dtype=float), 0.0, None)
    if float(np.linalg.norm(curve)) <= 1e-12:
        curve = np.clip(np.mean(x_mix, axis=0), 0.0, None)
    if float(np.max(curve)) > 1e-12:
        curve = curve / float(np.max(curve))

    losses: list[float] = []
    converged = False
    coeffs = np.zeros((x_mix.shape[0], backgrounds.shape[0] + 1), dtype=float)
    recon = np.zeros_like(x_mix)
    for iteration in range(int(max_iter)):
        coeffs, recon = solve_fixed_coefficients(x_mix, curve, backgrounds)
        losses.append(float(np.mean((x_mix - recon) ** 2)))
        analyte_coeff = coeffs[:, 0]
        residual_after_bg = x_mix - coeffs[:, 1:] @ backgrounds
        denom = float(np.dot(analyte_coeff, analyte_coeff))
        if denom > 1e-12:
            updated = np.clip((analyte_coeff[:, None] * residual_after_bg).sum(axis=0) / denom, 0.0, None)
            max_value = float(np.max(updated)) if updated.size else 0.0
            if max_value > 1e-12:
                updated = updated / max_value
            else:
                updated = curve.copy()
        else:
            updated = curve.copy()
        delta = float(np.linalg.norm(updated - curve) / max(np.linalg.norm(curve), 1e-12))
        curve = updated
        if iteration >= 2 and delta < tol:
            converged = True
            break
    coeffs, recon = solve_fixed_coefficients(x_mix, curve, backgrounds)
    return {
        "curve": curve,
        "coefficients": coeffs,
        "reconstruction": recon,
        "iterations": len(losses),
        "converged": converged,
        "final_loss": float(np.mean((x_mix - recon) ** 2)),
        "residual_norm_ratio": float(np.linalg.norm(x_mix - recon) / max(np.linalg.norm(x_mix), 1e-12)),
        "loss_trace": losses,
    }

class PriorNNSpectralModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.g_branch = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.LayerNorm(16),
            nn.Linear(16, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid(),
        )
        self.f_branch = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(), nn.LayerNorm(16),
            nn.Linear(16, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, input_size), nn.Softplus(),
        )

    def forward(self, concentration_inputs: torch.Tensor, wavenumber_template: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.g_branch(concentration_inputs).squeeze(-1), self.f_branch(wavenumber_template)

def fit_prior_nn(task: dict[str, Any], *, epochs: int, random_seed: int, device: str = "cpu") -> dict[str, Any]:
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed) % (2**32))
    x_mix = np.asarray(task["x_mix"], dtype=np.float32)
    concentrations = normalize_level(np.asarray(task["concentrations"], dtype=np.float32)).astype(np.float32)
    if len(task["bg"]) != 1:
        raise ValueError("Prior NN baseline requires exactly one known background")
    background = np.asarray(task["bg"][0], dtype=np.float32)
    x_axis = np.asarray(task["x_axis"], dtype=np.float32)
    x_norm = ((x_axis - x_axis.min()) / max(float(x_axis.max() - x_axis.min()), 1e-12)).astype(np.float32)
    resolved = torch.device(device)
    c_tensor = torch.tensor(concentrations[:, None], dtype=torch.float32, device=resolved)
    w_template = torch.tensor(np.tile(x_norm[None, :], (len(concentrations), 1)), dtype=torch.float32, device=resolved)
    y_true = torch.tensor(x_mix, dtype=torch.float32, device=resolved)
    bg = torch.tensor(background[None, :], dtype=torch.float32, device=resolved)
    model = PriorNNSpectralModel(x_mix.shape[1]).to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=50, min_lr=5e-5)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    loss_trace: list[float] = []
    started = time.time()
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        g, f = model(c_tensor, w_template)
        predicted = g[:, None] * f + (1.0 - g[:, None]) * bg
        loss = torch.mean((predicted - y_true) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        scheduler.step(value)
        loss_trace.append(value)
        if value < best_loss - 1e-10:
            best_loss = value
            best_epoch = epoch + 1
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Prior NN did not produce a valid state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        g, f_rows = model(c_tensor, w_template)
        curve = model.f_branch(torch.tensor(x_norm[None, :], dtype=torch.float32, device=resolved)).squeeze(0)
        reconstruction = g[:, None] * f_rows + (1.0 - g[:, None]) * bg
    curve_np = curve.cpu().numpy().astype(float)
    g_np = g.cpu().numpy().astype(float)
    recon_np = reconstruction.cpu().numpy().astype(float)
    return {
        "curve": curve_np,
        "coefficients": g_np,
        "reconstruction": recon_np,
        "epochs_requested": int(epochs),
        "epochs_run": int(epochs),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "runtime_sec": float(time.time() - started),
        "random_seed": int(random_seed),
        "residual_norm_ratio": float(np.linalg.norm(x_mix - recon_np) / max(np.linalg.norm(x_mix), 1e-12)),
        "loss_trace": loss_trace,
    }
