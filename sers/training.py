import time
from typing import Any, Dict

import numpy as np
import torch

from .features import compute_ground_truth_metrics, compute_input_features, compute_output_features
from .model import FourierFeatureMapping, SERSDecomposition, custom_loss, weights_init


def run_single_trial(
    task: Dict[str, Any],
    theta: Dict[str, Any],
    trial_id: str,
    epochs: int = 300,
    device: str | None = None,
    torch_seed: int | None = None,
    return_artifacts: bool = False,
    disable_fourier: bool = False,
) -> Dict[str, Any]:
    runtime_start = time.time()
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch_seed is not None:
        torch.manual_seed(int(torch_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(torch_seed))

    x_axis = np.asarray(task["x_axis"], dtype=np.float32)
    x_mix = np.asarray(task["x_mix"], dtype=np.float32)
    # Historical field name from the task schema. Values encode ordered or
    # relative levels, not absolute physical concentrations.
    concentrations = np.asarray(task["concentrations"], dtype=np.float32)
    backgrounds = [np.asarray(background, dtype=np.float32) for background in task["bg"]]
    gt_f = None if task["gt_f"] is None else np.asarray(task["gt_f"], dtype=np.float32)

    input_features = compute_input_features(x_mix, concentrations, backgrounds)

    num_concentrations, num_points = x_mix.shape
    x_norm = (x_axis - x_axis.min()) / max(float(x_axis.max() - x_axis.min()), 1e-12)
    # Normalize the relative level coordinate to [0, 1] before training.
    s_norm = (concentrations - concentrations.min()) / max(float(concentrations.max() - concentrations.min()), 1e-12)

    x_long = np.tile(x_norm, num_concentrations).reshape(-1, 1)
    s_long = np.repeat(s_norm, num_points).reshape(-1, 1)
    y_long = x_mix.reshape(-1, 1) * theta["scale_factor"]
    bg_long = [np.tile(background, num_concentrations).reshape(-1, 1) * theta["scale_factor"] for background in backgrounds]

    x_train = torch.tensor(x_long, dtype=torch.float32, device=resolved_device)
    s_train = torch.tensor(s_long, dtype=torch.float32, device=resolved_device)
    y_train = torch.tensor(y_long, dtype=torch.float32, device=resolved_device)
    bg_train = [torch.tensor(background, dtype=torch.float32, device=resolved_device) for background in bg_long]

    if disable_fourier:
        x_embed = x_train
    else:
        fourier_mapping = FourierFeatureMapping(num_frequencies=int(theta["num_frequencies"])).to(resolved_device)
        x_embed = fourier_mapping(x_train)
    model = SERSDecomposition(
        input_x_dim=x_embed.shape[1],
        hidden_dim=int(theta["hidden_dim"]),
        z_dim=int(theta["z_dim"]),
        num_backgrounds=len(backgrounds),
    ).to(resolved_device)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(theta["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs, 1), gamma=0.5)

    last_loss = None
    for epoch in range(epochs):
        model.train()
        coeffs, f_x = model(x_embed, s_train)
        loss, _ = custom_loss(
            y_true=y_train,
            coeffs=coeffs,
            f_x=f_x,
            backgrounds=bg_train,
            scale_mode=float(theta["scale_factor"]),
            lambda_penalty=float(theta["lambda_penalty"]),
            lambda_flat=float(theta["lambda_flat"]),
            apply_flatness=epoch >= epochs // 2,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())

    with torch.no_grad():
        model.eval()
        coeffs, f_x = model(x_embed, s_train)
        _, recon = custom_loss(
            y_true=y_train,
            coeffs=coeffs,
            f_x=f_x,
            backgrounds=bg_train,
            scale_mode=float(theta["scale_factor"]),
            lambda_penalty=float(theta["lambda_penalty"]),
            lambda_flat=float(theta["lambda_flat"]),
            apply_flatness=True,
        )

    coeffs_np = coeffs.detach().cpu().numpy().reshape(num_concentrations, num_points, -1)
    f_x_np = f_x.detach().cpu().numpy().reshape(num_concentrations, num_points)
    recon_np = recon.detach().cpu().numpy().reshape(num_concentrations, num_points) / theta["scale_factor"]

    target_coeff = coeffs_np[:, :, 0].mean(axis=1)
    bg_coeffs = coeffs_np[:, :, 1:].mean(axis=1)

    f0 = f_x_np[0]
    bg0 = bg_long[0].reshape(num_concentrations, num_points)[0]
    f_area = np.trapz(f0, x=x_axis)
    bg_area = np.trapz(bg0, x=x_axis)
    scale_ratio = 1.0 if abs(f_area) < 1e-12 else bg_area / f_area
    f_final = f0 * scale_ratio
    coeff_scale_ratio = 1.0 if abs(bg_area) < 1e-12 else f_area / bg_area
    g_final = target_coeff * coeff_scale_ratio

    if gt_f is not None:
        gt_f = gt_f.astype(np.float32) * theta["scale_factor"]

    output_features = compute_output_features(
        f_final=f_final,
        coeff_target=target_coeff,
        recon_matrix=recon_np,
        x_mix=x_mix,
        concentrations=concentrations,
        final_loss=last_loss if last_loss is not None else float("nan"),
    )
    ground_truth_metrics = compute_ground_truth_metrics(gt_f, f_final)
    output_summary = {
        "f_final_l2": float(np.linalg.norm(f_final)),
        "f_final_area": float(np.trapz(f_final, x=x_axis)),
        "f_final_mean": float(np.mean(f_final)),
        "f_final_std": float(np.std(f_final)),
        "f_final_min": float(np.min(f_final)),
        "f_final_max": float(np.max(f_final)),
        "target_coeff_mean": float(np.mean(target_coeff)),
        "target_coeff_std": float(np.std(target_coeff)),
    }

    result = {
        "trial_id": trial_id,
        "dataset_id": task["dataset_id"],
        "task_id": task["task_id"],
        "theta": theta,
        "input_features": input_features,
        "output_features": output_features,
        "output_summary": output_summary,
        "training_metrics": {
            "epochs": int(epochs),
            "final_loss": last_loss,
            "runtime_sec": float(time.time() - runtime_start),
        },
        "ground_truth_metrics": ground_truth_metrics,
        "metadata": {
            "preprocess_mode": task["metadata"].get("preprocess_mode"),
            "num_backgrounds": len(backgrounds),
            "bg_coeff_means": bg_coeffs.mean(axis=0).astype(float).tolist() if bg_coeffs.size else [],
        },
    }
    if return_artifacts:
        result["artifacts"] = {
            "x_axis": x_axis.astype(float).tolist(),
            "f_final": f_final.astype(float).tolist(),
            "gt_f": None if gt_f is None else gt_f.astype(float).tolist(),
            "target_coeff": target_coeff.astype(float).tolist(),
            "background_coeffs": bg_coeffs.astype(float).tolist(),
            "g_final": g_final.astype(float).tolist(),
            "recon_matrix": recon_np.astype(float).tolist(),
            "x_mix": x_mix.astype(float).tolist(),
        }
    return result
