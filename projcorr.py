"""
projcorr: Projection Correlation and Improved Projection Correlation on GPU (PyTorch)
=====================================================================================

PyTorch implementation of Projection Correlation (PC) and Improved Projection
Correlation (IPC / IPC-RCV) independence tests from:
  1. Zhu, L., Xu, K., Li, R. and Zhong, W. (2017). Projection correlation between two
     random vectors. Biometrika, 104(4), 829-843.
  2. Zhang, Y. and Zhu, L. (2024). Projective independence tests in high dimensions:
     the curses and the cures. Biometrika, 111(3), 1013-1027.

Optimized for GPU acceleration:
  - Automatic anchor chunking / streaming: memory footprint bounded by O(C * n^2)
    allowing scaling to arbitrarily large n without GPU OOM.
  - Batched permutation testing: parallelized tensor operations across all permutations.
  - Vectorized marginal screening: O(p * q) pairwise feature utilities computed via
    batched matrix multiplications instead of slow nested Python loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class HTestResult:
    """Statistical hypothesis test result object matching R's `htest` class."""
    statistic: Dict[str, float]
    p_value: float
    estimate: Dict[str, float]
    raw_estimate: Dict[str, float]
    method: str
    data_name: str
    alternative: str = "greater"
    null_value: Dict[str, float] = field(default_factory=lambda: {"correlation": 0.0})
    parameter: Dict[str, int] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        stat_str = ", ".join(f"{k} = {v:.5g}" for k, v in self.statistic.items())
        param_str = ", ".join(f"{k} = {v}" for k, v in self.parameter.items())
        if param_str:
            stat_line = f"{stat_str}, {param_str}, p-value = {self._format_p_value()}"
        else:
            stat_line = f"{stat_str}, p-value = {self._format_p_value()}"

        est_lines = []
        for k, v in self.estimate.items():
            est_lines.append(f"{k:>12}")
            est_lines.append(f"{v:12.7f}" if math.isfinite(v) else f"{'NA':>12}")

        lines = [
            f"\n\t{self.method}\n",
            f"data:  {self.data_name}",
            stat_line,
            f"alternative hypothesis: true correlation is greater than 0",
            "sample estimates:",
            est_lines[0],
            est_lines[1]
        ]
        return "\n".join(lines) + "\n"

    def _format_p_value(self) -> str:
        if not math.isfinite(self.p_value):
            return "NA"
        if self.p_value < 2.2e-16:
            return "< 2.2e-16"
        return f"{self.p_value:.5g}"


# =============================================================================
# Helper & Conversion Utilities
# =============================================================================

def _get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(data: Any, name: str, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if hasattr(data, "values"):  # pandas DataFrame / Series
        arr = data.values
    else:
        arr = data

    if isinstance(arr, torch.Tensor):
        t = arr.to(device=device, dtype=dtype)
    else:
        t = torch.as_tensor(arr, device=device, dtype=dtype)

    if t.ndim == 1:
        t = t.unsqueeze(1)
    elif t.ndim != 2:
        raise ValueError(f"`{name}` must be a 2D matrix or DataFrame, got shape {t.shape}.")

    return t


def _validate_paired_data(
    x: Any,
    y: Any,
    na_action: str = "fail",
    min_n: int = 2,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
    dev = _get_device(device)
    tx = _to_tensor(x, "x", dev)
    ty = _to_tensor(y, "y", dev)

    if tx.shape[0] != ty.shape[0]:
        raise ValueError(f"`x` and `y` must have the same number of rows: {tx.shape[0]} vs {ty.shape[0]}.")

    valid_mask = torch.isfinite(tx).all(dim=1) & torch.isfinite(ty).all(dim=1)
    omitted = int((~valid_mask).sum().item())

    if omitted > 0:
        if na_action == "fail":
            raise ValueError(f"Found {omitted} rows with missing or non-finite values and na_action='fail'.")
        tx = tx[valid_mask]
        ty = ty[valid_mask]

    n = tx.shape[0]
    if n < min_n:
        raise ValueError(f"At least {min_n} complete paired observations are required, got {n}.")

    data_name = "x and y"
    return tx, ty, omitted, data_name


def _median_1d(v: torch.Tensor) -> torch.Tensor:
    """Exact median of 1D tensor matching R & C++."""
    v_sorted = torch.sort(v.flatten())[0]
    n = len(v_sorted)
    if n % 2 == 1:
        return v_sorted[n // 2]
    return 0.5 * (v_sorted[n // 2 - 1] + v_sorted[n // 2])


def _median_cols(mat: torch.Tensor) -> torch.Tensor:
    """Exact median of each column of a 2D tensor [n, p]."""
    s = torch.sort(mat, dim=0)[0]
    n = mat.shape[0]
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# =============================================================================
# 1. Classic Projection Correlation (Zhu et al. 2017)
# =============================================================================

def double_center_3d(a: torch.Tensor) -> torch.Tensor:
    """Double centers a batch of matrices of shape [B, n, n]."""
    row_mean = a.mean(dim=2, keepdim=True)
    col_mean = a.mean(dim=1, keepdim=True)
    grand_mean = a.mean(dim=(1, 2), keepdim=True)
    return a - row_mean - col_mean + grand_mean


def anchor_angles_chunk(z: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """
    Computes angular matrices for an anchor chunk k in [start, end).
    Input: z [n, p]
    Output: angle [C, n, n] with anchor row and col set to 0.
    """
    n, _ = z.shape
    C = end - start
    k_idx = torch.arange(start, end, device=z.device)
    c_idx = torch.arange(C, device=z.device)

    # delta[c, i, :] = z[i] - z[k_idx[c]]
    delta = z.unsqueeze(0) - z[k_idx].unsqueeze(1)  # [C, n, p]
    norms = torch.norm(delta, dim=2, keepdim=True)  # [C, n, 1]
    safe_norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    unit = delta / safe_norms

    # Zero out anchor row
    unit[c_idx, k_idx, :] = 0.0

    cosine = torch.bmm(unit, unit.transpose(1, 2))
    cosine = torch.clamp(cosine, -1.0, 1.0)
    angle = torch.acos(cosine)

    # Zero out anchor row and column
    angle[c_idx, k_idx, :] = 0.0
    angle[c_idx, :, k_idx] = 0.0
    return angle


def pcor_components(
    tx: torch.Tensor,
    ty: torch.Tensor,
    anchor_chunk_size: int = 64
) -> Tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
    """
    Computes observed Pcov^2, S2, and self covariances with bounded memory O(C * n^2).
    """
    n = tx.shape[0]
    nd = float(n)
    cross_sum = torch.tensor(0.0, dtype=torch.float64, device=tx.device)
    s2_sum = torch.tensor(0.0, dtype=torch.float64, device=tx.device)
    self_x_sum = torch.tensor(0.0, dtype=torch.float64, device=tx.device)
    self_y_sum = torch.tensor(0.0, dtype=torch.float64, device=tx.device)

    sx_list = []
    sy_list = []

    for start in range(0, n, anchor_chunk_size):
        end = min(n, start + anchor_chunk_size)
        ax = anchor_angles_chunk(tx, start, end)
        ay = anchor_angles_chunk(ty, start, end)

        dc_ax = double_center_3d(ax)
        dc_ay = double_center_3d(ay)

        cross_sum += (dc_ax * dc_ay).sum()
        self_x_sum += (dc_ax * dc_ax).sum()
        self_y_sum += (dc_ay * dc_ay).sum()

        sx_c = ax.sum(dim=(1, 2))
        sy_c = ay.sum(dim=(1, 2))
        s2_sum += (sx_c * sy_c).sum()

        sx_list.append(sx_c)
        sy_list.append(sy_c)

    pcov2 = (cross_sum / (nd ** 3)).item()
    s2 = (s2_sum / (nd ** 5)).item()
    self_x = (self_x_sum / (nd ** 3)).item()
    self_y = (self_y_sum / (nd ** 3)).item()
    sx = torch.cat(sx_list)
    sy = torch.cat(sy_list)
    return pcov2, s2, self_x, self_y, sx, sy


def pcor_test(
    x: Any,
    y: Any,
    n_perm: int = 999,
    seed: Optional[int] = None,
    anchor_chunk_size: Optional[int] = None,
    perm_batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    na_action: str = "fail"
) -> HTestResult:
    """
    Projection Correlation Independence Test (Zhu et al. 2017).

    Computes sample projection correlation and permutation test on GPU.
    Uses automatic anchor chunking to scale to large sample sizes without OOM,
    and batched parallel permutations.

    Parameters:
    -----------
    x, y : Array-like or torch.Tensor of shape (n, p) and (n, q)
    n_perm : int, number of permutations (default: 999)
    seed : int, optional random seed for reproducibility
    anchor_chunk_size : int, optional anchor batch size for memory control (default: min(n, 64))
    perm_batch_size : int, number of permutations evaluated simultaneously on GPU (default: 128)
    device : 'cuda' or 'cpu', defaults to cuda if available
    na_action : 'fail' or 'omit'
    """
    tx, ty, omitted, data_name = _validate_paired_data(x, y, na_action, min_n=2, device=device)
    n = tx.shape[0]
    nd = float(n)

    if anchor_chunk_size is None:
        anchor_chunk_size = min(n, 64)

    # 1. Compute observed statistics with chunking
    pcov2, s2, self_x, self_y, sx, sy = pcor_components(tx, ty, anchor_chunk_size)

    denominator = math.sqrt(max(self_x, 0.0) * max(self_y, 0.0))
    pc2 = (pcov2 / denominator) if denominator > 0 else 0.0
    estimate = math.sqrt(min(1.0, max(0.0, pc2)))

    stat_denom = math.pi ** 2 - s2
    if stat_denom <= 0:
        raise ValueError("The projection-correlation statistic has a non-positive denominator.")
    statistic = n * pcov2 / stat_denom

    # 2. Permutation calibration
    if seed is not None:
        torch.manual_seed(seed)

    # Generate permutations
    perms_list = []
    for _ in range(n_perm):
        perms_list.append(torch.randperm(n, device=tx.device))
    perms = torch.stack(perms_list)  # [n_perm, n]

    pi_sq = math.pi ** 2
    perm_stats = []

    # Check if precomputing full 3D tensor fits comfortably in memory (<= 300 MB)
    mem_est_bytes = n * n * n * 8 * 2
    if mem_est_bytes <= 300 * 1024 * 1024:
        # Fast path: precompute full 3D tensors Dx and Dy
        ax_all = anchor_angles_chunk(tx, 0, n)
        ay_all = anchor_angles_chunk(ty, 0, n)
        dc_ax = double_center_3d(ax_all)
        dc_ay = double_center_3d(ay_all)

        # Evaluate permutations in batches of size perm_batch_size
        for start in range(0, n_perm, perm_batch_size):
            p_chunk = perms[start:start + perm_batch_size]  # [B, n]
            B = p_chunk.shape[0]
            p_rows = p_chunk.unsqueeze(-1).expand(B, n, n)
            p_cols = p_chunk.unsqueeze(1).expand(B, n, n)

            # s2_sum: [B]
            s_sum = (sy[p_chunk] * sx.unsqueeze(0)).sum(dim=1)

            # cross_sum: accumulate over anchors k
            c_sum = torch.zeros(B, dtype=torch.float64, device=tx.device)
            for k in range(n):
                dy_k = dc_ay[p_chunk[:, k]]  # [B, n, n]
                dy_perm = torch.gather(torch.gather(dy_k, 1, p_rows), 2, p_cols)
                c_sum += (dy_perm * dc_ax[k].unsqueeze(0)).sum(dim=(1, 2))

            perm_pcov2 = c_sum / (nd ** 3)
            perm_s2 = s_sum / (nd ** 5)
            stat_b = nd * perm_pcov2 / (pi_sq - perm_s2)
            perm_stats.extend(stat_b.cpu().tolist())
    else:
        # Memory-safe streaming path for very large n (n > 300)
        for b in range(n_perm):
            p = perms[b]
            ty_p = ty[p]
            p_pcov2, p_s2, _, _, _, _ = pcor_components(tx, ty_p, anchor_chunk_size)
            stat_b = nd * p_pcov2 / (pi_sq - p_s2)
            perm_stats.append(stat_b)

    exceed = sum(1 for s in perm_stats if s >= statistic)
    p_value = (1 + exceed) / (n_perm + 1)

    return HTestResult(
        statistic={"T": statistic},
        p_value=p_value,
        estimate={"PC": estimate},
        raw_estimate={"Pcov^2": pcov2},
        parameter={"permutations": n_perm},
        method="Projection correlation permutation test",
        data_name=data_name,
        details={
            "n": n, "p": tx.shape[1], "q": ty.shape[1], "s2": s2,
            "self_x": self_x, "self_y": self_y, "exceedances": exceed,
            "seed": seed, "device": str(tx.device), "rows_omitted": omitted
        }
    )


# =============================================================================
# 2. Improved Projection Correlation (Zhang & Zhu 2024)
# =============================================================================

def angular_matrix(z: torch.Tensor, sigma2: Optional[float] = None) -> Tuple[torch.Tensor, float]:
    """
    Computes pairwise angular kernel matrix from Zhang & Zhu (2024).
    Kernel: acos((sigma2 + z_i^T z_j) / sqrt((sigma2 + ||z_i||^2) * (sigma2 + ||z_j||^2)))
    """
    row_norms2 = (z * z).sum(dim=1)
    if sigma2 is None:
        sigma2 = float(_median_1d(row_norms2).item())

    aug_norm2 = sigma2 + row_norms2
    if (aug_norm2 <= 0).any():
        raise ValueError("Scale and row norms yield an undefined angular kernel.")

    cosine = (sigma2 + torch.matmul(z, z.T)) / torch.sqrt(torch.outer(aug_norm2, aug_norm2))
    cosine = torch.clamp(cosine, -1.0, 1.0)
    angle = torch.acos(cosine)
    angle.fill_diagonal_(0.0)
    return angle, sigma2


def ipcov2_from_angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes distinct-index U-statistic for IPCov^2 in O(n^2)."""
    n = float(a.shape[0])
    s1 = (a * b).sum()
    row_a = a.sum(dim=1)
    row_b = b.sum(dim=1)
    row_product = (row_a * row_b).sum()
    total_product = a.sum() * b.sum()

    term1 = s1 / (n * (n - 1.0))
    term2 = -2.0 * (row_product - s1) / (n * (n - 1.0) * (n - 2.0))
    term3 = (total_product - 4.0 * row_product + 2.0 * s1) / (n * (n - 1.0) * (n - 2.0) * (n - 3.0))
    return term1 + term2 + term3


def ipcor_test(
    x: Any,
    y: Any,
    sigma_x2: Optional[float] = None,
    sigma_y2: Optional[float] = None,
    p_method: str = "asymptotic",
    n_perm: int = 999,
    perm_batch_size: int = 256,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    na_action: str = "fail"
) -> HTestResult:
    """
    Improved Projection Correlation Independence Test (Zhang & Zhu 2024).

    Computes U-statistic estimator of improved projection correlation with
    asymptotic-normal or batched GPU permutation calibration.

    Parameters:
    -----------
    x, y : Array-like or torch.Tensor of shape (n, p) and (n, q)
    sigma_x2, sigma_y2 : Optional squared kernel scales (default: median of row norms^2)
    p_method : 'asymptotic', 'permutation', or 'auto' (selects permutation)
    n_perm : int, number of permutations if p_method is permutation (default: 999)
    perm_batch_size : int, batch size for GPU parallel permutations (default: 256)
    seed : int, optional random seed
    device : 'cuda' or 'cpu'
    na_action : 'fail' or 'omit'
    """
    tx, ty, omitted, data_name = _validate_paired_data(x, y, na_action, min_n=4, device=device)
    n = tx.shape[0]

    realized = "permutation" if p_method == "auto" else p_method
    if realized not in ("asymptotic", "permutation"):
        raise ValueError(f"Unknown p_method: {p_method}. Expected 'asymptotic', 'permutation', or 'auto'.")

    ax, sx2 = angular_matrix(tx, sigma_x2)
    ay, sy2 = angular_matrix(ty, sigma_y2)

    raw = ipcov2_from_angles(ax, ay)
    self_x = ipcov2_from_angles(ax, ax)
    self_y = ipcov2_from_angles(ay, ay)

    if self_x.item() <= 0 or self_y.item() <= 0:
        estimate = float("nan")
        statistic = float("nan")
        p_value = float("nan")
        ipc2 = float("nan")
        exceed = None
    else:
        denom = torch.sqrt(self_x * self_y)
        ipc2 = (raw / denom).item()
        statistic = n * ipc2 / math.sqrt(2.0)
        estimate = math.sqrt(max(0.0, ipc2)) if ipc2 >= -1e-12 else float("nan")

        if realized == "asymptotic":
            p_value = 0.5 * math.erfc(statistic / math.sqrt(2.0))
            exceed = None
        else:
            if seed is not None:
                torch.manual_seed(seed)

            # Batched GPU permutations for IPC
            perms_list = [torch.randperm(n, device=tx.device) for _ in range(n_perm)]
            perms = torch.stack(perms_list)  # [n_perm, n]

            perm_stats = []
            nd = float(n)
            row_a = ax.sum(dim=1)
            row_b = ay.sum(dim=1)
            total_product = ax.sum() * ay.sum()

            for start in range(0, n_perm, perm_batch_size):
                p_chunk = perms[start:start + perm_batch_size]  # [B, n]
                B = p_chunk.shape[0]
                p_rows = p_chunk.unsqueeze(-1).expand(B, n, n)
                p_cols = p_chunk.unsqueeze(1).expand(B, n, n)
                ay_batch = ay.unsqueeze(0).expand(B, n, n)
                ay_perm = torch.gather(torch.gather(ay_batch, 1, p_rows), 2, p_cols)

                s1 = (ay_perm * ax.unsqueeze(0)).sum(dim=(1, 2))  # [B]
                row_product = (row_b[p_chunk] * row_a.unsqueeze(0)).sum(dim=1)  # [B]

                raw_b = (s1 / (nd * (nd - 1.0))
                         - 2.0 * (row_product - s1) / (nd * (nd - 1.0) * (nd - 2.0))
                         + (total_product - 4.0 * row_product + 2.0 * s1) / (nd * (nd - 1.0) * (nd - 2.0) * (nd - 3.0)))
                stat_b = nd * (raw_b / denom) / math.sqrt(2.0)
                perm_stats.extend(stat_b.cpu().tolist())

            exceed = sum(1 for s in perm_stats if s >= statistic)
            p_value = (1 + exceed) / (n_perm + 1)

    param = {"permutations": n_perm} if realized == "permutation" else {}
    return HTestResult(
        statistic={"Z": statistic},
        p_value=p_value,
        estimate={"IPC": estimate},
        raw_estimate={"IPCov^2": raw.item()},
        parameter=param,
        method=f"Improved projection correlation {realized} test",
        data_name=data_name,
        details={
            "n": n, "p": tx.shape[1], "q": ty.shape[1], "ipc_squared": ipc2,
            "self_ipcov2_x": self_x.item(), "self_ipcov2_y": self_y.item(),
            "sigma_x2": sx2, "sigma_y2": sy2, "calibration": realized,
            "exceedances": exceed, "seed": seed, "device": str(tx.device),
            "rows_omitted": omitted
        }
    )


# =============================================================================
# 3. Refitted Cross-Validation (IPC-RCV, Zhang & Zhu 2024)
# =============================================================================

def marginal_ipc2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes all marginal IPC^2 utilities between every feature pair (x_j, y_l)
    fully vectorized via matrix multiplications. Output shape: [p, q].
    """
    n, p = x.shape
    q = y.shape[1]
    nd = float(n)

    # 1. 1D angular matrices for all p features of x: shape [p, n, n]
    norm2_x = x * x
    sig2_x = _median_cols(norm2_x)
    aug_x = sig2_x.unsqueeze(0) + norm2_x
    xxT = torch.bmm(x.T.unsqueeze(2), x.T.unsqueeze(1))
    cos_x = (sig2_x.view(p, 1, 1) + xxT) / torch.sqrt(torch.bmm(aug_x.T.unsqueeze(2), aug_x.T.unsqueeze(1)))
    cos_x = torch.clamp(cos_x, -1.0, 1.0)
    ax = torch.acos(cos_x)
    diag_idx = torch.arange(n, device=x.device)
    ax[:, diag_idx, diag_idx] = 0.0

    # 2. 1D angular matrices for all q features of y: shape [q, n, n]
    norm2_y = y * y
    sig2_y = _median_cols(norm2_y)
    aug_y = sig2_y.unsqueeze(0) + norm2_y
    yyT = torch.bmm(y.T.unsqueeze(2), y.T.unsqueeze(1))
    cos_y = (sig2_y.view(q, 1, 1) + yyT) / torch.sqrt(torch.bmm(aug_y.T.unsqueeze(2), aug_y.T.unsqueeze(1)))
    cos_y = torch.clamp(cos_y, -1.0, 1.0)
    ay = torch.acos(cos_y)
    ay[:, diag_idx, diag_idx] = 0.0

    # 3. Vectorized U-statistics: matrix multiplication over features
    ax_flat = ax.view(p, n * n)
    ay_flat = ay.view(q, n * n)
    s1 = torch.matmul(ax_flat, ay_flat.T)  # [p, q]

    row_a = ax.sum(dim=2)  # [p, n]
    row_b = ay.sum(dim=2)  # [q, n]
    row_prod = torch.matmul(row_a, row_b.T)  # [p, q]

    tot_a = ax_flat.sum(dim=1, keepdim=True)  # [p, 1]
    tot_b = ay_flat.sum(dim=1, keepdim=True)  # [q, 1]
    tot_prod = torch.matmul(tot_a, tot_b.T)  # [p, q]

    raw_mat = (s1 / (nd * (nd - 1.0))
               - 2.0 * (row_prod - s1) / (nd * (nd - 1.0) * (nd - 2.0))
               + (tot_prod - 4.0 * row_prod + 2.0 * s1) / (nd * (nd - 1.0) * (nd - 2.0) * (nd - 3.0)))

    # Self-IPCov2 for x: [p]
    s1_x = (ax_flat * ax_flat).sum(dim=1)
    r_x = (row_a * row_a).sum(dim=1)
    t_x = tot_a.squeeze(1) ** 2
    self_x = (s1_x / (nd * (nd - 1.0))
              - 2.0 * (r_x - s1_x) / (nd * (nd - 1.0) * (nd - 2.0))
              + (t_x - 4.0 * r_x + 2.0 * s1_x) / (nd * (nd - 1.0) * (nd - 2.0) * (nd - 3.0)))

    # Self-IPCov2 for y: [q]
    s1_y = (ay_flat * ay_flat).sum(dim=1)
    r_y = (row_b * row_b).sum(dim=1)
    t_y = tot_b.squeeze(1) ** 2
    self_y = (s1_y / (nd * (nd - 1.0))
              - 2.0 * (r_y - s1_y) / (nd * (nd - 1.0) * (nd - 2.0))
              + (t_y - 4.0 * r_y + 2.0 * s1_y) / (nd * (nd - 1.0) * (nd - 2.0) * (nd - 3.0)))

    denom = torch.sqrt(torch.outer(self_x, self_y))
    return raw_mat / denom


def _select_ipc_pairs(
    x_sub: torch.Tensor,
    y_sub: torch.Tensor,
    n_pairs: int
) -> Tuple[List[int], List[int], List[Tuple[float, int, int]]]:
    """
    Screens and selects top coordinate pairs by marginal IPC2 descending.
    Matches R's column-major flattening convention.
    """
    p = x_sub.shape[1]
    q = y_sub.shape[1]
    util = marginal_ipc2(x_sub, y_sub)

    pairs = []
    for l in range(q):
        for j in range(p):
            pairs.append((util[j, l].item(), j, l))

    pairs.sort(key=lambda item: item[0], reverse=True)
    top_pairs = pairs[:n_pairs]
    sel_x = sorted(list(set(item[1] for item in top_pairs)))
    sel_y = sorted(list(set(item[2] for item in top_pairs)))
    return sel_x, sel_y, top_pairs


def _rcv_statistic(
    x: torch.Tensor,
    y: torch.Tensor,
    first_half: torch.Tensor,
    n_pairs: int
) -> Tuple[float, float, float, Dict[str, Any]]:
    n = x.shape[0]
    mask = torch.ones(n, dtype=torch.bool, device=x.device)
    mask[first_half] = False
    second_half = torch.arange(n, device=x.device)[mask]

    # Stage 1: Screen on first_half, evaluate on second_half
    sel_x1, sel_y1, pairs1 = _select_ipc_pairs(x[first_half], y[first_half], n_pairs)
    ax1, _ = angular_matrix(x[second_half][:, sel_x1])
    ay1, _ = angular_matrix(y[second_half][:, sel_y1])
    r1 = ipcov2_from_angles(ax1, ay1)
    s1x = ipcov2_from_angles(ax1, ax1)
    s1y = ipcov2_from_angles(ay1, ay1)
    q1 = (r1 / torch.sqrt(s1x * s1y)).item() if s1x.item() > 0 and s1y.item() > 0 else float("nan")

    # Stage 2: Screen on second_half, evaluate on first_half
    sel_x2, sel_y2, pairs2 = _select_ipc_pairs(x[second_half], y[second_half], n_pairs)
    ax2, _ = angular_matrix(x[first_half][:, sel_x2])
    ay2, _ = angular_matrix(y[first_half][:, sel_y2])
    r2 = ipcov2_from_angles(ax2, ay2)
    s2x = ipcov2_from_angles(ax2, ax2)
    s2y = ipcov2_from_angles(ay2, ay2)
    q2 = (r2 / torch.sqrt(s2x * s2y)).item() if s2x.item() > 0 and s2y.item() > 0 else float("nan")

    stat = (q1 + q2) / 2.0
    details = {
        "q1": q1, "q2": q2,
        "sel_x1": sel_x1, "sel_y1": sel_y1, "pairs1": pairs1,
        "sel_x2": sel_x2, "sel_y2": sel_y2, "pairs2": pairs2
    }
    return stat, q1, q2, details


def ipcor_rcv_test(
    x: Any,
    y: Any,
    n_pairs: Optional[int] = None,
    p_method: str = "asymptotic",
    n_perm: int = 999,
    seed: Optional[int] = None,
    first_half: Optional[Union[List[int], torch.Tensor]] = None,
    device: Optional[Union[str, torch.device]] = None,
    na_action: str = "fail"
) -> HTestResult:
    """
    Improved Projection Correlation Test with Refitted Cross-Validation (Zhang & Zhu 2024).

    Implements Algorithm 1: split-sample marginal screening, cross-fitted testing,
    and averaging of the two squared IPC values.

    Parameters:
    -----------
    x, y : Array-like or torch.Tensor of shape (n, p) and (n, q)
    n_pairs : int, number of coordinate pairs (default: ceil(log(n)))
    p_method : 'asymptotic', 'permutation', or 'auto'
    n_perm : int, number of permutations (default: 999)
    seed : int, optional random seed
    first_half : optional pre-specified sample indices for the first split
    device : 'cuda' or 'cpu'
    na_action : 'fail' or 'omit'
    """
    tx, ty, omitted, data_name = _validate_paired_data(x, y, na_action, min_n=8, device=device)
    n = tx.shape[0]
    if n % 2 != 0:
        raise ValueError("`ipcor_rcv_test` requires an even sample size.")

    if n_pairs is None:
        n_pairs = int(math.ceil(math.log(n)))

    p_total = tx.shape[1] * ty.shape[1]
    if n_pairs > p_total:
        raise ValueError(f"`n_pairs` ({n_pairs}) cannot exceed total pairs ({p_total}).")

    realized = "permutation" if p_method == "auto" else p_method
    if realized not in ("asymptotic", "permutation"):
        raise ValueError(f"Unknown p_method: {p_method}.")

    if first_half is None:
        if seed is not None:
            torch.manual_seed(seed)
        perm = torch.randperm(n, device=tx.device)
        split_half = torch.sort(perm[:n // 2])[0]
    else:
        split_half = torch.as_tensor(first_half, dtype=torch.long, device=tx.device)

    observed_stat, t1, t2, rcv_details = _rcv_statistic(tx, ty, split_half, n_pairs)
    standardized = n * observed_stat / 2.0

    if not math.isfinite(observed_stat):
        p_value = float("nan")
        exceed = None
    else:
        if realized == "asymptotic":
            p_value = 0.5 * math.erfc(standardized / math.sqrt(2.0))
            exceed = None
        else:
            if seed is not None:
                torch.manual_seed(seed + 1 if seed is not None else 0)
            perm_stats = []
            for _ in range(n_perm):
                p = torch.randperm(n, device=tx.device)
                cand, _, _, _ = _rcv_statistic(tx, ty[p], split_half, n_pairs)
                stat_p = n * cand / 2.0
                perm_stats.append(stat_p)
            exceed = sum(1 for s in perm_stats if s >= standardized)
            p_value = (1 + exceed) / (n_perm + 1)

    param = {"permutations": n_perm} if realized == "permutation" else {}
    return HTestResult(
        statistic={"Z": standardized},
        p_value=p_value,
        estimate={"RCV T": observed_stat},
        raw_estimate={"T1": t1, "T2": t2},
        parameter=param,
        method=f"Improved projection correlation RCV {realized} test",
        data_name=data_name,
        details={
            "n": n, "p": tx.shape[1], "q": ty.shape[1], "n_pairs": n_pairs,
            "calibration": realized, "first_half": split_half.cpu().tolist(),
            "selected_pairs": {
                "first_to_second": rcv_details["pairs1"],
                "second_to_first": rcv_details["pairs2"]
            },
            "exceedances": exceed, "seed": seed, "device": str(tx.device),
            "rows_omitted": omitted
        }
    )
