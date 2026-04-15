import torch
from tqdm import tqdm


def build_density_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    x_bins: torch.Tensor | None = None,
    y_bins: torch.Tensor | None = None,
    n_bins_x: int = 2000,
    n_bins_y: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a normalized 2D probability matrix over (x, y).

    With typical event counts in the low thousands the previous default of
    2000×2000 gave 4 M cells, the vast majority empty and noise-dominated.
    200×200 = 40 k cells gives enough resolution to retain structure while
    keeping bins statistically populated. Callers that need a different
    resolution can still pass n_bins_x / n_bins_y or pre-built x_bins / y_bins.
    """
    x = x.flatten()
    y = y.flatten()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("Cannot create probability matrix from empty tensors.")

    if x_bins is None:
        x_min, x_max = x.min(), x.max()
        if x_min == x_max:
            x_max = x_max + 1e-6
        x_bins = torch.linspace(x_min, x_max, n_bins_x + 1, device=x.device, dtype=x.dtype)
    else:
        n_bins_x = x_bins.numel() - 1
        x_bins = x_bins.to(device=x.device, dtype=x.dtype)

    if y_bins is None:
        y_min, y_max = y.min(), y.max()
        if y_min == y_max:
            y_max = y_max + 1e-6
        y_bins = torch.linspace(y_min, y_max, n_bins_y + 1, device=y.device, dtype=y.dtype)
    else:
        n_bins_y = y_bins.numel() - 1
        y_bins = y_bins.to(device=y.device, dtype=y.dtype)

    # Include samples that land exactly on the first/last bin edge while still
    # excluding values truly outside the supplied range.
    x_idx = torch.bucketize(x, x_bins, right=True) - 1
    y_idx = torch.bucketize(y, y_bins, right=True) - 1

    valid_idx = (
        (x >= x_bins[0])
        & (x <= x_bins[-1])
        & (y >= y_bins[0])
        & (y <= y_bins[-1])
        & (x_idx >= 0)
        & (x_idx < n_bins_x)
        & (y_idx >= 0)
        & (y_idx < n_bins_y)
    )

    x_idx = x_idx[valid_idx]
    y_idx = y_idx[valid_idx]

    counts = torch.zeros((n_bins_x, n_bins_y), dtype=torch.float32, device=x.device)
    counts.index_put_(
        (x_idx, y_idx),
        torch.ones_like(x_idx, dtype=torch.float32),
        accumulate=True,
    )

    probs = counts / counts.sum().clamp_min(1e-8)

    return probs, x_bins, y_bins


def lookup_density_values(
    x: torch.Tensor,
    y: torch.Tensor,
    matrix: torch.Tensor,
    x_bins: torch.Tensor,
    y_bins: torch.Tensor,
) -> torch.Tensor:
    """Read `matrix` at the cell corresponding to each (x, y) query.

    Events that are NaN / ±inf, or that fall outside the training bin range
    on either axis, return 0.0 (they contribute no evidence to the likelihood
    ratio). This preserves the Zoglauer §4.5.3 semantics that an event must
    fall inside the response's support to update R.
    """
    x = x.flatten().to(device=matrix.device, dtype=x_bins.dtype)
    y = y.flatten().to(device=matrix.device, dtype=y_bins.dtype)

    values = torch.full_like(x, 0.0, dtype=matrix.dtype, device=matrix.device)
    finite = torch.isfinite(x) & torch.isfinite(y)
    if not torch.any(finite):
        return values

    n_x = matrix.shape[0]
    n_y = matrix.shape[1]

    # Use right=True so values on the upper bin edge map to the last bin,
    # matching the convention in build_density_matrix.
    x_idx_full = torch.bucketize(x, x_bins, right=True) - 1
    y_idx_full = torch.bucketize(y, y_bins, right=True) - 1

    in_range = (
        finite
        & (x >= x_bins[0])
        & (x <= x_bins[-1])
        & (y >= y_bins[0])
        & (y <= y_bins[-1])
        & (x_idx_full >= 0)
        & (x_idx_full < n_x)
        & (y_idx_full >= 0)
        & (y_idx_full < n_y)
    )

    values[in_range] = matrix[x_idx_full[in_range], y_idx_full[in_range]]
    return values


def build_density_matrix_1d(
    x: torch.Tensor,
    x_bins: torch.Tensor | None = None,
    n_bins_x: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a normalized 1D probability vector over x.

    Mirrors :func:`build_density_matrix` but in a single dimension. Used to
    factor a 2D joint P(ΔE, y) into the marginal P(ΔE) and the conditional
    P(y | ΔE) so that ΔE evidence is not double-counted when the likelihood
    ratio multiplies several sub-spaces that all contain ΔE.
    """
    x = x.flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        raise ValueError("Cannot create probability vector from empty tensor.")

    if x_bins is None:
        x_min, x_max = x.min(), x.max()
        if x_min == x_max:
            x_max = x_max + 1e-6
        x_bins = torch.linspace(x_min, x_max, n_bins_x + 1, device=x.device, dtype=x.dtype)
    else:
        n_bins_x = x_bins.numel() - 1
        x_bins = x_bins.to(device=x.device, dtype=x.dtype)

    x_idx = torch.bucketize(x, x_bins, right=True) - 1
    in_range = (x >= x_bins[0]) & (x <= x_bins[-1]) & (x_idx >= 0) & (x_idx < n_bins_x)
    x_idx = x_idx[in_range]

    counts = torch.zeros((n_bins_x,), dtype=torch.float32, device=x.device)
    counts.index_put_((x_idx,), torch.ones_like(x_idx, dtype=torch.float32), accumulate=True)

    probs = counts / counts.sum().clamp_min(1e-8)
    return probs, x_bins


def lookup_density_values_1d(
    x: torch.Tensor,
    matrix: torch.Tensor,
    x_bins: torch.Tensor,
) -> torch.Tensor:
    """1D analogue of :func:`lookup_density_values`. OOD / NaN → 0.0."""
    x = x.flatten().to(device=matrix.device, dtype=x_bins.dtype)
    values = torch.full_like(x, 0.0, dtype=matrix.dtype, device=matrix.device)
    finite = torch.isfinite(x)
    if not torch.any(finite):
        return values

    n_x = matrix.shape[0]
    x_idx_full = torch.bucketize(x, x_bins, right=True) - 1
    in_range = finite & (x >= x_bins[0]) & (x <= x_bins[-1]) & (x_idx_full >= 0) & (x_idx_full < n_x)
    values[in_range] = matrix[x_idx_full[in_range]]
    return values


def conditional_from_joint(joint: torch.Tensor, axis: int = 0, eps: float = 1e-12) -> torch.Tensor:
    """Convert a joint P(x, y) matrix into a conditional P(y | x) or P(x | y).

    With ``axis=0`` (default) each row sums to 1.0 where the row has any mass
    and to 0.0 where the row is completely empty. Use ``axis=1`` to normalize
    columns instead.

    Rationale: when the likelihood ratio multiplies P(ΔE, angle) · P(ΔE, arm)
    as if the two sub-spaces were conditionally independent, ΔE is counted
    twice. The fix is to multiply P(ΔE) · P(angle | ΔE) · P(arm | ΔE), which
    requires this helper to turn the two pre-existing joints into conditionals.
    """
    if joint.ndim != 2:
        raise ValueError(f"conditional_from_joint expects a 2D matrix, got shape {tuple(joint.shape)}")
    if axis == 0:
        row_sums = joint.sum(dim=1, keepdim=True)
        safe = row_sums.clamp_min(eps)
        cond = joint / safe
        # Zero-out rows that were completely empty.
        cond = torch.where(row_sums > 0, cond, torch.zeros_like(cond))
        return cond
    if axis == 1:
        col_sums = joint.sum(dim=0, keepdim=True)
        safe = col_sums.clamp_min(eps)
        cond = joint / safe
        cond = torch.where(col_sums > 0, cond, torch.zeros_like(cond))
        return cond
    raise ValueError(f"axis must be 0 or 1, got {axis}")
