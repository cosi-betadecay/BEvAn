import torch


def _build_default_edges(
    x: torch.Tensor,
    n_bins: int,
    spacing: str,
    log_floor: float | None,
) -> torch.Tensor:
    """Construct default bin edges for histogramming.

    ``spacing="linear"`` gives uniform linspace edges spanning the data range.
    ``spacing="log"`` gives geometrically-spaced edges over ``[floor, x.max()]``
    where ``floor = max(log_floor, smallest_positive_value)``. Zoglauer-style:
    densifies resolution near zero for residual-like features (ΔE, ARM) whose
    signal clusters at small values and whose tails are long.

    Values below the floor at lookup time clamp into the first bin via
    ``lookup_density_values``' existing out-of-support semantics.
    """
    if spacing == "linear":
        x_min, x_max = x.min(), x.max()
        if x_min == x_max:
            x_max = x_max + 1e-6
        return torch.linspace(x_min, x_max, n_bins + 1, device=x.device, dtype=x.dtype)
    if spacing == "log":
        floor = log_floor if log_floor is not None else 1e-3
        positive = x[x > 0]
        if positive.numel() == 0:
            raise ValueError("Cannot build log-spaced edges: no positive values in x.")
        x_min = max(float(positive.min().item()), float(floor))
        x_max = float(x.max().item())
        if x_min >= x_max:
            x_max = x_min + 1e-6
        log_edges = torch.linspace(
            float(torch.log(torch.tensor(x_min))),
            float(torch.log(torch.tensor(x_max))),
            n_bins + 1,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.exp(log_edges)
    raise ValueError(f"Unknown spacing {spacing!r}; expected 'linear' or 'log'.")


def build_density_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    x_bins: torch.Tensor | None = None,
    y_bins: torch.Tensor | None = None,
    n_bins_x: int = 200,
    n_bins_y: int = 200,
    spacing_x: str = "linear",
    spacing_y: str = "linear",
    log_x_floor: float | None = None,
    log_y_floor: float | None = None,
    smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a normalized 2D probability matrix over (x, y).

    With typical event counts in the low thousands the previous default of
    2000×2000 gave 4 M cells, the vast majority empty and noise-dominated.
    200×200 = 40 k cells gives enough resolution to retain structure while
    keeping bins statistically populated. Callers that need a different
    resolution can still pass n_bins_x / n_bins_y or pre-built x_bins / y_bins.

    ``smoothing`` adds Laplace-style pseudo-counts (α per cell) to every bin
    before normalization. This prevents zero-density bins from dominating the
    log-likelihood ratio for minority-class test events that happen to land
    in a cell their class never populated. Default 0.0 preserves legacy
    behaviour; the per-class likelihood builders pass a non-zero value.
    """
    x = x.flatten()
    y = y.flatten()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("Cannot create probability matrix from empty tensors.")

    if x_bins is None:
        x_bins = _build_default_edges(x, n_bins_x, spacing_x, log_x_floor)
    else:
        n_bins_x = x_bins.numel() - 1
        x_bins = x_bins.to(device=x.device, dtype=x.dtype)

    if y_bins is None:
        y_bins = _build_default_edges(y, n_bins_y, spacing_y, log_y_floor)
    else:
        n_bins_y = y_bins.numel() - 1
        y_bins = y_bins.to(device=y.device, dtype=y.dtype)

    # Clamp out-of-range values into the edge bins rather than dropping them.
    # This matters especially for log-spaced axes whose first edge is a
    # positive floor (e.g. 1e-3): values that are exactly 0 or sit below the
    # floor otherwise get discarded from both the training histogram and the
    # lookup, silently producing zero density for events that are physically
    # the "best" (ΔE≈0, ARM≈0) examples of a class.
    x_idx = torch.bucketize(x, x_bins, right=True) - 1
    y_idx = torch.bucketize(y, y_bins, right=True) - 1
    x_idx = x_idx.clamp(min=0, max=n_bins_x - 1)
    y_idx = y_idx.clamp(min=0, max=n_bins_y - 1)

    counts = torch.zeros((n_bins_x, n_bins_y), dtype=torch.float32, device=x.device)
    counts.index_put_(
        (x_idx, y_idx),
        torch.ones_like(x_idx, dtype=torch.float32),
        accumulate=True,
    )

    if smoothing > 0:
        counts = counts + smoothing

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
    # matching the convention in build_density_matrix. Out-of-range values
    # (below the first edge / above the last) are clamped into the edge bins
    # to match the build-side behavior; NaN entries still get 0.0.
    x_idx_full = torch.bucketize(x, x_bins, right=True) - 1
    y_idx_full = torch.bucketize(y, y_bins, right=True) - 1
    x_idx_full = x_idx_full.clamp(min=0, max=n_x - 1)
    y_idx_full = y_idx_full.clamp(min=0, max=n_y - 1)

    values[finite] = matrix[x_idx_full[finite], y_idx_full[finite]]
    return values


def build_density_matrix_1d(
    x: torch.Tensor,
    x_bins: torch.Tensor | None = None,
    n_bins_x: int = 20,
    spacing_x: str = "linear",
    log_x_floor: float | None = None,
    smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a normalized 1D probability vector over x.

    Mirrors :func:`build_density_matrix` but in a single dimension. Used to
    factor a 2D joint P(ΔE, y) into the marginal P(ΔE) and the conditional
    P(y | ΔE) so that ΔE evidence is not double-counted when the likelihood
    ratio multiplies several sub-spaces that all contain ΔE.

    ``smoothing`` adds Laplace-style pseudo-counts per bin (see
    :func:`build_density_matrix`).
    """
    x = x.flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        raise ValueError("Cannot create probability vector from empty tensor.")

    if x_bins is None:
        x_bins = _build_default_edges(x, n_bins_x, spacing_x, log_x_floor)
    else:
        n_bins_x = x_bins.numel() - 1
        x_bins = x_bins.to(device=x.device, dtype=x.dtype)

    # Clamp below-floor / above-ceiling values into the edge bins (see
    # rationale in build_density_matrix).
    x_idx = torch.bucketize(x, x_bins, right=True) - 1
    x_idx = x_idx.clamp(min=0, max=n_bins_x - 1)

    counts = torch.zeros((n_bins_x,), dtype=torch.float32, device=x.device)
    counts.index_put_((x_idx,), torch.ones_like(x_idx, dtype=torch.float32), accumulate=True)

    if smoothing > 0:
        counts = counts + smoothing

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
    x_idx_full = x_idx_full.clamp(min=0, max=n_x - 1)
    values[finite] = matrix[x_idx_full[finite]]
    return values
