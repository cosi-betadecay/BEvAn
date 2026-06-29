import torch


def build_default_edges(
    x: torch.Tensor,
    n_bins: int,
    spacing: str,
    log_floor: float | None,
) -> torch.Tensor:
    """Build default bin edges for a 1D histogram.

    Args:
        x (torch.Tensor): Input tensor from which to derive the bin edges.
        n_bins (int): Number of bins to create.
        spacing (str): Spacing type for the bins ('linear' or 'log').
        log_floor (float | None): Floor value for log-spaced bins.

    Raises:
        ValueError: If the spacing is not 'linear' or 'log'.
        ValueError: If there are no positive values when building log-spaced edges.

    Returns:
        torch.Tensor: The bin edges.
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


def bins_from_counts(n_min: int, d: int, rho_floor: float, min_bins: int = 2) -> int:
    """Bins-per-axis sized so the scarcer class keeps ~``rho_floor`` events per cell.

    A ``d``-dimensional histogram has ``B ** d`` cells; choosing
    ``B = floor((n_min / rho_floor) ** (1 / d))`` keeps the expected count per
    cell near ``rho_floor`` for the limiting (minority) class, so the estimated
    density is not a set of memorized single-event spikes. Total cells stay
    ``O(n_min / rho_floor)`` — linear in the data, never blowing up the matrix.

    Args:
        n_min (int): Finite-feature event count of the limiting class for this term.
        d (int): Histogram dimensionality (1 or 2).
        rho_floor (float): Target events per cell.
        min_bins (int): Smallest bins-per-axis to return.

    Returns:
        int: Bins-per-axis, at least ``min_bins``.
    """
    if n_min <= 0:
        return min_bins
    return max(int((n_min / rho_floor) ** (1.0 / d)), min_bins)


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

    Args:
        x (torch.Tensor): _x_ values for the 2D histogram.
        y (torch.Tensor): _y_ values for the 2D histogram.
        x_bins (torch.Tensor | None, optional): Pre-built bin edges for the x-axis. Defaults to None.
        y_bins (torch.Tensor | None, optional): Pre-built bin edges for the y-axis. Defaults to None.
        n_bins_x (int, optional): Number of bins for the x-axis. Defaults to 200.
        n_bins_y (int, optional): Number of bins for the y-axis. Defaults to 200.
        spacing_x (str, optional): Spacing type for the x-axis ('linear' or 'log'). Defaults to "linear".
        spacing_y (str, optional): Spacing type for the y-axis ('linear' or 'log'). Defaults to "linear".
        log_x_floor (float | None, optional): Floor value for log-spaced x-axis bins. Defaults to None.
        log_y_floor (float | None, optional): Floor value for log-spaced y-axis bins. Defaults to None.
        smoothing (float, optional): Smoothing parameter for Laplace-style pseudo-counts. Defaults to 0.0.

    Raises:
        ValueError: If the input tensors are empty or if the binning parameters are invalid.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The normalized 2D probability matrix and the bin edges for both axes.
    """
    x = x.flatten()
    y = y.flatten()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("Cannot create probability matrix from empty tensors.")

    if x_bins is None:
        x_bins = build_default_edges(x, n_bins_x, spacing_x, log_x_floor)
    else:
        n_bins_x = x_bins.numel() - 1
        x_bins = x_bins.to(device=x.device, dtype=x.dtype)

    if y_bins is None:
        y_bins = build_default_edges(y, n_bins_y, spacing_y, log_y_floor)
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
    """Lookup density values from a 2D probability matrix for given (x, y) pairs.

    Args:
        x (torch.Tensor): _x_ values for which to lookup density values.
        y (torch.Tensor): _y_ values for which to lookup density values.
        matrix (torch.Tensor): 2D probability matrix from which to lookup values.
        x_bins (torch.Tensor): _x_ bin edges corresponding to the matrix.
        y_bins (torch.Tensor): _y_ bin edges corresponding to the matrix.

    Returns:
        torch.Tensor: Density values corresponding to the input (x, y) pairs. Out-of-range or NaN values are assigned a density of 0.0.
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

    Args:
        x (torch.Tensor): _x_ values for the 1D histogram.
        x_bins (torch.Tensor | None, optional): The bin edges for the x-axis. Defaults to None.
        n_bins_x (int, optional): The number of bins for the x-axis. Defaults to 20.
        spacing_x (str, optional): The spacing for the x-axis bins. Defaults to "linear".
        log_x_floor (float | None, optional): The floor value for the logarithmic spacing of the x-axis. Defaults to None.
        smoothing (float, optional): The amount of smoothing to apply to the probability vector. Defaults to 0.0.

    Raises:
        ValueError: If the input tensors are empty or if the binning parameters are invalid.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The normalized 1D probability vector and the bin edges for the x-axis.
    """
    x = x.flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        raise ValueError("Cannot create probability vector from empty tensor.")

    if x_bins is None:
        x_bins = build_default_edges(x, n_bins_x, spacing_x, log_x_floor)
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
    """1D analogue of :func:`lookup_density_values`. OOD / NaN → 0.0.

    Args:
        x (torch.Tensor): _x_ values for which to lookup density values.
        matrix (torch.Tensor): The density matrix.
        x_bins (torch.Tensor): The bin edges for the x-axis.

    Returns:
        torch.Tensor: The density values for the given x values.
    """
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
