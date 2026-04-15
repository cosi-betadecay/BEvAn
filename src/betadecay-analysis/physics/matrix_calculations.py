import omegaconf
import torch
from tqdm import tqdm

from mathematics.geometry import theta_geo, theta_kin


def annihilation_angle(
    positions: torch.Tensor, n_hits: int | None = None, sizes: torch.Tensor | None = None
) -> torch.Tensor:
    """Smallest pairwise hit-direction cosine — a β⁺ two-photon geometry check.

    Computes cos(α) for every pair of hit-to-hit direction vectors in the event
    and returns the cosine closest to −1 (i.e. the most "back-to-back" pair).
    For a fully absorbed β⁺→2γ event the two 511-keV photons are emitted
    anti-parallel, so a good event pushes this value toward −1.

    This is **not** one of Zoglauer's §4.5 data-space dimensions
    (cos φ_kin, cos φ_geo, cos θ_kin, cos θ_geo). It is a β⁺-annihilation
    specific topology estimator adapted for the signal-vs-background task.
    """

    def all_vector_cosines_matrix_calculation(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
            unbatched = True
        elif positions.ndim == 3:
            unbatched = False
        else:
            raise ValueError(f"Expected positions shape (N, 3) or (B, N, 3), got {tuple(positions.shape)}")

        n = positions.shape[1]
        if n < 3:
            empty = positions.new_empty((positions.shape[0], 0))
            return empty.squeeze(0) if unbatched else empty

        deltas = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 3)
        i, j = torch.triu_indices(n, n, offset=1, device=positions.device)
        vectors = deltas[:, i, j, :]  # (B, M, 3), M = N choose 2
        vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

        cosine_matrix = vectors @ vectors.transpose(-1, -2)  # (B, M, M)
        m = vectors.shape[1]
        a, b = torch.triu_indices(m, m, offset=1, device=positions.device)
        cosines = cosine_matrix[:, a, b]  # (B, K)

        return cosines.amin().reshape(-1)

    def all_vector_cosines_blockwise(
        positions: torch.Tensor,
        eps: float = 1e-12,
        combo_block_size: int = 1024,
        vec_block_size: int = 128,
    ) -> torch.Tensor:
        if positions.ndim != 3:
            raise ValueError(f"Expected positions shape (B, N, 3), got {tuple(positions.shape)}")

        n = positions.shape[1]
        if n < 3:
            return positions.new_empty((positions.shape[0], 0))

        total_b = positions.shape[0]
        global_best = torch.full((total_b,), 1.0, dtype=positions.dtype, device=positions.device)

        pair_i, pair_j = torch.triu_indices(n, n, offset=1, device=positions.device)

        combo_starts = range(0, total_b, combo_block_size)
        for b0 in tqdm(combo_starts, desc="Annihilation angle blocks", unit="block", leave=False):
            b1 = min(b0 + combo_block_size, total_b)
            pos_chunk = positions[b0:b1]  # (Bc, N, 3)

            deltas = pos_chunk[:, :, None, :] - pos_chunk[:, None, :, :]  # (Bc, N, N, 3)
            vectors = deltas[:, pair_i, pair_j, :]  # (Bc, M, 3)
            vectors = vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)

            bsz, m, _ = vectors.shape
            chunk_best = torch.full((bsz,), 1.0, dtype=vectors.dtype, device=vectors.device)

            for i0 in range(0, m, vec_block_size):
                i1 = min(i0 + vec_block_size, m)
                vi = vectors[:, i0:i1, :]  # (Bc, bi, 3)

                for j0 in range(i0, m, vec_block_size):
                    j1 = min(j0 + vec_block_size, m)
                    vj = vectors[:, j0:j1, :]  # (Bc, bj, 3)

                    block = torch.matmul(vi, vj.transpose(-1, -2))  # (Bc, bi, bj)

                    if i0 == j0:
                        bi = i1 - i0
                        bj = j1 - j0
                        diag = torch.eye(bi, bj, dtype=torch.bool, device=block.device).unsqueeze(0)
                        block = block.masked_fill(diag, 1.0)

                    block_min = block.amin(dim=(1, 2))
                    chunk_best = torch.minimum(chunk_best, block_min)

            global_best[b0:b1] = chunk_best

        return global_best.amin().reshape(-1)

    if sizes is not None:
        if positions.ndim != 3 or sizes.ndim != 1:
            raise ValueError(
                "Sized annihilation_angle expects positions shape (B, N, 3) and sizes shape (B,), "
                f"got {tuple(positions.shape)} and {tuple(sizes.shape)}"
            )
        if positions.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Positions batch dimension and sizes length must match, "
                f"got {tuple(positions.shape)} and {tuple(sizes.shape)}"
            )

        outputs = []
        for size in torch.unique(sizes, sorted=True):
            size_int = int(size.item())
            group = sizes == size
            if size_int < 3:
                outputs.append(positions.new_tensor(float("nan")).reshape(1))
                continue
            outputs.append(annihilation_angle(positions[group, :size_int, :], n_hits=size_int))

        stacked = torch.stack(outputs).flatten()
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return positions.new_tensor(float("nan")).reshape(1)
        return finite.amin().reshape(1)

    if n_hits is None:
        n_hits = positions.shape[0] if positions.ndim == 2 else positions.shape[1]

    if n_hits <= 14:
        cosines = all_vector_cosines_matrix_calculation(positions)
    else:
        blockwise_positions = positions.unsqueeze(0) if positions.ndim == 2 else positions
        cosines = all_vector_cosines_blockwise(blockwise_positions)

    if cosines.numel() == 0:
        return positions.new_tensor(float("nan")).reshape(1)

    angle_diffs = torch.abs(cosines + 1.0)

    return cosines[torch.argmin(angle_diffs)].reshape(1)


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    cfg: omegaconf.dictconfig.DictConfig,
    sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Angular Resolution Measure — min |θ_geo − θ_kin| across hit permutations.

    This is the closest analogue to Zoglauer's dφ-criterion (eq. 4.21–4.22 in
    §4.5.1.3): the difference between the total scatter angle obtained from
    pure geometry (hit positions) and from Compton kinematics (energies).
    For a correctly sequenced, fully absorbed event ARM → 0.
    """
    if sizes is not None:
        if energies.ndim != 2 or positions.ndim != 3 or sizes.ndim != 1:
            raise ValueError(
                "Sized arm expects energies shape (B, N), positions shape (B, N, 3), and sizes shape (B,), "
                f"got {tuple(energies.shape)}, {tuple(positions.shape)}, and {tuple(sizes.shape)}"
            )
        if energies.shape[0] != sizes.shape[0] or positions.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Energies/positions batch dimension and sizes length must match, "
                f"got {tuple(energies.shape)}, {tuple(positions.shape)}, and {tuple(sizes.shape)}"
            )

        outputs = []
        for size in torch.unique(sizes, sorted=True):
            size_int = int(size.item())
            group = sizes == size
            if size_int < 3:
                outputs.append(positions.new_tensor(1.0).reshape(1))
                continue
            outputs.append(arm(energies[group, :size_int], positions[group, :size_int, :], cfg))
        return torch.stack(outputs).min().reshape(1)

    n_pos = positions.shape[0] if positions.ndim == 2 else positions.shape[1]
    if n_pos < 3:
        return positions.new_tensor(1.0)  # TODO: find a good replacement for this that actually makes sense

    arm_value = torch.min(torch.abs(theta_geo(positions, cfg)[0] - theta_kin(energies, cfg)[0]))

    return arm_value.reshape(1)


def delta_E(
    energies: torch.Tensor, ref_energy: float = 511.0, sizes: torch.Tensor | None = None
) -> torch.Tensor:
    """Annihilation-energy residual |ΣE − ref_energy| (default ref = 511 keV).

    This is **not** Zoglauer's E_in (the photon energy before a Compton
    interaction, §4.5.3). It is an absolute distance from the expected total
    deposit for a fully absorbed β⁺ single-photon annihilation signature,
    used here as a signal-vs-background discriminator.
    """
    if sizes is not None:
        if energies.ndim != 2 or sizes.ndim != 1:
            raise ValueError(
                "Sized delta_E expects energies shape (B, N) and sizes shape (B,), "
                f"got {tuple(energies.shape)} and {tuple(sizes.shape)}"
            )
        if energies.shape[0] != sizes.shape[0]:
            raise ValueError(
                "Energies batch dimension and sizes length must match, "
                f"got {tuple(energies.shape)} and {tuple(sizes.shape)}"
            )
        seq = torch.arange(energies.shape[1], device=energies.device).unsqueeze(0)
        valid = seq < sizes.unsqueeze(1)
        E = torch.where(valid, energies, torch.zeros_like(energies)).sum(dim=1)
    else:
        E = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
    delta_E = torch.abs(E - ref_energy)

    return delta_E.min().reshape(1)


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
