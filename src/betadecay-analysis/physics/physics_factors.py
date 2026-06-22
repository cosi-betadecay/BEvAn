import omegaconf
import torch
from mathematics.geometry import theta_geo, theta_kin

############################################
# Annihilation Angle
############################################

# PRODUCTION statistic: the vertex-anchored back-to-back annihilation score,
# ``_back_to_back_score``. A real e+ annihilation emits two 511 keV photons in
# opposite directions from a common vertex; the sim truth (data/Activation.sim)
# shows that vertex sits within ~2 mm of a hit 95% of the time, so each hit is
# tried as the vertex and the two most anti-parallel arm directions from it give a
# back-to-back cosine (≈ -1 for a real pair). A single Compton photon can deposit
# at most ~511 keV total, so a candidate whose two-arm energy does not exceed that
# ceiling is a single-photon collinear fake and is rejected; survivors pay a
# deficit-only penalty rewarding a two-arm total toward 1022 keV (partial deposits
# are common — per-arm energy is NOT gated at 511). Lower score = more
# annihilation-like. NaN where no subset has the structure (only ~29% of
# annihilations deposit both photons — this is a high-purity minority signal;
# delta_E carries the single-photon 511 keV events).


# --- Back-to-back annihilation score (vertex-anchored, energy-gated) ----------
# Thresholds are derived from data/Activation.sim truth (see project memory):
B2B_E_FLOOR = 511.0 - 3.0 * (2.25 / 2.355)  # ~508 keV: a single Compton photon
#                                             cannot deposit more than this total
B2B_E_TARGET = 1022.0  # two fully-absorbed 511 keV photons
B2B_SIGMA_E = 250.0  # deficit-tolerant energy scale (partial deposits dominate)
B2B_LAMBDA = 0.3  # energy-penalty strength, in cosine units


def _per_subset_back_to_back(
    positions: torch.Tensor, energies: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Vertex-anchored back-to-back annihilation score, one value per subset.

    For each hit taken as the annihilation vertex, the two most anti-parallel
    arm directions from that vertex give a back-to-back cosine (≈ -1 for a real
    pair). The two-arm energy (all non-vertex hits — the split itself does not
    change their sum) must exceed ``B2B_E_FLOOR`` or the candidate is a
    single-photon collinear fake and is rejected; survivors add a deficit-only
    Gaussian penalty toward ``B2B_E_TARGET``. The subset's score is the best
    (lowest) vertex. Lower = more annihilation-like.

    Args:
        positions: ``(B, n, 3)`` hit positions for B subsets of size n.
        energies: ``(B, n)`` per-hit deposited energy.

    Returns:
        ``(B,)`` scores; ``+inf`` for a subset with no gate-passing vertex
        (also where ``n < 3``, or for NaN-padded subsets).
    """
    B, n, _ = positions.shape
    if n < 3:
        return positions.new_full((B,), float("inf"))
    inf = positions.new_tensor(float("inf"))
    total_E = energies.sum(dim=1)  # (B,)
    best = positions.new_full((B,), float("inf"))
    for v in range(n):
        others = [k for k in range(n) if k != v]
        D = positions[:, others, :] - positions[:, v : v + 1, :]  # (B, m, 3)
        D = D / torch.linalg.norm(D, dim=-1, keepdim=True).clamp_min(eps)
        C = D @ D.transpose(-1, -2)  # (B, m, m)
        m = len(others)
        eye = torch.eye(m, dtype=torch.bool, device=positions.device).unsqueeze(0)
        cos_bb = C.masked_fill(eye, float("inf")).amin(dim=(1, 2))  # most anti-parallel
        e_arms = total_E - energies[:, v]  # two-arm energy (vertex excluded)
        deficit = (B2B_E_TARGET - e_arms).clamp_min(0.0)
        e_term = B2B_LAMBDA * (1.0 - torch.exp(-(deficit**2) / (2.0 * B2B_SIGMA_E**2)))
        score = torch.where(e_arms > B2B_E_FLOOR, cos_bb + e_term, inf)
        best = torch.minimum(best, score)
    return best


def _back_to_back_score(
    positions: torch.Tensor,
    energies: torch.Tensor,
    sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-event back-to-back score over the reconstructed subset pool.

    Mirrors the sized reduction of ``delta_E``/``arm``: group combos by subset
    size, score each, and return the per-event best (lowest = most
    annihilation-like), or NaN when no subset has >=3 hits with a gate-passing
    vertex.
    """
    if sizes is None:
        P = positions if positions.ndim == 3 else positions.unsqueeze(0)
        E = energies if energies.ndim == 2 else energies.unsqueeze(0)
        score = _per_subset_back_to_back(P, E)
        finite = score[torch.isfinite(score)]
        return (finite.amin() if finite.numel() else positions.new_tensor(float("nan"))).reshape(1)

    if positions.ndim != 3 or energies.ndim != 2 or sizes.ndim != 1:
        raise ValueError(
            "Sized back-to-back expects positions (B, N, 3), energies (B, N), sizes (B,), "
            f"got {tuple(positions.shape)}, {tuple(energies.shape)}, {tuple(sizes.shape)}"
        )
    if positions.shape[0] != sizes.shape[0] or energies.shape[0] != sizes.shape[0]:
        raise ValueError(
            "Batch dimension and sizes length must match, "
            f"got {tuple(positions.shape)}, {tuple(energies.shape)}, {tuple(sizes.shape)}"
        )

    outputs = []
    for size in torch.unique(sizes, sorted=True):
        size_int = int(size.item())
        if size_int < 3:
            outputs.append(positions.new_tensor(float("nan")).reshape(1))
            continue
        group = sizes == size
        score = _per_subset_back_to_back(positions[group, :size_int, :], energies[group, :size_int])
        finite = score[torch.isfinite(score)]
        if finite.numel() == 0:
            outputs.append(positions.new_tensor(float("nan")).reshape(1))
            continue
        outputs.append(finite.amin().reshape(1))

    stacked = torch.stack(outputs).flatten()
    finite = stacked[torch.isfinite(stacked)]
    if finite.numel() == 0:
        return positions.new_tensor(float("nan")).reshape(1)
    return finite.amin().reshape(1)


def annihilation_angle(
    positions: torch.Tensor,
    sizes: torch.Tensor | None = None,
    energies: torch.Tensor | None = None,
) -> torch.Tensor:
    # With per-hit energies, return the vertex-anchored back-to-back annihilation
    # score (geometry + single-photon energy gate). Production always supplies
    # energies; the legacy pure-geometry min-cos fallback was removed (unused).
    return _back_to_back_score(positions, energies, sizes=sizes)


############################################
# Angular Resolution Measure (ARM)
############################################


def arm(
    energies: torch.Tensor,
    positions: torch.Tensor,
    cfg: omegaconf.dictconfig.DictConfig,
    reconstructed_unit_vector,
    sizes: torch.Tensor | None = None,
) -> torch.Tensor:

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
            if size_int < 2:
                outputs.append(positions.new_tensor(float("nan")).reshape(1))
                continue
            outputs.append(
                arm(
                    energies[group, :size_int],
                    positions[group, :size_int, :],
                    cfg,
                    reconstructed_unit_vector,
                )
            )

        stacked = torch.stack(outputs).flatten()
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return positions.new_tensor(float("nan")).reshape(1)
        return finite.amin().reshape(1)

    n_pos = positions.shape[0] if positions.ndim == 2 else positions.shape[1]
    if n_pos < 2:
        return positions.new_tensor(float("nan")).reshape(1)

    arm_value = torch.min(
        torch.abs(theta_geo(positions, cfg, reconstructed_unit_vector)[0] - theta_kin(energies, cfg)[0])
    )

    return arm_value.reshape(1)


############################################
# Energy
############################################


def delta_E(
    energies: torch.Tensor, ref_energy: float = 511.0, sizes: torch.Tensor | None = None
) -> torch.Tensor:
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
