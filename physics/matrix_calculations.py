# delta E (1) and 180 degrees flying way (2) if 511 how close the compton cone (3)
import omegaconf
import torch

from mathematics.geometry import theta_geo, theta_kin


def delta_E_and_annihilation_angle(
    energies: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    def delta_E(energies: torch.Tensor, ref_energy: float = 511) -> torch.Tensor:
        E = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
        delta_E = torch.abs(E - ref_energy)
        return delta_E

    def annihilation_angle(positions: torch.Tensor):
        def all_vector_cosines(positions: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
            if positions.ndim == 2:
                positions = positions.unsqueeze(0)
                unbatched = True
            elif positions.ndim == 3:
                unbatched = False
            else:
                raise ValueError(
                    f"Expected positions shape (N, 3) or (B, N, 3), got {tuple(positions.shape)}"
                )

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
            cosines = cosine_matrix[:, a, b]  # (B, K), K = M choose 2

            return cosines.squeeze(0) if unbatched else cosines

        cosines = all_vector_cosines(positions)
        if cosines.numel() == 0:
            return False

        angle_diffs = torch.abs(cosines + 1.0)
        if cosines.ndim == 1:
            annihilation_angle = cosines[torch.argmin(angle_diffs)]
        else:
            annihilation_angle = torch.gather(
                cosines, 1, torch.argmin(angle_diffs, dim=1, keepdim=True)
            ).squeeze(1)

        return annihilation_angle

    return delta_E(energies), annihilation_angle(positions)


def delta_E_and_compton_cone(
    energies: torch.Tensor, positions: torch.Tensor, cfg: omegaconf.dictconfig.DictConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    def delta_E(energies: torch.Tensor, ref_energy: float = 511) -> torch.Tensor:
        E = energies.sum() if energies.ndim == 1 else energies.sum(dim=1)
        delta_E = torch.abs(E - ref_energy)
        return delta_E

    arm = (
        theta_geo(positions, cfg)[0] - theta_kin(energies, cfg)[0]
    )  # not sure if this works as we want when thinking of MAP

    return delta_E(energies), arm
