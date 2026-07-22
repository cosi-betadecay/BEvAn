import pytest
import torch


@pytest.fixture()
def separable_data() -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    """Per-bucket features with disjoint β/bg ranges so a correct fit classifies perfectly."""
    n = 200
    nan = torch.full((n,), float("nan"))

    def bucket(delta_e: torch.Tensor, arm: torch.Tensor, anni: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"delta_E": delta_e, "arm": arm, "anni": anni}

    beta_de = torch.linspace(0.01, 1.0, n)
    bg_de = torch.linspace(50.0, 100.0, n)
    return {
        "bdecay": {
            1: bucket(beta_de.clone(), nan.clone(), nan.clone()),
            2: bucket(beta_de.clone(), torch.linspace(-0.05, 0.05, n), nan.clone()),
            3: bucket(beta_de.clone(), torch.linspace(-0.05, 0.05, n), torch.linspace(-1.0, -0.9, n)),
        },
        "bg": {
            1: bucket(bg_de.clone(), nan.clone(), nan.clone()),
            2: bucket(bg_de.clone(), torch.linspace(0.4, 0.6, n), nan.clone()),
            3: bucket(bg_de.clone(), torch.linspace(0.4, 0.6, n), torch.linspace(0.5, 0.9, n)),
        },
    }
