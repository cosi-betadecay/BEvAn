import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def test_annihilation_cosine_deviance(real_tensors, wandb_run):
    """Compare how close the reconstructed annihilation cosine lands to the
    ideal 180° (cos = -1) for true β⁺ events vs. background events.

    ``annihilation_angle`` returns the min cosine across all hit-pair
    difference vectors per event — the closer to -1, the more back-to-back
    the topology. True β⁺ annihilations should on average sit closer to
    -1 than unrelated bg topologies.
    """
    bdecay_angle = real_tensors["bdecay_angle"]
    bg_angle = real_tensors["bg_angle"]

    # annihilation_angle is NaN for events with < 3 hits (no second pair-
    # vector to form a cosine with), so a large fraction of the sim drops out
    # here. Surface those counts so the truncation is visible in W&B.
    bdecay_total = int(bdecay_angle.numel())
    bg_total = int(bg_angle.numel())
    bdecay_nan = int((~torch.isfinite(bdecay_angle)).sum().item())
    bg_nan = int((~torch.isfinite(bg_angle)).sum().item())

    bdecay_finite = bdecay_angle[torch.isfinite(bdecay_angle)].cpu().numpy().astype("float64")
    bg_finite = bg_angle[torch.isfinite(bg_angle)].cpu().numpy().astype("float64")

    assert bdecay_finite.size > 0, "no finite true-event cosines available"
    assert bg_finite.size > 0, "no finite false-event cosines available"

    # Deviance from the ideal back-to-back cosine of -1.
    bdecay_dev = np.abs(bdecay_finite + 1.0)
    bg_dev = np.abs(bg_finite + 1.0)

    mean_true = float(bdecay_dev.mean())
    mean_false = float(bg_dev.mean())

    assert mean_true < mean_false, (
        f"True events' mean deviance from 180° ({mean_true:.4f}) should be below "
        f"false events' mean deviance ({mean_false:.4f})"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [f"true (n={bdecay_finite.size})", f"false (n={bg_finite.size})"]
    values = [mean_true, mean_false]
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("mean |cos(θ) − (−1)|")
    ax.set_title("Annihilation reconstruction — mean deviance from 180°")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.4f}",
            ha="center",
            va="bottom",
        )

    wandb_run.log(
        {
            "annihilation_deviance/mean_true": mean_true,
            "annihilation_deviance/mean_false": mean_false,
            "annihilation_deviance/bars": wandb.Image(fig),
            "annihilation_deviance/n_true_finite": int(bdecay_finite.size),
            "annihilation_deviance/n_false_finite": int(bg_finite.size),
            "annihilation_deviance/n_true_nan": bdecay_nan,
            "annihilation_deviance/n_false_nan": bg_nan,
            "annihilation_deviance/n_true_total": bdecay_total,
            "annihilation_deviance/n_false_total": bg_total,
            "annihilation_deviance/frac_true_kept": bdecay_finite.size / max(bdecay_total, 1),
            "annihilation_deviance/frac_false_kept": bg_finite.size / max(bg_total, 1),
        }
    )
    plt.close(fig)
