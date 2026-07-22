import argparse
import os
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv

from modeling.metrics import metrics
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.datasets import Datasets
from pipeline.eval import Evaluator, prior_free_scores
from pipeline.priors import (
    apply_prior_counts,
    count_prior_events,
    validate_prior_counts,
    validate_prior_sims,
)
from pipeline.train import Trainer, flatten_config
from utils.local_results import (
    save_confusion_matrix,
    save_density_terms,
    save_score_curves,
    save_sky_backprojection,
    write_metrics_json,
)
from utils.reader_extraction import load_geometry, open_sim_reader
from utils.wandb_logging import log_density_terms, log_score_curves, log_sky_backprojection

PROJECT = "cosi-betadecay"
RESULTS_DIR = Path("results")


def main(
    geo_file: str,
    sim_file: str,
    tra_file: str,
    use_wandb: bool,
    prior_sims: list[str],
) -> None:
    """Run the β⁺ annihilation detection pipeline on a given MEGAlib dataset.

    Args:
        geo_file (str): Path to the MEGAlib geometry setup file.
        sim_file (str): Path to the MEGAlib .sim file containing simulated events.
        tra_file (str): Path to the MEGAlib .tra file containing Compton events.
        use_wandb (bool): Whether to use Weights & Biases for logging.
        prior_sims (list): Dedicated prior ``.sim`` files the per-bucket class
            priors are counted from (see :mod:`pipeline.priors`); the training
            split is never used for the prior.

    Raises:
        ValueError: If any of the input files have the wrong extension.
        FileNotFoundError: If any of the input files do not exist.
    """
    sim_path, tra_path = Path(sim_file), Path(tra_file)
    if sim_path.suffix != ".sim":
        raise ValueError(f"--sim-file must be a .sim file, got: {sim_file}")
    if tra_path.suffix != ".tra":
        raise ValueError(f"--tra-file must be a .tra file, got: {tra_file}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Simulation file not found: {sim_file}")
    if not tra_path.is_file():
        raise FileNotFoundError(f"Tra file not found: {tra_file}")
    validate_prior_sims(prior_sims)

    # Count the class priors from the dedicated prior files before the pipeline
    # begins: a bad prior file fails fast, before the backprojection and
    # feature extraction.
    prior_counts = count_prior_events(geo_file, prior_sims)
    validate_prior_counts(prior_counts)

    if use_wandb:
        wandb.init(project=PROJECT)

    imager = FarFieldImager(
        geometry_file=geo_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(tra_file)
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = open_sim_reader(load_geometry(geo_file), sim_file)

    datasets = Datasets(reader_sim, unit_vector)
    data, event_keys = datasets.compute_event_features_keyed()
    train, eval_data = datasets.split_dataset(data)

    # Fixed reference config (no hyperparameter search or calibration): fit on
    # the full training set.
    trainer = Trainer()
    flat = flatten_config(trainer.config())
    selection: dict[str, object] = dict(flat)
    if use_wandb:
        wandb.log(selection)

    trainer.fit(train)

    # The prior counted upfront replaces the training-split counts.
    apply_prior_counts(trainer, prior_counts)
    selection["prior_sim_files"] = list(prior_sims)
    selection["prior_counts"] = prior_counts

    # Per-run results folder: results/<timestamp>/<dataset>/, created up front so
    # results/ always exists even if a step writes nothing. Density terms are
    # model-level, so log + save them once.
    name = Path(sim_file).stem
    out_dir = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_density_terms(trainer)
    save_density_terms(trainer, out_dir)

    evaluator = Evaluator(trainer)
    split_metrics = {}
    for split_name, split_data in (("train", train), ("eval", eval_data)):
        counts = evaluator.evaluate(split_data, split_name=split_name)
        # Pool the prior-free scores once per split; metrics, figures, and wandb
        # curves all consume the same tensors.
        pooled = evaluator.pooled_log_ratio(split_data)
        split_metrics[split_name] = {**counts, **metrics(counts), **prior_free_scores(pooled)}
        save_confusion_matrix(counts, out_dir, split_name)
        save_score_curves(pooled, out_dir, split_name)
        log_score_curves(pooled, split_name)

    # Sky-image comparison at the deployed operating point: tag every scorable
    # event, then backproject the tagged cones next to the all-events image. If
    # the classifier works, the β⁺ panel concentrates at the source and the
    # background panel shows no hotspot there.
    tagged = evaluator.classify_events(data, event_keys)
    sky_images, sky_counts = imager.backproject_file_grouped(tra_file, tagged)
    sky_panels = [
        ("All events", imager.image_2d(), n_events),
        ("β⁺-tagged", sky_images["bdecay"], sky_counts["bdecay"]),
        ("Background-tagged", sky_images["bg"], sky_counts["bg"]),
    ]
    save_sky_backprojection(sky_panels, out_dir, imager.phi_range, imager.theta_range)
    log_sky_backprojection(sky_panels, imager.phi_range, imager.theta_range)

    # One metrics.json per dataset, both splits, with the chosen config.
    write_metrics_json(
        out_dir,
        {
            "dataset": name,
            "geo_file": geo_file,
            "sim_file": sim_file,
            "tra_file": tra_file,
            "selection": selection,
            "train": split_metrics["train"],
            "eval": split_metrics["eval"],
        },
    )

    if use_wandb:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: geo/sim/tra paths, ``--prior-sim`` files, ``--wandb``.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run β⁺ annihilation detection on one .sim/.tra dataset.")
    parser.add_argument(
        "--geo-file",
        default="$MEGALIB/resource/examples/geomega/special/SPILike.geo.setup",
        help="MEGAlib geometry setup file",
    )
    parser.add_argument("--sim-file", required=True, help="MEGAlib .sim file (simulated events)")
    parser.add_argument("--tra-file", required=True, help="MEGAlib .tra file (Compton events)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--prior-sim",
        nargs="+",
        required=True,
        metavar="SIM",
        help="One or more dedicated prior .sim files; the per-bucket class priors are counted "
        "from these files (the training split is never used for the prior)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.wandb:
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)
    main(
        args.geo_file,
        args.sim_file,
        args.tra_file,
        args.wandb,
        args.prior_sim,
    )
