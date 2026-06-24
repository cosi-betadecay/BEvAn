import argparse
import csv
from pathlib import Path

from dataset.datasets import Datasets
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator, metrics, prior_free_scores
from pipeline.train import Trainer
from utils.reader_extraction import get_reader


def run_inference(
    model_file: str,
    geo_file: str,
    sim_file: str,
    tra_file: str,
    out_file: str | None,
) -> dict:
    """Load a saved model and run real inference on a (separate) simulation.

    The model artifact carries the fitted densities and calibrated operating
    point, but ARM is measured against the source direction, which is specific to
    this dataset — so the ``.tra`` is still back-projected here to reconstruct it
    (``delta_E`` and the annihilation angle need nothing external). Every event is
    evaluated (no train/eval split): the held-out test set is this whole file.

    Args:
        model_file: Path to a ``.pt`` artifact written by :mod:`train_model`.
        geo_file: MEGAlib geometry setup file for this dataset.
        sim_file: MEGAlib ``.sim`` file (simulated events) to score.
        tra_file: MEGAlib ``.tra`` file (Compton events) for source reconstruction.
        out_file: Optional CSV path for the confusion counts and metrics.

    Returns:
        The pooled confusion counts plus prior-free ``best_f1``/``auc`` for the run.

    Raises:
        ValueError: If ``sim_file``/``tra_file`` have the wrong extension.
        FileNotFoundError: If the model, sim, or tra file does not exist.
    """
    model_path, sim_path, tra_path = Path(model_file), Path(sim_file), Path(tra_file)
    if sim_path.suffix != ".sim":
        raise ValueError(f"--sim-file must be a .sim file, got: {sim_file}")
    if tra_path.suffix != ".tra":
        raise ValueError(f"--tra-file must be a .tra file, got: {tra_file}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not sim_path.is_file():
        raise FileNotFoundError(f"Simulation file not found: {sim_file}")
    if not tra_path.is_file():
        raise FileNotFoundError(f"Tra file not found: {tra_file}")

    trainer = Trainer.load(model_file)

    imager = FarFieldImager(
        geometry_file=geo_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(tra_file)
    print(f"Accumulated {n_events} Compton events from {tra_file}")
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = get_reader(geo_file, sim_file)

    datasets = Datasets(reader_sim, unit_vector)
    data = datasets.compute_event_features()

    evaluator = Evaluator(trainer)
    counts = evaluator.evaluate(data, split_name="inference")
    result = {**counts, **metrics(counts), **prior_free_scores(evaluator, data)}

    if out_file is not None:
        row = {
            "model_file": model_file,
            "sim_file": sim_file,
            "tra_file": tra_file,
            "geo_file": geo_file,
            **result,
        }
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"Saved inference results to {out_path.resolve()}")

    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: the model, geo/sim/tra paths, and optional CSV output.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a saved β⁺ annihilation model on a .sim/.tra dataset (real inference)."
    )
    parser.add_argument("--model", required=True, help="Saved .pt model artifact from train_model.py")
    parser.add_argument("--geo-file", required=True, help="MEGAlib geometry setup file")
    parser.add_argument("--sim-file", required=True, help="MEGAlib .sim file (simulated events) to score")
    parser.add_argument("--tra-file", required=True, help="MEGAlib .tra file (Compton events)")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV path for the confusion counts and metrics (default: print only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model, args.geo_file, args.sim_file, args.tra_file, args.out)
