import argparse
from datetime import datetime
from pathlib import Path

import torch

from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.datasets import BUCKETS, CLASSES, Datasets
from pipeline.eval import Evaluator
from pipeline.train import Trainer
from utils.reader_extraction import load_geometry, open_sim_reader

PREDICTIONS_DIR = Path("predictions")


def run_inference(
    model_file: str,
    geo_file: str,
    sim_file: str,
    tra_file: str,
    out_file: str,
) -> torch.Tensor:
    """Load a saved model and classify every event of a (separate) simulation.

    The model artifact carries the fitted densities and the counted-prior
    operating point, but ARM is measured against the source direction, which is specific to
    this dataset — so the ``.tra`` is still back-projected here to reconstruct it
    (``delta_E`` and the annihilation angle need nothing external).

    The result is a per-event classification vector saved to ``out_file`` with
    :func:`torch.save`: one ``uint8`` entry per non-empty event of the ``.sim``
    in file order, ``1`` = β⁺ and ``0`` = background. Events the model cannot
    score (no finite features) are ``0``; zero-hit events are skipped by feature
    extraction and get no entry.

    Args:
        model_file: Path to a ``.pt`` artifact written by :mod:`train_model`.
        geo_file: MEGAlib geometry setup file for this dataset.
        sim_file: MEGAlib ``.sim`` file (simulated events) to score.
        tra_file: MEGAlib ``.tra`` file (Compton events) for source reconstruction.
        out_file: Destination path for the saved classification tensor.

    Returns:
        The per-event classification tensor that was saved.

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
    imager.backproject_file(tra_file)
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = open_sim_reader(load_geometry(geo_file), sim_file)

    datasets = Datasets(reader_sim, unit_vector)
    data, event_keys = datasets.compute_event_features_keyed()

    evaluator = Evaluator(trainer)
    tagged = evaluator.classify_events(data, event_keys)

    # Sorting the (chunk, id) keys recovers file order: chunks are visited in
    # order and IDs increase strictly within a chunk (see ChunkScopedIds).
    all_keys = sorted(key for cls in CLASSES for b in BUCKETS for key in event_keys[cls][b])
    classifications = torch.tensor(
        [1 if key in tagged["bdecay"] else 0 for key in all_keys], dtype=torch.uint8
    )

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifications, out_path)
    return classifications


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: the model, geo/sim/tra paths, and the output path.

    Returns:
        The parsed arguments.
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
        help="Output path for the per-event classification tensor "
        "(default: predictions/<sim-stem>_<timestamp>.pt)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = args.out or str(PREDICTIONS_DIR / f"{Path(args.sim_file).stem}_{timestamp}.pt")
    run_inference(args.model, args.geo_file, args.sim_file, args.tra_file, out_path)
