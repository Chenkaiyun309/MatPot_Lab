#!/usr/bin/env python3
"""Batch-relax CIF/POSCAR structures with CHGNet.

Examples
--------
Relax all generated structures in a directory:
    python3 modules/relax_structures.py \
        --input-dir calculator_runs/generated_structures \
        --output-dir calculator_runs/relaxed_results

Use recursive search and relax the cell:
    python3 modules/relax_structures.py \
        --input-dir some_structures \
        --output-dir Run_relax/relaxed_results \
        --recursive \
        --relax-cell
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

from model_loader import load_chgnet_model


STRUCTURE_SUFFIXES = {".cif", ".vasp", ".poscar", ".POSCAR"}
EV_A3_TO_GPA = 160.21766208
SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent

# =======================
# CONFIG（用户优先修改这里）
# =======================
DEFAULT_INPUT_DIR = APP_ROOT / "calculator_runs" / "generated_structures"
DEFAULT_OUTPUT_DIR = APP_ROOT / "calculator_runs" / "relaxed_results"
DEFAULT_RECURSIVE = False
DEFAULT_INPUT_FORMAT = "poscar"  # "poscar", "cif", or "auto"
DEFAULT_EDIFFG = -0.005  # VASP-like: negative means force criterion in eV/Angstrom
DEFAULT_NSW = 2000
DEFAULT_ISIF = 3  # 2: ions only; 3 are treated as ions + cell
DEFAULT_ASE_FILTER = "FrechetCellFilter"
DEFAULT_LOGINTERVAL = 1
DEFAULT_VERBOSE = False
DEFAULT_SUMMARY = None  # e.g. Path("Run_relax/relaxed_results/relax_summary.csv")
# =======================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relax all CIF/POSCAR structures in a directory with CHGNet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing structure files (.cif, POSCAR, *.POSCAR, *.vasp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where relaxed structures and the CSV summary will be saved.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=DEFAULT_RECURSIVE,
        help="Recursively scan the input directory for structure files.",
    )
    parser.add_argument(
        "--input-format",
        choices=("poscar", "cif", "auto"),
        default=DEFAULT_INPUT_FORMAT,
        help="Read only POSCAR-like files, only CIF files, or auto-select one file per structure id.",
    )
    parser.add_argument(
        "--ediffg",
        type=float,
        default=DEFAULT_EDIFFG,
        help="VASP-like EDIFFG. Negative values mean force threshold in eV/Angstrom.",
    )
    parser.add_argument(
        "--nsw",
        type=int,
        default=DEFAULT_NSW,
        help="VASP-like NSW: maximum ionic steps per structure.",
    )
    parser.add_argument(
        "--isif",
        type=int,
        default=DEFAULT_ISIF,
        help="VASP-like ISIF. 2 means ions only; 3/4/5/6/7 relax the cell in this script.",
    )
    parser.add_argument(
        "--ase-filter",
        type=str,
        default=DEFAULT_ASE_FILTER,
        help="ASE cell filter class name used when --relax-cell is enabled.",
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=DEFAULT_LOGINTERVAL,
        help="Record relaxation trajectory every N optimizer steps.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=DEFAULT_VERBOSE,
        help="Enable verbose CHGNet relaxation output.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Optional CSV summary output path. Default: <output-dir>/relax_summary.csv",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="CHGNet model: builtin name (0.3.0/0.2.0/r2scan) or a .pth.tar filename/path in the models/ folder. Default: 0.3.0.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--relax-cell",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser


def require_pymatgen():
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'pymatgen'. Install it in the environment that runs this script."
        ) from exc
    return Structure, AseAtomsAdaptor


def require_ase():
    try:
        from ase.filters import FrechetCellFilter
        from ase.io.trajectory import Trajectory
        from ase.optimize import FIRE
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'ase'. Install ASE in the environment that runs this script."
        ) from exc
    return FIRE, Trajectory, FrechetCellFilter


def require_chgnet():
    try:
        from chgnet.model.model import CHGNet
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'chgnet'. Install CHGNet in the environment that runs this script."
        ) from exc

    try:
        from chgnet.model.dynamics import CHGNetCalculator
    except Exception:
        try:
            from chgnet.model import CHGNetCalculator
        except Exception as exc:
            raise SystemExit(
                "Unable to import CHGNetCalculator from your CHGNet installation."
            ) from exc

    return CHGNet, CHGNetCalculator


def make_calculator(model, calculator_cls):
    for kwargs in ({"model": model}, {"chgnet": model}, {}):
        try:
            return calculator_cls(**kwargs)
        except TypeError:
            continue
    raise SystemExit("Unable to instantiate CHGNetCalculator with the installed CHGNet version.")


def is_poscar_like(path: Path) -> bool:
    return path.name == "POSCAR" or path.suffix.lower() in {".vasp", ".poscar"}


def is_structure_file(path: Path, input_format: str) -> bool:
    if not path.is_file():
        return False
    if input_format == "cif":
        return path.suffix.lower() == ".cif"
    if input_format == "poscar":
        return is_poscar_like(path)
    if path.name == "POSCAR":
        return True
    return path.suffix in STRUCTURE_SUFFIXES


def structure_identifier(path: Path) -> str:
    if path.name == "POSCAR":
        return path.parent.name or "POSCAR"
    return path.stem


def format_priority(path: Path, input_format: str) -> tuple[int, str]:
    is_poscar = is_poscar_like(path)
    if input_format == "poscar":
        score = 0 if is_poscar else 1
    elif input_format == "cif":
        score = 0 if path.suffix.lower() == ".cif" else 1
    else:
        score = 0 if is_poscar else 1
    return score, path.name


def discover_structures(input_dir: Path, recursive: bool, input_format: str) -> list[Path]:
    walker = input_dir.rglob("*") if recursive else input_dir.iterdir()
    grouped: dict[str, list[Path]] = {}
    for path in walker:
        if is_structure_file(path, input_format):
            grouped.setdefault(structure_identifier(path), []).append(path)

    selected = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda path: format_priority(path, input_format))
        selected.append(candidates[0])
    return selected


def make_filter(filter_name: str, atoms):
    try:
        ase_filters = importlib.import_module("ase.filters")
        filter_cls = getattr(ase_filters, filter_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise SystemExit(f"Unable to import ASE filter '{filter_name}'.") from exc
    return filter_cls(atoms)


def record_relaxation_step(rows: list[dict[str, str | int]], atoms, step: int) -> None:
    stress = atoms.get_stress(voigt=True) * EV_A3_TO_GPA
    rows.append(
        {
            "step": step,
            "energy_eV": f"{float(atoms.get_potential_energy()):.10f}",
            "volume_A3": f"{float(atoms.get_volume()):.10f}",
            "stress_xx_GPa": f"{float(stress[0]):.10f}",
            "stress_yy_GPa": f"{float(stress[1]):.10f}",
            "stress_zz_GPa": f"{float(stress[2]):.10f}",
            "stress_yz_GPa": f"{float(stress[3]):.10f}",
            "stress_xz_GPa": f"{float(stress[4]):.10f}",
            "stress_xy_GPa": f"{float(stress[5]):.10f}",
        }
    )


def write_step_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "step",
        "energy_eV",
        "volume_A3",
        "stress_xx_GPa",
        "stress_yy_GPa",
        "stress_zz_GPa",
        "stress_yz_GPa",
        "stress_xz_GPa",
        "stress_xy_GPa",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_relaxation_settings(args: argparse.Namespace) -> tuple[float, int, bool]:
    if args.fmax is not None:
        fmax = args.fmax
    else:
        if args.ediffg == 0:
            raise SystemExit("EDIFFG must be non-zero.")
        fmax = abs(args.ediffg)

    if fmax <= 0:
        raise SystemExit("Force threshold must be positive.")

    max_steps = args.max_steps if args.max_steps is not None else args.nsw
    if max_steps <= 0:
        raise SystemExit("NSW/max_steps must be a positive integer.")

    if args.relax_cell:
        relax_cell = True
    else:
        if args.isif < 0:
            raise SystemExit("ISIF must be non-negative.")
        relax_cell = args.isif >= 3

    return fmax, max_steps, relax_cell


def relax_structure(
    structure_path: Path,
    output_dir: Path,
    model,
    calculator_cls,
    structure_cls,
    adaptor_cls,
    fire_cls,
    trajectory_cls,
    fmax: float,
    max_steps: int,
    relax_cell: bool,
    ase_filter: str,
    loginterval: int,
    verbose: bool,
) -> dict[str, str]:
    structure_id = structure_identifier(structure_path)
    result_dir = output_dir / structure_id
    result_dir.mkdir(parents=True, exist_ok=True)

    structure = structure_cls.from_file(str(structure_path))
    atoms = adaptor_cls.get_atoms(structure)
    atoms.calc = make_calculator(model, calculator_cls)
    optimizer_target = make_filter(ase_filter, atoms) if relax_cell else atoms

    trajectory_path = result_dir / "relaxation.traj"
    step_csv_path = result_dir / "relaxation_steps.csv"
    optimizer_log_path = result_dir / "optimizer.log"
    input_copy_path = result_dir / f"{structure_path.name}"
    poscar_path = result_dir / f"{structure_id}_relaxed.POSCAR"

    structure.to(filename=str(input_copy_path))

    step_rows: list[dict[str, str | int]] = []
    step_index = {"value": 0}
    trajectory = trajectory_cls(str(trajectory_path), "w", atoms)

    def capture_step() -> None:
        record_relaxation_step(step_rows, atoms, step_index["value"])
        trajectory.write(atoms)
        step_index["value"] += 1

    capture_step()
    optimizer = fire_cls(optimizer_target, logfile=str(optimizer_log_path))
    optimizer.attach(capture_step, interval=loginterval)
    try:
        optimizer.run(fmax=fmax, steps=max_steps)
    finally:
        trajectory.close()
        write_step_csv(step_csv_path, step_rows)

    relaxed_structure = adaptor_cls.get_structure(atoms)
    relaxed_structure.to(fmt="poscar", filename=str(poscar_path))

    energy = step_rows[-1]["energy_eV"] if step_rows else ""
    num_sites = len(relaxed_structure)
    volume = f"{float(relaxed_structure.volume):.10f}"
    energy_per_atom = ""
    if energy:
        energy_per_atom = f"{float(energy) / num_sites:.10f}"

    return {
        "id": structure_id,
        "input_file": str(structure_path),
        "status": "ok",
        "num_recorded_steps": str(len(step_rows)),
        "num_sites": str(num_sites),
        "formula": relaxed_structure.composition.formula.replace(" ", ""),
        "energy_eV": energy,
        "energy_per_atom_eV": energy_per_atom,
        "volume_A3": volume,
        "poscar_output": str(poscar_path),
        "step_csv": str(step_csv_path),
        "trajectory_file": str(trajectory_path),
        "optimizer_log": str(optimizer_log_path),
    }


def write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "input_file",
        "status",
        "num_recorded_steps",
        "num_sites",
        "formula",
        "energy_eV",
        "energy_per_atom_eV",
        "volume_A3",
        "poscar_output",
        "step_csv",
        "trajectory_file",
        "optimizer_log",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    summary_path = args.summary.expanduser() if args.summary else output_dir / "relax_summary.csv"

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if args.loginterval <= 0:
        raise SystemExit("--loginterval must be a positive integer.")
    fmax, max_steps, relax_cell = resolve_relaxation_settings(args)

    structure_cls, adaptor_cls = require_pymatgen()
    fire_cls, trajectory_cls, _ = require_ase()
    chgnet_cls, calculator_cls = require_chgnet()
    structures = discover_structures(input_dir, recursive=args.recursive, input_format=args.input_format)
    if not structures:
        raise SystemExit(f"No structure files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_chgnet_model(chgnet_cls, model_arg=args.model)
    summary_rows: list[dict[str, str]] = []

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Structures found: {len(structures)}")
    print(f"EDIFFG/fmax: {fmax}")
    print(f"NSW/max_steps: {max_steps}")
    print(f"ISIF/relax_cell: {args.isif} / {relax_cell}")

    for index, structure_path in enumerate(structures, start=1):
        structure_id = structure_identifier(structure_path)
        print(f"[{index:03d}/{len(structures):03d}] Relaxing {structure_id} from {structure_path.name}")
        try:
            row = relax_structure(
                structure_path=structure_path,
                output_dir=output_dir,
                model=model,
                calculator_cls=calculator_cls,
                structure_cls=structure_cls,
                adaptor_cls=adaptor_cls,
                fire_cls=fire_cls,
                trajectory_cls=trajectory_cls,
                fmax=fmax,
                max_steps=max_steps,
                relax_cell=relax_cell,
                ase_filter=args.ase_filter,
                loginterval=args.loginterval,
                verbose=args.verbose,
            )
            print(
                f"  done: energy={row['energy_eV'] or 'n/a'} eV, "
                f"volume={row['volume_A3']} A^3, "
                f"steps={row['num_recorded_steps']}"
            )
        except Exception as exc:
            row = {
                "id": structure_id,
                "input_file": str(structure_path),
                "status": "failed",
                "error": str(exc),
            }
            print(f"  failed: {exc}")
        summary_rows.append(row)

    write_summary(summary_path, summary_rows)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
