#!/usr/bin/env python3
"""Compute CHGNet stress-strain curves for CIF/POSCAR structures.

Examples
--------
Run a uniaxial xx stress-strain scan for generated structures:
    python3 modules/stress_strain_curve.py \
        --input-dir calculator_runs/generated_structures \
        --output-dir calculator_runs/stress_strain_results \
        --direction xx \
        --strain-start -0.02 \
        --strain-end 0.10 \
        --num-points 13

Run a shear scan and relax atomic positions at each strain point:
    python3 modules/stress_strain_curve.py \
        --input-dir calculator_runs/relaxed_results \
        --recursive \
        --direction xy \
        --strain-start 0.00 \
        --strain-end 0.12 \
        --num-points 7 \
        --relax-atoms \
        --fmax 0.03 \
        --max-steps 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from relax_structures import (
    EV_A3_TO_GPA,
    discover_structures,
    make_calculator,
    require_ase,
    require_chgnet,
    require_pymatgen,
    structure_identifier,
)
from model_loader import load_chgnet_model


SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = APP_ROOT / "calculator_runs" / "generated_structures"
DEFAULT_OUTPUT_DIR = APP_ROOT / "calculator_runs" / "stress_strain_results"
DEFAULT_RECURSIVE = False
DEFAULT_INPUT_FORMAT = "poscar"
DEFAULT_DIRECTION = "xx"
DEFAULT_STRAIN_START = -0.02
DEFAULT_STRAIN_END = 0.10
DEFAULT_NUM_POINTS = 13
DEFAULT_RELAX_ATOMS = False
DEFAULT_FMAX = 0.03
DEFAULT_MAX_STEPS = 200
DEFAULT_WRITE_STRUCTURES = True
DEFAULT_PLOT = True
DEFAULT_SUMMARY = None

NORMAL_DIRECTIONS = {"xx", "yy", "zz"}
SHEAR_DIRECTIONS = {"xy", "xz", "yz"}
STRESS_COMPONENT_INDEX = {
    "xx": 0,
    "yy": 1,
    "zz": 2,
    "yz": 3,
    "xz": 4,
    "xy": 5,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute CHGNet stress-strain curves for all CIF/POSCAR structures in a directory."
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
        help="Directory where curve CSV files, plots, and the summary CSV will be saved.",
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
        "--direction",
        choices=("xx", "yy", "zz", "xy", "xz", "yz"),
        default=DEFAULT_DIRECTION,
        help="Applied engineering strain direction. Normal directions are uniaxial; off-diagonal directions are simple shear.",
    )
    parser.add_argument(
        "--strain-start",
        type=float,
        default=DEFAULT_STRAIN_START,
        help="Starting engineering strain value.",
    )
    parser.add_argument(
        "--strain-end",
        type=float,
        default=DEFAULT_STRAIN_END,
        help="Ending engineering strain value.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="Number of strain points between start and end, inclusive.",
    )
    parser.add_argument(
        "--relax-atoms",
        action="store_true",
        default=DEFAULT_RELAX_ATOMS,
        help="Relax atomic positions with FIRE at each strain point while keeping the strained cell fixed.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=DEFAULT_FMAX,
        help="Force threshold used when --relax-atoms is enabled.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum FIRE steps per strain point when --relax-atoms is enabled.",
    )
    parser.add_argument(
        "--write-structures",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WRITE_STRUCTURES,
        help="Write a POSCAR file for every strain point.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PLOT,
        help="Generate a PNG stress-strain curve for each structure.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Optional CSV summary output path. Default: <output-dir>/stress_strain_summary.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="CHGNet model: builtin name (0.3.0/0.2.0/r2scan) or a .pth.tar filename/path in the models/ folder. Default: 0.3.0.",
    )
    return parser


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'matplotlib'. Install matplotlib or run with --no-plot."
        ) from exc
    return plt


def validate_args(args: argparse.Namespace) -> None:
    if args.num_points <= 0:
        raise SystemExit("--num-points must be a positive integer.")
    if args.relax_atoms:
        if args.fmax <= 0:
            raise SystemExit("--fmax must be positive when --relax-atoms is enabled.")
        if args.max_steps <= 0:
            raise SystemExit("--max-steps must be a positive integer when --relax-atoms is enabled.")
    if args.direction in NORMAL_DIRECTIONS:
        if min(args.strain_start, args.strain_end) <= -1.0:
            raise SystemExit("Normal engineering strain must stay above -1.0.")


def build_strain_values(start: float, end: float, num_points: int) -> list[float]:
    if num_points == 1:
        return [float(start)]
    return [float(value) for value in np.linspace(start, end, num_points)]


def strain_mode(direction: str) -> str:
    return "uniaxial" if direction in NORMAL_DIRECTIONS else "simple_shear"


def deformation_gradient(direction: str, strain_value: float) -> np.ndarray:
    gradient = np.eye(3)
    if direction == "xx":
        gradient[0, 0] = 1.0 + strain_value
    elif direction == "yy":
        gradient[1, 1] = 1.0 + strain_value
    elif direction == "zz":
        gradient[2, 2] = 1.0 + strain_value
    elif direction == "xy":
        gradient[0, 1] = strain_value
    elif direction == "xz":
        gradient[0, 2] = strain_value
    elif direction == "yz":
        gradient[1, 2] = strain_value
    else:  # pragma: no cover - parser choices prevent this
        raise ValueError(f"Unsupported direction: {direction}")
    return gradient


def strain_token(strain_value: float) -> str:
    prefix = "p" if strain_value >= 0 else "m"
    return f"{prefix}{abs(strain_value):.5f}".replace(".", "p")


def apply_strain(base_atoms, direction: str, strain_value: float):
    atoms = base_atoms.copy()
    base_cell = np.array(base_atoms.cell)
    new_cell = base_cell @ deformation_gradient(direction, strain_value).T
    atoms.set_cell(new_cell, scale_atoms=True)
    return atoms


def stress_row(atoms, direction: str) -> dict[str, str]:
    stress = atoms.get_stress(voigt=True) * EV_A3_TO_GPA
    return {
        "stress_xx_GPa": f"{float(stress[0]):.10f}",
        "stress_yy_GPa": f"{float(stress[1]):.10f}",
        "stress_zz_GPa": f"{float(stress[2]):.10f}",
        "stress_yz_GPa": f"{float(stress[3]):.10f}",
        "stress_xz_GPa": f"{float(stress[4]):.10f}",
        "stress_xy_GPa": f"{float(stress[5]):.10f}",
        "target_stress_GPa": f"{float(stress[STRESS_COMPONENT_INDEX[direction]]):.10f}",
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_curve(rows: list[dict[str, str]], output_path: Path, structure_id: str, direction: str, strain_mode_label: str, plt) -> None:
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if not valid_rows:
        return

    strains = [float(row["strain_value"]) for row in valid_rows]
    stresses = [float(row["target_stress_GPa"]) for row in valid_rows]

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    axis.plot(strains, stresses, marker="o", linewidth=1.8, markersize=4.8, color="#0f766e")
    axis.axhline(0.0, color="#94a3b8", linewidth=0.9)
    axis.axvline(0.0, color="#94a3b8", linewidth=0.9)
    axis.set_xlabel("Engineering strain")
    axis.set_ylabel(f"{direction} stress (GPa)")
    axis.set_title(f"{structure_id} ({strain_mode_label})")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_fire_relaxation(atoms, fire_cls, fmax: float, max_steps: int, logfile: Path) -> int:
    optimizer = fire_cls(atoms, logfile=str(logfile))
    optimizer.run(fmax=fmax, steps=max_steps)
    return int(getattr(optimizer, "nsteps", 0))


def curve_fieldnames() -> list[str]:
    return [
        "point_index",
        "direction",
        "strain_mode",
        "strain_value",
        "energy_eV",
        "energy_per_atom_eV",
        "volume_A3",
        "stress_xx_GPa",
        "stress_yy_GPa",
        "stress_zz_GPa",
        "stress_yz_GPa",
        "stress_xz_GPa",
        "stress_xy_GPa",
        "target_stress_GPa",
        "num_relax_steps",
        "output_poscar",
        "optimizer_log",
        "status",
        "error",
    ]


def summary_fieldnames() -> list[str]:
    return [
        "id",
        "input_file",
        "status",
        "num_sites",
        "formula",
        "direction",
        "strain_mode",
        "num_points",
        "num_successful_points",
        "strain_start",
        "strain_end",
        "curve_csv",
        "plot_file",
        "peak_stress_GPa",
        "peak_stress_strain",
        "error",
    ]


def evaluate_structure(
    structure_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    strain_values: list[float],
    model,
    calculator_cls,
    structure_cls,
    adaptor_cls,
    fire_cls,
    plt,
) -> dict[str, str]:
    structure = structure_cls.from_file(str(structure_path))
    base_atoms = adaptor_cls.get_atoms(structure)
    structure_id = structure_identifier(structure_path)
    result_dir = output_dir / structure_id
    curve_csv_path = result_dir / "stress_strain_curve.csv"
    plot_path = result_dir / "stress_strain_curve.png"
    structure_dir = result_dir / "strained_structures"
    logs_dir = result_dir / "optimizer_logs"
    result_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict[str, str]] = []
    mode_label = strain_mode(args.direction)

    for point_index, strain_value in enumerate(strain_values, start=1):
        row: dict[str, str] = {
            "point_index": str(point_index),
            "direction": args.direction,
            "strain_mode": mode_label,
            "strain_value": f"{strain_value:.10f}",
            "num_relax_steps": "0",
            "output_poscar": "",
            "optimizer_log": "",
            "status": "ok",
            "error": "",
        }
        try:
            atoms = apply_strain(base_atoms, args.direction, strain_value)
            atoms.calc = make_calculator(model, calculator_cls)

            if args.relax_atoms:
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_dir / f"{point_index:03d}_{args.direction}_{strain_token(strain_value)}.log"
                row["optimizer_log"] = str(log_path)
                row["num_relax_steps"] = str(
                    run_fire_relaxation(atoms, fire_cls, args.fmax, args.max_steps, log_path)
                )

            energy = float(atoms.get_potential_energy())
            num_sites = len(atoms)
            row["energy_eV"] = f"{energy:.10f}"
            row["energy_per_atom_eV"] = f"{energy / num_sites:.10f}"
            row["volume_A3"] = f"{float(atoms.get_volume()):.10f}"
            row.update(stress_row(atoms, args.direction))

            if args.write_structures:
                structure_dir.mkdir(parents=True, exist_ok=True)
                poscar_path = structure_dir / f"{point_index:03d}_{args.direction}_{strain_token(strain_value)}.POSCAR"
                adaptor_cls.get_structure(atoms).to(fmt="poscar", filename=str(poscar_path))
                row["output_poscar"] = str(poscar_path)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            row.setdefault("energy_eV", "")
            row.setdefault("energy_per_atom_eV", "")
            row.setdefault("volume_A3", "")
            row.setdefault("stress_xx_GPa", "")
            row.setdefault("stress_yy_GPa", "")
            row.setdefault("stress_zz_GPa", "")
            row.setdefault("stress_yz_GPa", "")
            row.setdefault("stress_xz_GPa", "")
            row.setdefault("stress_xy_GPa", "")
            row.setdefault("target_stress_GPa", "")
        curve_rows.append(row)

    write_csv(curve_csv_path, curve_rows, curve_fieldnames())
    if args.plot:
        plot_curve(curve_rows, plot_path, structure_id, args.direction, mode_label, plt)

    success_rows = [row for row in curve_rows if row.get("status") == "ok" and row.get("target_stress_GPa")]
    if not success_rows:
        status = "failed"
        peak_row = None
    elif len(success_rows) == len(curve_rows):
        status = "ok"
        peak_row = max(success_rows, key=lambda row: abs(float(row["target_stress_GPa"])))
    else:
        status = "partial"
        peak_row = max(success_rows, key=lambda row: abs(float(row["target_stress_GPa"])))

    return {
        "id": structure_id,
        "input_file": str(structure_path),
        "status": status,
        "num_sites": str(len(structure)),
        "formula": structure.composition.formula.replace(" ", ""),
        "direction": args.direction,
        "strain_mode": mode_label,
        "num_points": str(len(curve_rows)),
        "num_successful_points": str(len(success_rows)),
        "strain_start": f"{strain_values[0]:.10f}",
        "strain_end": f"{strain_values[-1]:.10f}",
        "curve_csv": str(curve_csv_path),
        "plot_file": str(plot_path) if args.plot else "",
        "peak_stress_GPa": peak_row["target_stress_GPa"] if peak_row else "",
        "peak_stress_strain": peak_row["strain_value"] if peak_row else "",
        "error": "" if success_rows else "No successful strain points.",
    }


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    summary_path = args.summary.expanduser() if args.summary else output_dir / "stress_strain_summary.csv"

    validate_args(args)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    structures = discover_structures(input_dir, args.recursive, args.input_format)
    if not structures:
        raise SystemExit(f"No structure files found in: {input_dir}")

    strain_values = build_strain_values(args.strain_start, args.strain_end, args.num_points)
    structure_cls, adaptor_cls = require_pymatgen()
    fire_cls, _, _ = require_ase()
    chgnet_cls, calculator_cls = require_chgnet()
    plt = require_matplotlib() if args.plot else None
    model = load_chgnet_model(chgnet_cls, model_arg=args.model)

    summary_rows: list[dict[str, str]] = []
    for structure_path in structures:
        try:
            summary_rows.append(
                evaluate_structure(
                    structure_path=structure_path,
                    output_dir=output_dir,
                    args=args,
                    strain_values=strain_values,
                    model=model,
                    calculator_cls=calculator_cls,
                    structure_cls=structure_cls,
                    adaptor_cls=adaptor_cls,
                    fire_cls=fire_cls,
                    plt=plt,
                )
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "id": structure_identifier(structure_path),
                    "input_file": str(structure_path),
                    "status": "failed",
                    "num_sites": "",
                    "formula": "",
                    "direction": args.direction,
                    "strain_mode": strain_mode(args.direction),
                    "num_points": str(len(strain_values)),
                    "num_successful_points": "0",
                    "strain_start": f"{strain_values[0]:.10f}",
                    "strain_end": f"{strain_values[-1]:.10f}",
                    "curve_csv": "",
                    "plot_file": "",
                    "peak_stress_GPa": "",
                    "peak_stress_strain": "",
                    "error": str(exc),
                }
            )

    write_csv(summary_path, summary_rows, summary_fieldnames())
    print(f"Saved stress-strain summary to: {summary_path}")


if __name__ == "__main__":
    main()
