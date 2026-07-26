#!/usr/bin/env python3
"""Compute elastic constants and mechanical properties with CHGNet.

Examples
--------
Evaluate mechanical properties for generated structures:
    python3 modules/mechanical_properties.py \
        --input-dir calculator_runs/generated_structures \
        --output-dir calculator_runs/mechanical_results \
        --strain-max 0.01 \
        --num-points 5

Relax atomic positions at each strained state:
    python3 modules/mechanical_properties.py \
        --input-dir calculator_runs/relaxed_results \
        --recursive \
        --relax-atoms \
        --fmax 0.03 \
        --max-steps 200
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_OUTPUT_DIR = APP_ROOT / "calculator_runs" / "mechanical_results"
DEFAULT_RECURSIVE = False
DEFAULT_INPUT_FORMAT = "poscar"
DEFAULT_STRAIN_MAX = 0.01
DEFAULT_NUM_POINTS = 5
DEFAULT_RELAX_ATOMS = False
DEFAULT_FMAX = 0.03
DEFAULT_MAX_STEPS = 200
DEFAULT_WRITE_STRUCTURES = False
DEFAULT_PLOT = True
DEFAULT_SUMMARY = None
VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute CHGNet elastic constants and mechanical properties for CIF/POSCAR structures."
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
        help="Directory where mechanical results and the CSV summary will be saved.",
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
        "--strain-max",
        type=float,
        default=DEFAULT_STRAIN_MAX,
        help="Maximum strain amplitude used for each deformation mode.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="Number of strain points per deformation mode between -strain_max and +strain_max.",
    )
    parser.add_argument(
        "--relax-atoms",
        action="store_true",
        default=DEFAULT_RELAX_ATOMS,
        help="Relax atomic positions with FIRE at each strained state while keeping the strained cell fixed.",
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
        help="Maximum FIRE steps per strained state when --relax-atoms is enabled.",
    )
    parser.add_argument(
        "--write-structures",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WRITE_STRUCTURES,
        help="Write a POSCAR file for every strained state.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PLOT,
        help="Generate a PNG plot summarizing stress-strain fits.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Optional CSV summary output path. Default: <output-dir>/mechanical_summary.csv",
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
    if args.strain_max <= 0:
        raise SystemExit("--strain-max must be positive.")
    if args.strain_max >= 0.2:
        raise SystemExit("--strain-max should stay below 0.2 for linear elastic fitting.")
    if args.num_points < 3:
        raise SystemExit("--num-points must be at least 3.")
    if args.relax_atoms:
        if args.fmax <= 0:
            raise SystemExit("--fmax must be positive when --relax-atoms is enabled.")
        if args.max_steps <= 0:
            raise SystemExit("--max-steps must be positive when --relax-atoms is enabled.")


def build_strain_values(strain_max: float, num_points: int) -> list[float]:
    return [float(value) for value in np.linspace(-strain_max, strain_max, num_points)]


def strain_token(strain_value: float) -> str:
    prefix = "p" if strain_value >= 0 else "m"
    return f"{prefix}{abs(strain_value):.5f}".replace(".", "p")


def deformation_gradient(mode_index: int, strain_value: float) -> np.ndarray:
    gradient = np.eye(3)
    if mode_index == 0:
        gradient[0, 0] = 1.0 + strain_value
    elif mode_index == 1:
        gradient[1, 1] = 1.0 + strain_value
    elif mode_index == 2:
        gradient[2, 2] = 1.0 + strain_value
    elif mode_index == 3:
        gradient[1, 2] = strain_value
    elif mode_index == 4:
        gradient[0, 2] = strain_value
    elif mode_index == 5:
        gradient[0, 1] = strain_value
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported Voigt mode index: {mode_index}")
    return gradient


def apply_mode_strain(base_atoms, mode_index: int, strain_value: float):
    atoms = base_atoms.copy()
    base_cell = np.array(base_atoms.cell)
    new_cell = base_cell @ deformation_gradient(mode_index, strain_value).T
    atoms.set_cell(new_cell, scale_atoms=True)
    return atoms


def run_fire_relaxation(atoms, fire_cls, fmax: float, max_steps: int, logfile: Path) -> int:
    optimizer = fire_cls(atoms, logfile=str(logfile))
    optimizer.run(fmax=fmax, steps=max_steps)
    return int(getattr(optimizer, "nsteps", 0))


def response_fieldnames() -> list[str]:
    return [
        "mode_index",
        "mode_label",
        "point_index",
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
        "strain_max",
        "num_points_per_mode",
        "relax_atoms",
        "num_successful_evaluations",
        "elastic_tensor_csv",
        "response_csv",
        "properties_json",
        "plot_file",
        "bulk_voigt_GPa",
        "bulk_reuss_GPa",
        "bulk_hill_GPa",
        "shear_voigt_GPa",
        "shear_reuss_GPa",
        "shear_hill_GPa",
        "young_hill_GPa",
        "poisson_hill",
        "pugh_ratio",
        "min_eigenvalue_GPa",
        "max_asymmetry_GPa",
        "mechanically_stable",
        "error",
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component"] + [f"C{index + 1}" for index in range(6)])
        for row_index, label in enumerate(VOIGT_LABELS):
            writer.writerow([label] + [f"{float(value):.10f}" for value in matrix[row_index]])


def fit_elastic_tensor(response_rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, float]]:
    raw_tensor = np.zeros((6, 6))
    fit_meta: dict[str, float] = {}

    for mode_index in range(6):
        mode_rows = [row for row in response_rows if row.get("status") == "ok" and row.get("mode_index") == str(mode_index)]
        strains = np.array([float(row["strain_value"]) for row in mode_rows], dtype=float)
        if len(strains) < 2:
            raise ValueError(f"Not enough successful points for mode {VOIGT_LABELS[mode_index]}.")
        for stress_index, stress_key in enumerate(
            [
                "stress_xx_GPa",
                "stress_yy_GPa",
                "stress_zz_GPa",
                "stress_yz_GPa",
                "stress_xz_GPa",
                "stress_xy_GPa",
            ]
        ):
            stresses = np.array([float(row[stress_key]) for row in mode_rows], dtype=float)
            slope, intercept = np.polyfit(strains, stresses, 1)
            raw_tensor[stress_index, mode_index] = slope
            fit_meta[f"{stress_key}_fit_intercept_mode_{VOIGT_LABELS[mode_index]}"] = float(intercept)

    symmetric_tensor = 0.5 * (raw_tensor + raw_tensor.T)
    fit_meta["max_asymmetry_GPa"] = float(np.max(np.abs(raw_tensor - raw_tensor.T)))
    return symmetric_tensor, fit_meta


def mechanical_properties_from_tensor(elastic_tensor: np.ndarray) -> dict[str, float | bool | list[float]]:
    compliance = np.linalg.inv(elastic_tensor)
    c11, c22, c33 = elastic_tensor[0, 0], elastic_tensor[1, 1], elastic_tensor[2, 2]
    c12, c13, c23 = elastic_tensor[0, 1], elastic_tensor[0, 2], elastic_tensor[1, 2]
    c44, c55, c66 = elastic_tensor[3, 3], elastic_tensor[4, 4], elastic_tensor[5, 5]

    s11, s22, s33 = compliance[0, 0], compliance[1, 1], compliance[2, 2]
    s12, s13, s23 = compliance[0, 1], compliance[0, 2], compliance[1, 2]
    s44, s55, s66 = compliance[3, 3], compliance[4, 4], compliance[5, 5]

    bulk_voigt = (c11 + c22 + c33 + 2.0 * (c12 + c13 + c23)) / 9.0
    shear_voigt = (c11 + c22 + c33 - c12 - c13 - c23 + 3.0 * (c44 + c55 + c66)) / 15.0
    bulk_reuss = 1.0 / (s11 + s22 + s33 + 2.0 * (s12 + s13 + s23))
    shear_reuss = 15.0 / (
        4.0 * (s11 + s22 + s33)
        - 4.0 * (s12 + s13 + s23)
        + 3.0 * (s44 + s55 + s66)
    )
    bulk_hill = 0.5 * (bulk_voigt + bulk_reuss)
    shear_hill = 0.5 * (shear_voigt + shear_reuss)
    young_hill = 9.0 * bulk_hill * shear_hill / (3.0 * bulk_hill + shear_hill)
    poisson_hill = (3.0 * bulk_hill - 2.0 * shear_hill) / (2.0 * (3.0 * bulk_hill + shear_hill))
    pugh_ratio = bulk_hill / shear_hill

    eigenvalues = np.linalg.eigvalsh(elastic_tensor)
    return {
        "bulk_voigt_GPa": float(bulk_voigt),
        "bulk_reuss_GPa": float(bulk_reuss),
        "bulk_hill_GPa": float(bulk_hill),
        "shear_voigt_GPa": float(shear_voigt),
        "shear_reuss_GPa": float(shear_reuss),
        "shear_hill_GPa": float(shear_hill),
        "young_hill_GPa": float(young_hill),
        "poisson_hill": float(poisson_hill),
        "pugh_ratio": float(pugh_ratio),
        "min_eigenvalue_GPa": float(np.min(eigenvalues)),
        "eigenvalues_GPa": [float(value) for value in eigenvalues],
        "mechanically_stable": bool(np.all(eigenvalues > 1e-6)),
    }


def write_properties_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_mechanical_response(
    response_rows: list[dict[str, str]],
    elastic_tensor: np.ndarray,
    output_path: Path,
    structure_id: str,
    plt,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 6.8))
    axes = axes.flatten()
    stress_keys = [
        "stress_xx_GPa",
        "stress_yy_GPa",
        "stress_zz_GPa",
        "stress_yz_GPa",
        "stress_xz_GPa",
        "stress_xy_GPa",
    ]

    for mode_index, axis in enumerate(axes):
        mode_label = VOIGT_LABELS[mode_index]
        mode_rows = [row for row in response_rows if row.get("status") == "ok" and row.get("mode_index") == str(mode_index)]
        strains = np.array([float(row["strain_value"]) for row in mode_rows], dtype=float)
        target_stress = np.array([float(row[stress_keys[mode_index]]) for row in mode_rows], dtype=float)
        axis.set_title(f"{mode_label} mode")
        axis.axhline(0.0, color="#cbd5e1", linewidth=0.8)
        axis.axvline(0.0, color="#cbd5e1", linewidth=0.8)
        if len(strains):
            order = np.argsort(strains)
            strains = strains[order]
            target_stress = target_stress[order]
            axis.plot(strains, target_stress, marker="o", linewidth=1.8, color="#0f766e")
            axis.plot(strains, elastic_tensor[mode_index, mode_index] * strains, linestyle="--", linewidth=1.2, color="#dc2626")
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        axis.set_xlabel("Strain")
        axis.set_ylabel("Stress (GPa)")
        axis.grid(alpha=0.22)

    figure.suptitle(f"Mechanical Response: {structure_id}", fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


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
    logs_dir = result_dir / "optimizer_logs"
    structures_dir = result_dir / "strained_structures"
    response_csv_path = result_dir / "mechanical_response.csv"
    tensor_csv_path = result_dir / "elastic_tensor_GPa.csv"
    properties_json_path = result_dir / "mechanical_properties.json"
    plot_path = result_dir / "mechanical_response.png"
    result_dir.mkdir(parents=True, exist_ok=True)

    response_rows: list[dict[str, str]] = []
    for mode_index, mode_label in enumerate(VOIGT_LABELS):
        for point_index, strain_value in enumerate(strain_values, start=1):
            row: dict[str, str] = {
                "mode_index": str(mode_index),
                "mode_label": mode_label,
                "point_index": str(point_index),
                "strain_value": f"{strain_value:.10f}",
                "num_relax_steps": "0",
                "output_poscar": "",
                "optimizer_log": "",
                "status": "ok",
                "error": "",
            }
            try:
                atoms = apply_mode_strain(base_atoms, mode_index, strain_value)
                atoms.calc = make_calculator(model, calculator_cls)

                if args.relax_atoms:
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    log_path = logs_dir / f"{mode_label}_{point_index:03d}_{strain_token(strain_value)}.log"
                    row["optimizer_log"] = str(log_path)
                    row["num_relax_steps"] = str(
                        run_fire_relaxation(atoms, fire_cls, args.fmax, args.max_steps, log_path)
                    )

                energy = float(atoms.get_potential_energy())
                num_sites = len(atoms)
                stress = atoms.get_stress(voigt=True) * EV_A3_TO_GPA
                row["energy_eV"] = f"{energy:.10f}"
                row["energy_per_atom_eV"] = f"{energy / num_sites:.10f}"
                row["volume_A3"] = f"{float(atoms.get_volume()):.10f}"
                row["stress_xx_GPa"] = f"{float(stress[0]):.10f}"
                row["stress_yy_GPa"] = f"{float(stress[1]):.10f}"
                row["stress_zz_GPa"] = f"{float(stress[2]):.10f}"
                row["stress_yz_GPa"] = f"{float(stress[3]):.10f}"
                row["stress_xz_GPa"] = f"{float(stress[4]):.10f}"
                row["stress_xy_GPa"] = f"{float(stress[5]):.10f}"

                if args.write_structures:
                    structures_dir.mkdir(parents=True, exist_ok=True)
                    poscar_path = structures_dir / f"{mode_label}_{point_index:03d}_{strain_token(strain_value)}.POSCAR"
                    adaptor_cls.get_structure(atoms).to(fmt="poscar", filename=str(poscar_path))
                    row["output_poscar"] = str(poscar_path)
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                for key in (
                    "energy_eV",
                    "energy_per_atom_eV",
                    "volume_A3",
                    "stress_xx_GPa",
                    "stress_yy_GPa",
                    "stress_zz_GPa",
                    "stress_yz_GPa",
                    "stress_xz_GPa",
                    "stress_xy_GPa",
                ):
                    row.setdefault(key, "")
            response_rows.append(row)

    write_csv(response_csv_path, response_rows, response_fieldnames())

    successful_rows = [row for row in response_rows if row.get("status") == "ok"]
    if len(successful_rows) < 12:
        raise ValueError("Too few successful strained states to fit elastic constants.")

    elastic_tensor, fit_meta = fit_elastic_tensor(response_rows)
    write_matrix_csv(tensor_csv_path, elastic_tensor)
    properties = mechanical_properties_from_tensor(elastic_tensor)
    properties_payload: dict[str, object] = {
        "structure_id": structure_id,
        "formula": structure.composition.formula.replace(" ", ""),
        "strain_max": float(args.strain_max),
        "num_points_per_mode": int(args.num_points),
        "relax_atoms": bool(args.relax_atoms),
        "elastic_tensor_GPa": [[float(value) for value in row] for row in elastic_tensor.tolist()],
    }
    properties_payload.update(fit_meta)
    properties_payload.update(properties)
    write_properties_json(properties_json_path, properties_payload)
    if args.plot:
        plot_mechanical_response(response_rows, elastic_tensor, plot_path, structure_id, plt)

    return {
        "id": structure_id,
        "input_file": str(structure_path),
        "status": "ok",
        "num_sites": str(len(structure)),
        "formula": structure.composition.formula.replace(" ", ""),
        "strain_max": f"{float(args.strain_max):.10f}",
        "num_points_per_mode": str(int(args.num_points)),
        "relax_atoms": str(bool(args.relax_atoms)),
        "num_successful_evaluations": str(len(successful_rows)),
        "elastic_tensor_csv": str(tensor_csv_path),
        "response_csv": str(response_csv_path),
        "properties_json": str(properties_json_path),
        "plot_file": str(plot_path) if args.plot else "",
        "bulk_voigt_GPa": f"{float(properties['bulk_voigt_GPa']):.10f}",
        "bulk_reuss_GPa": f"{float(properties['bulk_reuss_GPa']):.10f}",
        "bulk_hill_GPa": f"{float(properties['bulk_hill_GPa']):.10f}",
        "shear_voigt_GPa": f"{float(properties['shear_voigt_GPa']):.10f}",
        "shear_reuss_GPa": f"{float(properties['shear_reuss_GPa']):.10f}",
        "shear_hill_GPa": f"{float(properties['shear_hill_GPa']):.10f}",
        "young_hill_GPa": f"{float(properties['young_hill_GPa']):.10f}",
        "poisson_hill": f"{float(properties['poisson_hill']):.10f}",
        "pugh_ratio": f"{float(properties['pugh_ratio']):.10f}",
        "min_eigenvalue_GPa": f"{float(properties['min_eigenvalue_GPa']):.10f}",
        "max_asymmetry_GPa": f"{float(fit_meta['max_asymmetry_GPa']):.10f}",
        "mechanically_stable": str(bool(properties["mechanically_stable"])),
        "error": "",
    }


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    summary_path = args.summary.expanduser() if args.summary else output_dir / "mechanical_summary.csv"

    validate_args(args)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    structures = discover_structures(input_dir, args.recursive, args.input_format)
    if not structures:
        raise SystemExit(f"No structure files found in: {input_dir}")

    strain_values = build_strain_values(args.strain_max, args.num_points)
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
                    "strain_max": f"{float(args.strain_max):.10f}",
                    "num_points_per_mode": str(int(args.num_points)),
                    "relax_atoms": str(bool(args.relax_atoms)),
                    "num_successful_evaluations": "0",
                    "elastic_tensor_csv": "",
                    "response_csv": "",
                    "properties_json": "",
                    "plot_file": "",
                    "bulk_voigt_GPa": "",
                    "bulk_reuss_GPa": "",
                    "bulk_hill_GPa": "",
                    "shear_voigt_GPa": "",
                    "shear_reuss_GPa": "",
                    "shear_hill_GPa": "",
                    "young_hill_GPa": "",
                    "poisson_hill": "",
                    "pugh_ratio": "",
                    "min_eigenvalue_GPa": "",
                    "max_asymmetry_GPa": "",
                    "mechanically_stable": "",
                    "error": str(exc),
                }
            )

    write_csv(summary_path, summary_rows, summary_fieldnames())
    print(f"Saved mechanical summary to: {summary_path}")


if __name__ == "__main__":
    main()
