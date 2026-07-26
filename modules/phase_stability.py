#!/usr/bin/env python3
"""Compute approximate phase stability for CHGNet alloy structures.

Examples
--------
Compare BCC/FCC/HCP for a fixed HEA composition:
    python3 modules/phase_stability.py \
        --mode fixed \
        --elements "Hf Nb Ta Ti Zr" \
        --ratios "1 1 1 1 1" \
        --phases "bcc,fcc,hcp"

Scan a binary alloy along composition:
    python3 modules/phase_stability.py \
        --mode binary \
        --element-a Ti \
        --element-b Nb \
        --x-start 0.1 \
        --x-end 0.9 \
        --num-points 9 \
        --phases "bcc,fcc"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from pathlib import Path
from types import SimpleNamespace

from generate_hea_structures import (
    allocate_counts,
    assign_species,
    build_formula,
    build_structure,
    normalize_ratios,
    validate_elements,
    write_outputs,
)
from relax_structures import (
    make_calculator,
    relax_structure,
    require_ase,
    require_chgnet,
    require_pymatgen,
    resolve_relaxation_settings,
)
from model_loader import load_chgnet_model


SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = APP_ROOT / "calculator_runs" / "phase_results"
DEFAULT_LATTICE_CONSTANTS = {"bcc": 3.0, "fcc": 3.6, "hcp": 2.95}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute approximate CHGNet phase stability for fixed or binary alloy compositions."
    )
    parser.add_argument("--mode", choices=("fixed", "binary"), default="fixed")
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phases", type=str, default="bcc,fcc,hcp")
    parser.add_argument("--samples-per-phase", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supercell", nargs=3, type=int, default=[3, 3, 3])
    parser.add_argument("--lattice-bcc", type=float, default=DEFAULT_LATTICE_CONSTANTS["bcc"])
    parser.add_argument("--lattice-fcc", type=float, default=DEFAULT_LATTICE_CONSTANTS["fcc"])
    parser.add_argument("--lattice-hcp", type=float, default=DEFAULT_LATTICE_CONSTANTS["hcp"])
    parser.add_argument("--hcp-c-over-a", type=float, default=1.633)
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument("--ediffg", type=float, default=-0.005)
    parser.add_argument("--nsw", type=int, default=200)
    parser.add_argument("--isif", type=int, default=3)
    parser.add_argument("--loginterval", type=int, default=1)
    parser.add_argument("--elements", type=str, default="")
    parser.add_argument("--ratios", type=str, default="")
    parser.add_argument("--element-a", type=str, default="Ti")
    parser.add_argument("--element-b", type=str, default="Nb")
    parser.add_argument("--x-start", type=float, default=0.1)
    parser.add_argument("--x-end", type=float, default=0.9)
    parser.add_argument("--num-points", type=int, default=9)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="CHGNet model: builtin name (0.3.0/0.2.0/r2scan) or a .pth.tar filename/path in the models/ folder. Default: 0.3.0.",
    )
    return parser


def split_values(raw_value: str) -> list[str]:
    return [token for token in raw_value.replace(",", " ").split() if token]


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "phase_task"


def parse_phases(raw_value: str) -> list[str]:
    phases = [phase.strip().lower() for phase in raw_value.split(",") if phase.strip()]
    if not phases:
        raise SystemExit("At least one phase must be provided.")
    invalid = [phase for phase in phases if phase not in {"bcc", "fcc", "hcp"}]
    if invalid:
        raise SystemExit(f"Unsupported phases: {', '.join(invalid)}")
    return phases


def build_binary_points(element_a: str, element_b: str, x_start: float, x_end: float, num_points: int) -> list[dict[str, object]]:
    elements = validate_elements([element_a, element_b])
    if num_points <= 0:
        raise SystemExit("--num-points must be positive.")
    if not (0.0 <= x_start <= 1.0 and 0.0 <= x_end <= 1.0):
        raise SystemExit("Binary scan fractions must be between 0 and 1.")

    if num_points == 1:
        return [
            {
                "point_index": 1,
                "elements": elements,
                "fractions": [x_start, 1.0 - x_start],
                "composition_x": x_start,
                "composition_label": f"{elements[0]} {x_start:.3f} / {elements[1]} {1.0 - x_start:.3f}",
            }
        ]

    step = (x_end - x_start) / (num_points - 1)
    points: list[dict[str, object]] = []
    for index in range(num_points):
        x_value = x_start + step * index
        points.append(
            {
                "point_index": index + 1,
                "elements": elements,
                "fractions": [x_value, 1.0 - x_value],
                "composition_x": x_value,
                "composition_label": f"{elements[0]} {x_value:.3f} / {elements[1]} {1.0 - x_value:.3f}",
            }
        )
    return points


def build_fixed_points(elements_text: str, ratios_text: str) -> list[dict[str, object]]:
    elements = validate_elements(split_values(elements_text))
    if not elements:
        raise SystemExit("Provide at least one element for fixed-composition mode.")
    if not ratios_text.strip():
        raise SystemExit("Provide ratios for fixed-composition mode.")
    fractions = normalize_ratios([float(value) for value in split_values(ratios_text)], len(elements))
    composition_label = ", ".join(f"{element} {fraction:.3f}" for element, fraction in zip(elements, fractions))
    return [
        {
            "point_index": 1,
            "elements": elements,
            "fractions": fractions,
            "composition_x": "",
            "composition_label": composition_label,
        }
    ]


def build_points(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.mode == "fixed":
        return build_fixed_points(args.elements, args.ratios)
    return build_binary_points(args.element_a, args.element_b, args.x_start, args.x_end, args.num_points)


def build_task_name(args: argparse.Namespace) -> str:
    if args.task_name.strip():
        return sanitize_slug(args.task_name)
    if args.mode == "fixed":
        return sanitize_slug("fixed_" + "_".join(split_values(args.elements) or ["alloy"]))
    return sanitize_slug(f"binary_{args.element_a}_{args.element_b}")


def lattice_for_phase(args: argparse.Namespace, phase: str) -> float:
    return {"bcc": args.lattice_bcc, "fcc": args.lattice_fcc, "hcp": args.lattice_hcp}[phase]


def evaluate_static_structure(structure_path: Path, output_dir: Path, model, calculator_cls, structure_cls, adaptor_cls) -> dict[str, str]:
    structure_id = structure_path.stem
    result_dir = output_dir / structure_id
    result_dir.mkdir(parents=True, exist_ok=True)

    structure = structure_cls.from_file(str(structure_path))
    atoms = adaptor_cls.get_atoms(structure)
    atoms.calc = make_calculator(model, calculator_cls)
    energy = float(atoms.get_potential_energy())
    final_structure = adaptor_cls.get_structure(atoms)
    final_poscar = result_dir / f"{structure_id}_single_point.POSCAR"
    final_structure.to(fmt="poscar", filename=str(final_poscar))
    return {
        "id": structure_id,
        "input_file": str(structure_path),
        "status": "ok",
        "num_recorded_steps": "1",
        "num_sites": str(len(final_structure)),
        "formula": final_structure.composition.formula.replace(" ", ""),
        "energy_eV": f"{energy:.10f}",
        "energy_per_atom_eV": f"{energy / len(final_structure):.10f}",
        "volume_A3": f"{float(final_structure.volume):.10f}",
        "poscar_output": str(final_poscar),
        "step_csv": "",
        "trajectory_file": "",
        "optimizer_log": "",
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_runs(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((int(row["point_index"]), str(row["phase"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    best_by_point: dict[int, tuple[str, float]] = {}

    for (point_index, phase), rows in grouped.items():
        ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("energy_per_atom_eV") not in ("", None)]
        energies = [float(row["energy_per_atom_eV"]) for row in ok_rows]
        volumes = [float(row["volume_per_atom_A3"]) for row in ok_rows if row.get("volume_per_atom_A3") not in ("", None)]
        mean_energy = statistics.mean(energies) if energies else None
        std_energy = statistics.pstdev(energies) if len(energies) > 1 else 0.0 if energies else None
        mean_volume = statistics.mean(volumes) if volumes else None
        if mean_energy is not None:
            current_best = best_by_point.get(point_index)
            if current_best is None or mean_energy < current_best[1]:
                best_by_point[point_index] = (phase, mean_energy)
        first_row = rows[0]
        summary_rows.append(
            {
                "mode": first_row["mode"],
                "task_name": first_row["task_name"],
                "point_index": point_index,
                "composition_label": first_row["composition_label"],
                "composition_x": first_row["composition_x"],
                "phase": phase,
                "requested_elements": first_row["requested_elements"],
                "requested_fractions": first_row["requested_fractions"],
                "num_samples": len(rows),
                "successful_samples": len(ok_rows),
                "mean_energy_per_atom_eV": f"{mean_energy:.10f}" if mean_energy is not None else "",
                "std_energy_per_atom_eV": f"{std_energy:.10f}" if std_energy is not None else "",
                "mean_volume_per_atom_A3": f"{mean_volume:.10f}" if mean_volume is not None else "",
                "sample_ids": ", ".join(str(row["sample_id"]) for row in rows),
            }
        )

    for row in summary_rows:
        point_index = int(row["point_index"])
        best_phase, best_energy = best_by_point.get(point_index, ("", float("nan")))
        delta_mev = ""
        if row.get("mean_energy_per_atom_eV") not in ("", None):
            delta_mev = f"{(float(row['mean_energy_per_atom_eV']) - best_energy) * 1000:.4f}"
        row["lowest_energy_phase"] = best_phase
        row["delta_to_best_meV_per_atom"] = delta_mev

    summary_rows.sort(key=lambda row: (int(row["point_index"]), str(row["phase"])))
    return summary_rows


def run_phase_stability(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.samples_per_phase <= 0:
        raise SystemExit("--samples-per-phase must be positive.")
    if args.loginterval <= 0:
        raise SystemExit("--loginterval must be positive.")
    if any(value <= 0 for value in args.supercell):
        raise SystemExit("--supercell values must be positive integers.")

    task_name = build_task_name(args)
    task_dir = args.output_dir.expanduser() / task_name
    structures_dir = task_dir / "structures"
    relaxed_dir = task_dir / "relaxed"
    task_dir.mkdir(parents=True, exist_ok=True)

    phases = parse_phases(args.phases)
    points = build_points(args)
    structure_cls, adaptor_cls = require_pymatgen()
    model_cls, calculator_cls = require_chgnet()
    model = load_chgnet_model(model_cls, model_arg=args.model)

    if args.no_relax:
        fire_cls = trajectory_cls = None
        fmax = max_steps = 0
        relax_cell = False
    else:
        fire_cls, trajectory_cls, _ = require_ase()
        settings = SimpleNamespace(fmax=None, max_steps=None, relax_cell=False, ediffg=args.ediffg, nsw=args.nsw, isif=args.isif)
        fmax, max_steps, relax_cell = resolve_relaxation_settings(settings)

    raw_rows: list[dict[str, object]] = []
    total_runs = len(points) * len(phases) * args.samples_per_phase
    run_counter = 0

    print(f"Task name: {task_name}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {task_dir}")
    print(f"Composition points: {len(points)}")
    print(f"Phases: {', '.join(phases)}")
    print(f"Samples per phase: {args.samples_per_phase}")
    print(f"Relax structures: {not args.no_relax}")

    for point in points:
        point_index = int(point["point_index"])
        elements = list(point["elements"])
        fractions = list(point["fractions"])
        requested_elements = " ".join(elements)
        requested_fractions = " ".join(f"{value:.6f}" for value in fractions)

        for phase in phases:
            lattice_constant = lattice_for_phase(args, phase)
            for sample_index in range(1, args.samples_per_phase + 1):
                run_counter += 1
                rng_seed = args.seed + point_index * 1000 + phases.index(phase) * 100 + sample_index - 1
                structure = build_structure(phase, lattice_constant, list(args.supercell), args.hcp_c_over_a)
                counts = allocate_counts(fractions, len(structure))
                structure = assign_species(structure, elements, counts, random.Random(rng_seed))
                formula = build_formula(elements, counts)
                sample_id = f"p{point_index:03d}_{phase}_s{sample_index:02d}_{formula}"

                phase_structure_dir = structures_dir / f"point_{point_index:03d}" / phase
                phase_structure_dir.mkdir(parents=True, exist_ok=True)
                cif_path = phase_structure_dir / f"{sample_id}.cif"
                poscar_path = phase_structure_dir / f"{sample_id}.POSCAR"
                write_outputs(structure, elements, cif_path, poscar_path)

                print(f"[{run_counter:03d}/{total_runs:03d}] Point {point_index}, phase={phase}, sample={sample_index}, composition={point['composition_label']}")
                try:
                    if args.no_relax:
                        result = evaluate_static_structure(
                            structure_path=poscar_path,
                            output_dir=relaxed_dir / f"point_{point_index:03d}" / phase,
                            model=model,
                            calculator_cls=calculator_cls,
                            structure_cls=structure_cls,
                            adaptor_cls=adaptor_cls,
                        )
                    else:
                        result = relax_structure(
                            structure_path=poscar_path,
                            output_dir=relaxed_dir / f"point_{point_index:03d}" / phase,
                            model=model,
                            calculator_cls=calculator_cls,
                            structure_cls=structure_cls,
                            adaptor_cls=adaptor_cls,
                            fire_cls=fire_cls,
                            trajectory_cls=trajectory_cls,
                            fmax=fmax,
                            max_steps=max_steps,
                            relax_cell=relax_cell,
                            ase_filter="FrechetCellFilter",
                            loginterval=args.loginterval,
                            verbose=False,
                        )
                    volume_per_atom = float(result["volume_A3"]) / float(result["num_sites"])
                    raw_rows.append(
                        {
                            "task_name": task_name,
                            "mode": args.mode,
                            "point_index": point_index,
                            "composition_label": point["composition_label"],
                            "composition_x": point["composition_x"],
                            "phase": phase,
                            "sample_index": sample_index,
                            "sample_id": sample_id,
                            "requested_elements": requested_elements,
                            "requested_fractions": requested_fractions,
                            "formula": result.get("formula", formula),
                            "status": result.get("status", "ok"),
                            "energy_eV": result.get("energy_eV", ""),
                            "energy_per_atom_eV": result.get("energy_per_atom_eV", ""),
                            "volume_A3": result.get("volume_A3", ""),
                            "volume_per_atom_A3": f"{volume_per_atom:.10f}",
                            "input_file": str(poscar_path),
                            "cif_file": str(cif_path),
                            "final_structure": result.get("poscar_output", ""),
                            "step_csv": result.get("step_csv", ""),
                            "trajectory_file": result.get("trajectory_file", ""),
                            "optimizer_log": result.get("optimizer_log", ""),
                            "error": "",
                        }
                    )
                    print(f"  done: {result.get('energy_per_atom_eV', 'n/a')} eV/atom")
                except Exception as exc:
                    raw_rows.append(
                        {
                            "task_name": task_name,
                            "mode": args.mode,
                            "point_index": point_index,
                            "composition_label": point["composition_label"],
                            "composition_x": point["composition_x"],
                            "phase": phase,
                            "sample_index": sample_index,
                            "sample_id": sample_id,
                            "requested_elements": requested_elements,
                            "requested_fractions": requested_fractions,
                            "formula": formula,
                            "status": "failed",
                            "energy_eV": "",
                            "energy_per_atom_eV": "",
                            "volume_A3": "",
                            "volume_per_atom_A3": "",
                            "input_file": str(poscar_path),
                            "cif_file": str(cif_path),
                            "final_structure": "",
                            "step_csv": "",
                            "trajectory_file": "",
                            "optimizer_log": "",
                            "error": str(exc),
                        }
                    )
                    print(f"  failed: {exc}")

    raw_path = task_dir / "phase_runs.csv"
    summary_path = task_dir / "phase_summary.csv"
    metadata_path = task_dir / "metadata.json"

    raw_fields = [
        "task_name", "mode", "point_index", "composition_label", "composition_x", "phase", "sample_index", "sample_id",
        "requested_elements", "requested_fractions", "formula", "status", "energy_eV", "energy_per_atom_eV",
        "volume_A3", "volume_per_atom_A3", "input_file", "cif_file", "final_structure", "step_csv",
        "trajectory_file", "optimizer_log", "error",
    ]
    summary_fields = [
        "mode", "task_name", "point_index", "composition_label", "composition_x", "phase", "requested_elements",
        "requested_fractions", "num_samples", "successful_samples", "mean_energy_per_atom_eV",
        "std_energy_per_atom_eV", "mean_volume_per_atom_A3", "lowest_energy_phase", "delta_to_best_meV_per_atom", "sample_ids",
    ]

    write_csv(raw_path, raw_rows, raw_fields)
    summary_rows = summarize_runs(raw_rows)
    write_csv(summary_path, summary_rows, summary_fields)
    metadata_path.write_text(json.dumps({"task_name": task_name, "mode": args.mode, "phases": phases, "samples_per_phase": args.samples_per_phase, "relaxed": not args.no_relax, "num_points": len(points)}, indent=2), encoding="utf-8")
    return raw_path, summary_path


def main() -> None:
    args = build_parser().parse_args()
    raw_path, summary_path = run_phase_stability(args)
    print(f"Raw run CSV: {raw_path}")
    print(f"Phase summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
