#!/usr/bin/env python3
"""Run CHGNet molecular dynamics for CIF/POSCAR structures.

Examples
--------
Run NVT MD for all generated structures:
    python3 modules/run_molecular_dynamics.py \
        --input-dir calculator_runs/generated_structures \
        --output-dir calculator_runs/md_results \
        --ensemble nvt \
        --temperature 1000 \
        --steps 50

Run NPT MD recursively:
    python3 modules/run_molecular_dynamics.py \
        --input-dir some_structures \
        --output-dir calculator_runs/md_results \
        --recursive \
        --ensemble npt \
        --temperature 1200 \
        --pressure 0.000101325 \
        --steps 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from model_loader import load_chgnet_model


STRUCTURE_SUFFIXES = {".cif", ".vasp", ".poscar", ".POSCAR"}
EV_A3_TO_GPA = 160.21766208
SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_DIR = APP_ROOT / "calculator_runs" / "generated_structures"
DEFAULT_OUTPUT_DIR = APP_ROOT / "calculator_runs" / "md_results"
DEFAULT_RECURSIVE = False
DEFAULT_INPUT_FORMAT = "poscar"
DEFAULT_ENSEMBLE = "nvt"
DEFAULT_THERMOSTAT = "Berendsen_inhomogeneous"
DEFAULT_TEMPERATURE = 1000
DEFAULT_TIMESTEP = 2.0
DEFAULT_PRESSURE = 0.000101325
DEFAULT_STEPS = 50
DEFAULT_LOGINTERVAL = 10
DEFAULT_SUMMARY = None
DEFAULT_USE_DEVICE = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CHGNet molecular dynamics for all CIF/POSCAR structures in a directory."
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
        help="Directory where MD outputs and the CSV summary will be saved.",
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
        "--ensemble",
        choices=("nve", "nvt", "npt"),
        default=DEFAULT_ENSEMBLE,
        help="MD ensemble to use.",
    )
    parser.add_argument(
        "--thermostat",
        type=str,
        default=DEFAULT_THERMOSTAT,
        help="Thermostat passed to CHGNet MolecularDynamics.",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=DEFAULT_TEMPERATURE,
        help="Target temperature in K.",
    )
    parser.add_argument(
        "--starting-temperature",
        type=int,
        default=None,
        help="Optional starting temperature in K. Defaults to CHGNet behavior if omitted.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=DEFAULT_TIMESTEP,
        help="MD timestep in femtoseconds.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=DEFAULT_PRESSURE,
        help="Target pressure for NPT runs.",
    )
    parser.add_argument(
        "--taut",
        type=float,
        default=None,
        help="Optional thermostat time constant.",
    )
    parser.add_argument(
        "--taup",
        type=float,
        default=None,
        help="Optional barostat time constant.",
    )
    parser.add_argument(
        "--bulk-modulus",
        type=float,
        default=None,
        help="Optional bulk modulus for NPT runs.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of MD steps to run.",
    )
    parser.add_argument(
        "--loginterval",
        type=int,
        default=DEFAULT_LOGINTERVAL,
        help="Write MD log/trajectory every N MD steps.",
    )
    parser.add_argument(
        "--use-device",
        type=str,
        default=DEFAULT_USE_DEVICE,
        help="Optional CHGNet device override: cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Optional CSV summary output path. Default: <output-dir>/md_summary.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="CHGNet model: builtin name (0.3.0/0.2.0/r2scan) or a .pth.tar filename/path in the models/ folder. Default: 0.3.0.",
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
        from ase.io.trajectory import Trajectory
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'ase'. Install ASE in the environment that runs this script."
        ) from exc
    return Trajectory


def require_chgnet():
    try:
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import MolecularDynamics
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'chgnet'. Install CHGNet in the environment that runs this script."
        ) from exc
    return CHGNet, MolecularDynamics


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


def record_md_step(rows: list[dict[str, str]], atoms, step: int, timestep_fs: float) -> None:
    try:
        stress = atoms.get_stress(voigt=True) * EV_A3_TO_GPA
        pressure_values = [f"{float(value):.10f}" for value in stress]
    except Exception:
        pressure_values = ["", "", "", "", "", ""]
    try:
        epot = float(atoms.get_potential_energy())
    except Exception:
        epot = 0.0
    try:
        ekin = float(atoms.get_kinetic_energy())
    except Exception:
        ekin = 0.0
    try:
        temperature = float(atoms.get_temperature())
    except Exception:
        temperature = 0.0
    rows.append(
        {
            "step": str(step),
            "time_ps": f"{step * timestep_fs / 1000:.10f}",
            "etot_eV": f"{epot + ekin:.10f}",
            "epot_eV": f"{epot:.10f}",
            "ekin_eV": f"{ekin:.10f}",
            "temperature_K": f"{temperature:.10f}",
            "pressure_xx_GPa": pressure_values[0],
            "pressure_yy_GPa": pressure_values[1],
            "pressure_zz_GPa": pressure_values[2],
            "pressure_yz_GPa": pressure_values[3],
            "pressure_xz_GPa": pressure_values[4],
            "pressure_xy_GPa": pressure_values[5],
        }
    )


def write_md_step_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "step",
        "time_ps",
        "etot_eV",
        "epot_eV",
        "ekin_eV",
        "temperature_K",
        "pressure_xx_GPa",
        "pressure_yy_GPa",
        "pressure_zz_GPa",
        "pressure_yz_GPa",
        "pressure_xz_GPa",
        "pressure_xy_GPa",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "input_file",
        "status",
        "ensemble",
        "temperature_K",
        "steps",
        "timestep_fs",
        "pressure",
        "starting_temperature_K",
        "num_frames",
        "final_energy_eV",
        "final_volume_A3",
        "pressure_xx_GPa",
        "pressure_yy_GPa",
        "pressure_zz_GPa",
        "pressure_yz_GPa",
        "pressure_xz_GPa",
        "pressure_xy_GPa",
        "final_structure",
        "step_csv",
        "trajectory_file",
        "log_file",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_md_for_structure(
    structure_path: Path,
    output_dir: Path,
    structure_cls,
    adaptor_cls,
    trajectory_cls,
    model,
    molecular_dynamics_cls,
    args: argparse.Namespace,
) -> dict[str, str]:
    structure_id = structure_identifier(structure_path)
    result_dir = output_dir / structure_id
    result_dir.mkdir(parents=True, exist_ok=True)

    structure = structure_cls.from_file(str(structure_path))
    input_copy_path = result_dir / structure_path.name
    structure.to(filename=str(input_copy_path))

    trajectory_path = result_dir / "md.traj"
    logfile_path = result_dir / "md.log"
    step_csv_path = result_dir / "md_steps.csv"
    final_poscar_path = result_dir / f"{structure_id}_md_final.POSCAR"

    md_rows: list[dict[str, str]] = []

    md = molecular_dynamics_cls(
        atoms=structure,
        model=model,
        ensemble=args.ensemble,
        thermostat=args.thermostat,
        temperature=args.temperature,
        starting_temperature=args.starting_temperature,
        timestep=args.timestep,
        pressure=args.pressure,
        taut=args.taut,
        taup=args.taup,
        bulk_modulus=args.bulk_modulus,
        trajectory=str(trajectory_path),
        logfile=str(logfile_path),
        loginterval=args.loginterval,
        use_device=args.use_device,
    )

    def capture_md_step() -> None:
        step = getattr(md.dyn, "nsteps", 0)
        record_md_step(md_rows, md.atoms, step=step, timestep_fs=args.timestep)

    capture_md_step()
    md.dyn.attach(capture_md_step, interval=args.loginterval)
    md.run(args.steps)
    write_md_step_csv(step_csv_path, md_rows)

    traj = trajectory_cls(str(trajectory_path))
    final_atoms = traj[-1]
    num_frames = len(traj)
    final_structure = adaptor_cls.get_structure(final_atoms)
    final_structure.to(fmt="poscar", filename=str(final_poscar_path))
    final_energy = ""
    try:
        final_energy = f"{float(final_atoms.get_potential_energy()):.10f}"
    except Exception:
        final_energy = ""
    final_volume = ""
    try:
        final_volume = f"{float(final_atoms.get_volume()):.10f}"
    except Exception:
        final_volume = ""
    pressure_components = ["", "", "", "", "", ""]
    try:
        stress = final_atoms.get_stress(voigt=True) * EV_A3_TO_GPA
        pressure_components = [f"{float(value):.10f}" for value in stress]
    except Exception:
        pressure_components = ["", "", "", "", "", ""]
    traj.close()

    return {
        "id": structure_id,
        "input_file": str(structure_path),
        "status": "ok",
        "ensemble": args.ensemble,
        "temperature_K": str(args.temperature),
        "steps": str(args.steps),
        "timestep_fs": str(args.timestep),
        "pressure": str(args.pressure),
        "starting_temperature_K": str(args.starting_temperature),
        "num_frames": str(num_frames),
        "final_energy_eV": final_energy,
        "final_volume_A3": final_volume,
        "pressure_xx_GPa": pressure_components[0],
        "pressure_yy_GPa": pressure_components[1],
        "pressure_zz_GPa": pressure_components[2],
        "pressure_yz_GPa": pressure_components[3],
        "pressure_xz_GPa": pressure_components[4],
        "pressure_xy_GPa": pressure_components[5],
        "final_structure": str(final_poscar_path),
        "step_csv": str(step_csv_path),
        "trajectory_file": str(trajectory_path),
        "log_file": str(logfile_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    summary_path = args.summary.expanduser() if args.summary else output_dir / "md_summary.csv"

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if args.steps <= 0:
        raise SystemExit("--steps must be a positive integer.")
    if args.loginterval <= 0:
        raise SystemExit("--loginterval must be a positive integer.")
    if args.timestep <= 0:
        raise SystemExit("--timestep must be positive.")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive.")
    if args.starting_temperature is None:
        args.starting_temperature = args.temperature

    structure_cls, adaptor_cls = require_pymatgen()
    trajectory_cls = require_ase()
    chgnet_cls, molecular_dynamics_cls = require_chgnet()
    structures = discover_structures(input_dir, recursive=args.recursive, input_format=args.input_format)
    if not structures:
        raise SystemExit(f"No structure files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_chgnet_model(chgnet_cls, model_arg=args.model, use_device=args.use_device)
    summary_rows: list[dict[str, str]] = []

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Structures found: {len(structures)}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Temperature: {args.temperature} K")
    print(f"Starting temperature: {args.starting_temperature} K")
    print(f"Steps: {args.steps}")
    print(f"Timestep: {args.timestep} fs")

    for index, structure_path in enumerate(structures, start=1):
        structure_id = structure_identifier(structure_path)
        print(f"[{index:03d}/{len(structures):03d}] Running MD for {structure_id} from {structure_path.name}")
        try:
            row = run_md_for_structure(
                structure_path=structure_path,
                output_dir=output_dir,
                structure_cls=structure_cls,
                adaptor_cls=adaptor_cls,
                trajectory_cls=trajectory_cls,
                model=model,
                molecular_dynamics_cls=molecular_dynamics_cls,
                args=args,
            )
            print(
                f"  done: frames={row['num_frames']}, "
                f"final_energy={row['final_energy_eV'] or 'n/a'} eV, "
                f"final_volume={row['final_volume_A3'] or 'n/a'} A^3"
            )
        except Exception as exc:
            row = {
                "id": structure_id,
                "input_file": str(structure_path),
                "status": "failed",
                "ensemble": args.ensemble,
                "temperature_K": str(args.temperature),
                "steps": str(args.steps),
                "timestep_fs": str(args.timestep),
                "pressure": str(args.pressure),
                "starting_temperature_K": str(args.starting_temperature),
                "pressure_xx_GPa": "",
                "pressure_yy_GPa": "",
                "pressure_zz_GPa": "",
                "pressure_yz_GPa": "",
                "pressure_xz_GPa": "",
                "pressure_xy_GPa": "",
                "step_csv": "",
                "error": str(exc),
            }
            print(f"  failed: {exc}")
        summary_rows.append(row)

    write_summary(summary_path, summary_rows)
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
