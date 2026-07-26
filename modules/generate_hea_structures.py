#!/usr/bin/env python3
"""Generate a BCC/FCC/HCP high-entropy alloy structure with pymatgen.

Configuration may be supplied either via individual CLI flags or, for backward
compatibility, via an input text file passed to --input. CLI flags take
precedence over the file when both are given.

Example
-------
    python3 modules/generate_hea_structures.py \
        --elements "Hf Nb Ta Ti Zr" --ratios "1 1 1 1 1" --crystal-structure bcc
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Poscar


DEFAULT_CANDIDATE_ELEMENTS = [
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Zr",
    "Nb",
    "Mo",
    "Ta",
    "W",
]

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent

DEFAULT_OUTPUT_DIR = APP_ROOT / "calculator_runs" / "generated_structures"


def load_pymatgen():
    try:
        from pymatgen.core import Element, Lattice, Structure
        from pymatgen.io.vasp.inputs import Poscar
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "pymatgen is required. Install it first, for example: pip install pymatgen"
        ) from exc
    return Element, Lattice, Structure, Poscar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a BCC/FCC/HCP high-entropy alloy structure."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to a legacy input text file. CLI flags override file values.",
    )
    parser.add_argument(
        "--elements",
        type=str,
        default=None,
        help="Space- or comma-separated element symbols, e.g. \"Hf Nb Ta Ti Zr\".",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="Space- or comma-separated composition ratios matching --elements.",
    )
    parser.add_argument(
        "--num-elements",
        type=int,
        default=None,
        help="Randomly choose this many elements from --candidate-elements.",
    )
    parser.add_argument(
        "--candidate-elements",
        type=str,
        default=None,
        help="Pool of candidate elements for random selection (used with --num-elements).",
    )
    parser.add_argument(
        "--crystal-structure",
        choices=("bcc", "fcc", "hcp"),
        default=None,
        help="Crystal structure to build.",
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=None,
        help="Cubic lattice constant in Angstroms.",
    )
    parser.add_argument(
        "--hcp-c-over-a",
        type=float,
        default=None,
        help="HCP c/a ratio.",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell replication, e.g. 4 4 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for element assignment.",
    )
    parser.add_argument(
        "--num-structures",
        type=int,
        default=None,
        help="Number of structures to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CIF/POSCAR files.",
    )
    return parser


def split_values(raw_value: str) -> list[str]:
    return raw_value.replace(",", " ").split()


def parse_input_file(input_path: Path) -> dict[str, str]:
    if not input_path.is_file():
        raise ValueError(f"Input file not found: {input_path}")

    config: dict[str, str] = {}
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            raise ValueError(f"Invalid line {line_number} in {input_path}: {raw_line}")

        key = key.strip().lower().replace("-", "_")
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid line {line_number} in {input_path}: {raw_line}")
        config[key] = value

    return config


def build_config(config_map: dict[str, str], input_path: Path | None) -> argparse.Namespace:
    elements = split_values(config_map["elements"]) if "elements" in config_map else None
    candidate_elements = (
        split_values(config_map["candidate_elements"])
        if "candidate_elements" in config_map
        else DEFAULT_CANDIDATE_ELEMENTS
    )

    try:
        num_elements = int(config_map["num_elements"]) if "num_elements" in config_map else None
        ratios = [float(value) for value in split_values(config_map["ratios"])]
        lattice_constant = float(config_map.get("lattice_constant", "3.0"))
        supercell = [int(value) for value in split_values(config_map.get("supercell", "4 4 4"))]
        hcp_c_over_a = float(config_map.get("hcp_c_over_a", "1.633"))
        seed = int(config_map.get("seed", "42"))
        num_structures = int(config_map.get("num_structures", "1"))
    except ValueError as exc:
        raise ValueError(f"Failed to parse numeric values in {input_path}: {exc}") from exc

    if len(supercell) != 3:
        raise ValueError("supercell must contain exactly 3 integers, e.g. 4 4 4.")
    if num_structures <= 0:
        raise ValueError("num_structures must be positive.")
    crystal_structure = config_map.get("crystal_structure", "bcc").strip().lower()
    if crystal_structure not in {"bcc", "fcc", "hcp"}:
        raise ValueError("crystal_structure must be one of: bcc, fcc, hcp.")
    if hcp_c_over_a <= 0:
        raise ValueError("hcp_c_over_a must be positive.")

    output_dir = Path(config_map.get("output_dir", "calculator_runs/generated_structures"))
    if not output_dir.is_absolute():
        output_dir = APP_ROOT / output_dir

    return argparse.Namespace(
        elements=elements,
        num_elements=num_elements,
        candidate_elements=candidate_elements,
        ratios=ratios,
        crystal_structure=crystal_structure,
        lattice_constant=lattice_constant,
        hcp_c_over_a=hcp_c_over_a,
        supercell=supercell,
        seed=seed,
        num_structures=num_structures,
        output_dir=output_dir,
    )


def apply_cli_overrides(config: argparse.Namespace, cli_args: argparse.Namespace) -> argparse.Namespace:
    """Override file/derived config with explicit CLI flags."""
    if cli_args.elements is not None:
        config.elements = split_values(cli_args.elements)
    if cli_args.candidate_elements is not None:
        config.candidate_elements = split_values(cli_args.candidate_elements)
    if cli_args.num_elements is not None:
        config.num_elements = cli_args.num_elements
    if cli_args.ratios is not None:
        config.ratios = [float(v) for v in split_values(cli_args.ratios)]
    if cli_args.crystal_structure is not None:
        config.crystal_structure = cli_args.crystal_structure
    if cli_args.lattice_constant is not None:
        config.lattice_constant = cli_args.lattice_constant
    if cli_args.hcp_c_over_a is not None:
        config.hcp_c_over_a = cli_args.hcp_c_over_a
    if cli_args.supercell is not None:
        config.supercell = list(cli_args.supercell)
    if cli_args.seed is not None:
        config.seed = cli_args.seed
    if cli_args.num_structures is not None:
        config.num_structures = cli_args.num_structures
    if cli_args.output_dir is not None:
        output_dir = cli_args.output_dir
        if not output_dir.is_absolute():
            output_dir = APP_ROOT / output_dir
        config.output_dir = output_dir
    return config


def default_config() -> argparse.Namespace:
    """Built-in defaults used when neither --input nor explicit flags fully specify a value."""
    return argparse.Namespace(
        elements=None,
        num_elements=None,
        candidate_elements=list(DEFAULT_CANDIDATE_ELEMENTS),
        ratios=[1.0],
        crystal_structure="bcc",
        lattice_constant=3.0,
        hcp_c_over_a=1.633,
        supercell=[4, 4, 4],
        seed=42,
        num_structures=1,
        output_dir=DEFAULT_OUTPUT_DIR,
    )


def validate_elements(symbols: Iterable[str]) -> list[str]:
    Element, _, _, _ = load_pymatgen()
    validated = []
    for symbol in symbols:
        Element(symbol)
        validated.append(symbol)
    if len(set(validated)) != len(validated):
        raise ValueError(f"Duplicate elements are not allowed: {validated}")
    return validated


def choose_elements(args: argparse.Namespace, rng: random.Random) -> list[str]:
    if args.elements:
        elements = validate_elements(args.elements)
        if args.num_elements is not None and args.num_elements != len(elements):
            raise ValueError(
                f"num_elements={args.num_elements} does not match the number of elements ({len(elements)})."
            )
        return elements

    if args.num_elements is None:
        raise ValueError("Provide either --elements or --num-elements.")

    candidates = validate_elements(args.candidate_elements)
    if args.num_elements <= 0:
        raise ValueError("num_elements must be positive.")
    if args.num_elements > len(candidates):
        raise ValueError(
            f"num_elements={args.num_elements} exceeds available candidates ({len(candidates)})."
        )
    return rng.sample(candidates, args.num_elements)


def normalize_ratios(ratios: list[float], num_elements: int) -> list[float]:
    if len(ratios) != num_elements:
        raise ValueError(f"Received {len(ratios)} ratios for {num_elements} elements.")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive.")
    total = sum(ratios)
    return [ratio / total for ratio in ratios]


def allocate_counts(fractions: list[float], total_sites: int) -> list[int]:
    if total_sites < len(fractions):
        raise ValueError(
            f"Total sites ({total_sites}) are fewer than the number of elements ({len(fractions)})."
        )

    raw_counts = [fraction * total_sites for fraction in fractions]
    counts = [int(count) for count in raw_counts]
    remainders = [raw - count for raw, count in zip(raw_counts, counts)]

    assigned = sum(counts)
    missing = total_sites - assigned
    order = sorted(range(len(fractions)), key=lambda idx: remainders[idx], reverse=True)
    for idx in order[:missing]:
        counts[idx] += 1

    zero_indices = [idx for idx, count in enumerate(counts) if count == 0]
    donor_indices = sorted(range(len(counts)), key=lambda idx: counts[idx], reverse=True)
    donor_cursor = 0
    for zero_idx in zero_indices:
        while donor_cursor < len(donor_indices) and counts[donor_indices[donor_cursor]] <= 1:
            donor_cursor += 1
        if donor_cursor >= len(donor_indices):
            raise ValueError("Unable to assign at least one site to every element with the chosen supercell.")
        donor_idx = donor_indices[donor_cursor]
        counts[donor_idx] -= 1
        counts[zero_idx] += 1

    return counts


def build_bcc_structure(lattice_constant: float, supercell: list[int]) -> "Structure":
    _, Lattice, Structure, _ = load_pymatgen()
    structure = Structure(
        Lattice.cubic(lattice_constant),
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    structure.make_supercell(supercell)
    return structure


def build_fcc_structure(lattice_constant: float, supercell: list[int]) -> "Structure":
    _, Lattice, Structure, _ = load_pymatgen()
    structure = Structure(
        Lattice.cubic(lattice_constant),
        ["Fe", "Fe", "Fe", "Fe"],
        [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ],
    )
    structure.make_supercell(supercell)
    return structure


def build_hcp_structure(lattice_constant: float, c_over_a: float, supercell: list[int]) -> "Structure":
    _, Lattice, Structure, _ = load_pymatgen()
    structure = Structure(
        Lattice.hexagonal(lattice_constant, lattice_constant * c_over_a),
        ["Fe", "Fe"],
        [
            [0, 0, 0],
            [2 / 3, 1 / 3, 0.5],
        ],
    )
    structure.make_supercell(supercell)
    return structure


def build_structure(crystal_structure: str, lattice_constant: float, supercell: list[int], hcp_c_over_a: float) -> "Structure":
    if lattice_constant <= 0:
        raise ValueError("lattice_constant must be positive.")
    if any(multiplier <= 0 for multiplier in supercell):
        raise ValueError("All supercell values must be positive integers.")

    builders = {
        "bcc": lambda: build_bcc_structure(lattice_constant, supercell),
        "fcc": lambda: build_fcc_structure(lattice_constant, supercell),
        "hcp": lambda: build_hcp_structure(lattice_constant, hcp_c_over_a, supercell),
    }
    try:
        return builders[crystal_structure]()
    except KeyError as exc:
        raise ValueError(f"Unsupported crystal_structure: {crystal_structure}") from exc


def assign_species(
    structure: "Structure", elements: list[str], counts: list[int], rng: random.Random
) -> "Structure":
    species = []
    for element, count in zip(elements, counts):
        species.extend([element] * count)

    if len(species) != len(structure):
        raise ValueError("Species count does not match the number of sites in the structure.")

    rng.shuffle(species)
    for idx, element in enumerate(species):
        structure.replace(idx, element)
    return structure


def format_composition(elements: list[str], counts: list[int], total_sites: int) -> str:
    parts = []
    for element, count in zip(elements, counts):
        fraction = count / total_sites
        parts.append(f"{element}:{count} ({fraction:.4f})")
    return ", ".join(parts)


def build_formula(elements: list[str], counts: list[int]) -> str:
    return "".join(f"{element}{count}" for element, count in zip(elements, counts))


def group_structure_by_elements(structure: "Structure", elements: list[str]) -> "Structure":
    _, _, Structure, _ = load_pymatgen()
    ordered_sites = []
    for element in elements:
        ordered_sites.extend(site for site in structure if site.specie.symbol == element)

    if len(ordered_sites) != len(structure):
        raise ValueError("Failed to group all sites by element before writing outputs.")

    return Structure.from_sites(ordered_sites)


def write_outputs(
    structure: "Structure", elements: list[str], cif_output: Path, poscar_output: Path
) -> None:
    _, _, _, Poscar = load_pymatgen()
    cif_output.parent.mkdir(parents=True, exist_ok=True)
    poscar_output.parent.mkdir(parents=True, exist_ok=True)
    grouped_structure = group_structure_by_elements(structure, elements)
    grouped_structure.to(filename=str(cif_output))
    Poscar(grouped_structure).write_file(str(poscar_output))


def build_output_paths(output_dir: Path, formula: str, index: int) -> tuple[Path, Path]:
    stem = f"{formula}_{index:03d}"
    return output_dir / f"{stem}.cif", output_dir / f"{stem}.POSCAR"


def main() -> None:
    parser = build_parser()
    try:
        cli_args = parser.parse_args()

        if cli_args.input is not None:
            config_map = parse_input_file(cli_args.input)
            config = build_config(config_map, cli_args.input)
            config = apply_cli_overrides(config, cli_args)
            print(f"Input file: {cli_args.input}")
        else:
            # Start from built-in defaults, then layer any explicit CLI flags on top.
            config = apply_cli_overrides(default_config(), cli_args)
    except KeyError as exc:
        parser.error(f"Missing required key in input file: {exc.args[0]}")
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Target structures: {config.num_structures}")
    print(f"Output directory: {config.output_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    fixed_elements = validate_elements(config.elements) if config.elements else None

    for index in range(1, config.num_structures + 1):
        rng = random.Random(config.seed + index - 1)
        elements = fixed_elements if fixed_elements else choose_elements(config, rng)
        fractions = normalize_ratios(config.ratios, len(elements))
        structure = build_structure(
            config.crystal_structure,
            config.lattice_constant,
            config.supercell,
            config.hcp_c_over_a,
        )
        total_sites = len(structure)
        counts = allocate_counts(fractions, total_sites)
        structure = assign_species(structure, elements, counts, rng)
        formula = build_formula(elements, counts)
        cif_output, poscar_output = build_output_paths(config.output_dir, formula, index)
        write_outputs(structure, elements, cif_output, poscar_output)
        summaries.append((index, formula, elements, counts, cif_output, poscar_output))

    first_summary = summaries[0]
    print(f"Crystal structure: {config.crystal_structure.upper()}")
    print(f"Supercell: {config.supercell[0]} x {config.supercell[1]} x {config.supercell[2]}")
    print(f"Total sites per structure: {len(build_structure(config.crystal_structure, config.lattice_constant, config.supercell, config.hcp_c_over_a))}")
    print(f"Example composition: {format_composition(first_summary[2], first_summary[3], total_sites)}")
    for index, formula, _, _, cif_output, poscar_output in summaries:
        print(f"[{index}] {formula}")
        print(f"  CIF: {cif_output}")
        print(f"  POSCAR: {poscar_output}")


if __name__ == "__main__":
    main()
