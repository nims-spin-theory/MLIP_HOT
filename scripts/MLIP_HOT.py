"""
MLIP-HOT Orchestrator

A unified CLI to run the three tasks in this package:
- Optimization (MLIP_optimize.py)
- Formation energy calculation (MLIP_form.py)
- Convex hull distance evaluation (MLIP_hull.py)

Configuration is provided via a YAML file.

Usage examples:
    Show help:
        python MLIP_HOT.py -h

    Run with pipeline config:
        python MLIP_HOT.py -c ../configs/pipeline.yaml

    Dry-run (print commands without executing):
        python MLIP_HOT.py -c ../configs/pipeline.yaml --dry-run --print-commands

YAML config schema (minimal):

    # Select one or more tasks; use "pipeline" to run sequentially
    task: pipeline  # or "optimize" | "form" | "hull"

    # Optional global MPI settings (per-task overrides supported)
    mpi_nproc: 4

    # Optional global model (used to select reference DBs; also defaults optimize.model)
    model: "MatterSim"

    optimize:
        input: "../example/example_data.csv"
        output: "../example/example_result"  # directory
        size: 1
        rank: 0
        print_col: null  # optional: print per-row identifier from this column
        strain: "0.0"  # scalar or matrix string
        primitive_cell_conversion: false
        fix_symmetry: false
        checkpoint_path: null
        # mpi_nproc: 4  # optional per-stage override

    form:
        database_elements: "../example/terminal_elements_mp.csv"
        composition_column_input: "optimized_formula"
        composition_column_elements: "composition"
        energy_column_input: "Energy (eV/atom)"
        energy_column_elements: "Energy (eV/atom)"
        out_column: "Formation Energy (eV/atom)"
    
    hull:
        database_convex: "../example/example_result/convex_hull_compounds_mp.csv"
        composition_column_input: "optimized_formula"
        composition_column_convex: "optimized_formula"
        formE_column_input: "Formation Energy (eV/atom)"
        formE_column_convex: "Formation Energy (eV/atom)"
        out_column: "Hull Distance (eV/atom)"
        # mpi_nproc: 4  # optional per-stage override

Notes:
- Paths in configs/CLI are resolved relative to the config file directory.
- MPI tasks (optimize, hull) use mpirun when nproc > 1.
- Optimize stage outputs:
    - Single run: structure_optimization_result.csv
    - Chunked runs: structure_optimization_result_<size>_<rank>.csv
 - Formation energy outputs (pipeline auto-naming):
     - Single run: formation_energy.csv
     - Chunked runs: formation_energy_<size>_<rank>.csv
 - Hull distance outputs (pipeline auto-naming):
     - Single run: hull_distance.csv
     - Chunked runs: hull_distance_<size>_<rank>.csv
"""

import argparse
import os
import sys
import shlex
import subprocess
from typing import Dict, Any, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable or "python"


def str2bool(v):
    """Parse common boolean strings for argparse."""
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean (true/false), got: {v}")


def resolve_path(p: Optional[str], base: Optional[str] = None) -> Optional[str]:
    """Resolve a path relative to a base directory.

    - Absolute paths are returned unchanged.
    - Relative paths are resolved against `base` if provided, otherwise CWD.
    """
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    base_dir = base or os.getcwd()
    return os.path.abspath(os.path.join(base_dir, p))


def get_reference_db_paths_for_model(model: Optional[str]) -> Dict[str, str]:
        """Return default reference DB CSV paths for a given model.

        Paths are resolved relative to this orchestrator's location:
            <MLIP_HOT.py dir>/../referenceDB/

        Naming convention (OQMD):
            - elements CSV:     OQMD_<model>_elements.csv
            - convex hull CSV:  OQMD_<model>_convex_hull.csv

        Only returns paths that exist on disk, allowing users to add new files
        later without code changes; model is used to select filenames.
        """
        if not model:
                return {}
        m = str(model).strip()
        base_ref_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "referenceDB"))
        elements_csv = os.path.join(base_ref_dir, f"OQMD_{m}_elements.csv")
        convex_csv = os.path.join(base_ref_dir, f"OQMD_{m}_convex_hull.csv")
        paths: Dict[str, str] = {}
        if os.path.exists(elements_csv):
                paths["elements"] = elements_csv
        if os.path.exists(convex_csv):
                paths["convex"] = convex_csv
        return paths


def compute_form_input_from_optimize_cfg(opt_cfg: Dict[str, Any], base_dir: str) -> Optional[str]:
    """Compute the formation energy input path based on optimize config.

    Mirrors current MLIP_optimize.py output naming:
      - size == 1: <output>/structure_optimization_result.csv
      - size > 1:  <output>/structure_optimization_result_<size>_<rank>.csv
    """
    out_dir = opt_cfg.get("output")
    if not out_dir:
        return None
    try:
        size = int(opt_cfg.get("size", 1))
    except Exception:
        size = 1
    try:
        rank = int(opt_cfg.get("rank", 0))
    except Exception:
        rank = 0
    out_dir_abs = resolve_path(out_dir, base=base_dir)
    base_name = "structure_optimization_result"
    if size == 1:
        return os.path.join(out_dir_abs, f"{base_name}.csv")
    return os.path.join(out_dir_abs, f"{base_name}_{size}_{rank}.csv")


def derive_form_output_from_optimize_cfg(opt_cfg: Dict[str, Any], base_dir: str) -> Optional[str]:
    """Derive formation energy output path based on optimize config.

    Naming:
      - size == 1 → <optimize.output>/formation_energy.csv
      - size > 1  → <optimize.output>/formation_energy_<size>_<rank>.csv
    """
    out_dir = opt_cfg.get("output")
    if not out_dir:
        return None
    try:
        size = int(opt_cfg.get("size", 1))
    except Exception:
        size = 1
    try:
        rank = int(opt_cfg.get("rank", 0))
    except Exception:
        rank = 0
    out_dir_abs = resolve_path(out_dir, base=base_dir)
    base_name = "formation_energy"
    if size == 1:
        return os.path.join(out_dir_abs, f"{base_name}.csv")
    return os.path.join(out_dir_abs, f"{base_name}_{size}_{rank}.csv")


def compute_hull_candidate_from_form_cfg(form_cfg: Dict[str, Any], base_dir: str) -> Optional[str]:
    """Resolve the formation energy output to be used as hull candidate input."""
    out_path = form_cfg.get("output")
    if not out_path:
        return None
    return resolve_path(out_path, base=base_dir)


def derive_hull_output_from_optimize_cfg(opt_cfg: Dict[str, Any], base_dir: str) -> Optional[str]:
    """Derive hull distance output path based on optimize config.

    Naming:
      - size == 1 → <optimize.output>/hull_distance.csv
      - size > 1  → <optimize.output>/hull_distance_<size>_<rank>.csv
    """
    out_dir = opt_cfg.get("output")
    if not out_dir:
        return None
    try:
        size = int(opt_cfg.get("size", 1))
    except Exception:
        size = 1
    try:
        rank = int(opt_cfg.get("rank", 0))
    except Exception:
        rank = 0
    out_dir_abs = resolve_path(out_dir, base=base_dir)
    base_name = "hull_distance"
    if size == 1:
        return os.path.join(out_dir_abs, f"{base_name}.csv")
    return os.path.join(out_dir_abs, f"{base_name}_{size}_{rank}.csv")


def load_config(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".yml", ".yaml"]:
        raise ValueError(f"Config must be a YAML file (.yml/.yaml), got: {ext}")
    try:
        import yaml  # lazy import
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping (YAML dict)")
    return data


def build_base_cmd(script_name: str, mpi_nproc: Optional[int]) -> List[str]:
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    nproc = int(mpi_nproc) if mpi_nproc is not None else 1
    if nproc and nproc > 1:
        return ["mpirun", "-n", str(nproc), PYTHON_EXE, script_path]
    return [PYTHON_EXE, script_path]


def run_cmd(cmd: List[str], print_commands: bool, dry_run: bool) -> int:
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    if print_commands:
        print(f"[CMD] {cmd_str}")
    if dry_run:
        return 0
    # Execute in the user's current working directory (do not force script dir)
    proc = subprocess.run(cmd)
    return proc.returncode


def run_optimize(cfg: Dict[str, Any], global_mpi_nproc: Optional[int], print_commands: bool, dry_run: bool, base_dir: str) -> int:
    stage_nproc = cfg.get("mpi_nproc", None)
    mpi_nproc = stage_nproc if stage_nproc is not None else global_mpi_nproc
    cmd = build_base_cmd("MLIP_optimize.py", mpi_nproc)
    # Required args
    if "input" not in cfg or "output" not in cfg:
        raise ValueError("optimize.input and optimize.output are required")
    # Resolve paths
    input_csv = resolve_path(cfg.get("input"), base=base_dir)
    output = resolve_path(cfg.get("output"), base=base_dir)
    checkpoint_path = resolve_path(cfg.get("checkpoint_path"), base=base_dir) if cfg.get("checkpoint_path") else None
    # Build args
    model_val = cfg.get("model")
    if not model_val:
        raise ValueError("Global model is required (provide top-level 'model' in YAML or --model)")
    cmd += [
        "-i", input_csv,
        "-m", str(model_val),
        "-o", output,
        "-s", str(cfg.get("size", 1)),
        "-r", str(cfg.get("rank", 0)),
    ]
    strain = cfg.get("strain")
    if strain is not None:
        cmd += ["--strain", str(strain)]
    if bool(cfg.get("primitive_cell_conversion", False)):
        cmd += ["--primitive-cell-conversion"]
    if bool(cfg.get("fix_symmetry", False)):
        cmd += ["--fix-symmetry"]
    print_col = cfg.get("print_col")
    if print_col:
        cmd += ["--print-col", str(print_col)]
    if checkpoint_path:
        cmd += ["--checkpoint_path", checkpoint_path]
    return run_cmd(cmd, print_commands, dry_run)


def run_form(cfg: Dict[str, Any], print_commands: bool, dry_run: bool, base_dir: str) -> int:
    cmd = build_base_cmd("MLIP_form.py", mpi_nproc=None)  # no MPI for form stage
    # Required args (output can be auto-derived)
    required = ["input", "database_elements"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"form.{k} is required")
    # Resolve paths
    input_path = resolve_path(cfg.get("input"), base=base_dir)
    terminal_path = resolve_path(cfg.get("database_elements"), base=base_dir)
    # Auto-derive output path when missing: <input_dir>/<basename>_formation_energy.csv
    output_cfg = cfg.get("output")
    if output_cfg:
        output_path = resolve_path(output_cfg, base=base_dir)
    else:
        in_base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.getcwd(), f"{in_base}_formation_energy.csv")
        if print_commands:
            print(f"[INFO] Form: Auto-set output = {output_path}")
        # also reflect in cfg for downstream wiring if needed
        cfg["output"] = output_path
    # Build args
    cmd += [
        "-i", input_path,
        "-e", terminal_path,
        "-o", output_path,
    ]
    # Optional columns
    if cfg.get("composition_column_input"):
        cmd += ["--composition_column_input", str(cfg.get("composition_column_input"))]
    if cfg.get("composition_column_elements"):
        cmd += ["--composition_column_elements", str(cfg.get("composition_column_elements"))]
    if cfg.get("energy_column_input"):
        cmd += ["--energy_column_input", str(cfg.get("energy_column_input"))]
    if cfg.get("energy_column_elements"):
        cmd += ["--energy_column_elements", str(cfg.get("energy_column_elements"))]
    if cfg.get("out_column"):
        cmd += ["--out_column", str(cfg.get("out_column"))]
    return run_cmd(cmd, print_commands, dry_run)


def run_hull(cfg: Dict[str, Any], global_mpi_nproc: Optional[int], print_commands: bool, dry_run: bool, base_dir: str) -> int:
    stage_nproc = cfg.get("mpi_nproc", None)
    mpi_nproc = stage_nproc if stage_nproc is not None else global_mpi_nproc
    cmd = build_base_cmd("MLIP_hull.py", mpi_nproc)
    # Required args (output can be auto-derived)
    required = ["input", "database_convex"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"hull.{k} is required")
    # Resolve paths
    candidate_path = resolve_path(cfg.get("input"), base=base_dir)
    convex_path = resolve_path(cfg.get("database_convex"), base=base_dir)
    # Auto-derive output path when missing: <input_dir>/<basename>_hull_distance.csv
    output_cfg = cfg.get("output")
    if output_cfg:
        output_path = resolve_path(output_cfg, base=base_dir)
    else:
        in_base = os.path.splitext(os.path.basename(candidate_path))[0]
        output_path = os.path.join(os.getcwd(), f"{in_base}_hull_distance.csv")
        if print_commands:
            print(f"[INFO] Hull: Auto-set output = {output_path}")
        # also reflect in cfg for downstream wiring if needed
        cfg["output"] = output_path
    # Build args
    cmd += [
        "-i", candidate_path,
        "-c", convex_path,
        "-o", output_path,
    ]
    # Optional columns
    if cfg.get("composition_column_input"):
        cmd += ["--composition_column_input", str(cfg.get("composition_column_input"))]
    if cfg.get("composition_column_convex"):
        cmd += ["--composition_column_convex", str(cfg.get("composition_column_convex"))]
    if cfg.get("formE_column_input"):
        cmd += ["--formE_column_input", str(cfg.get("formE_column_input"))]
    if cfg.get("formE_column_convex"):
        cmd += ["--formE_column_convex", str(cfg.get("formE_column_convex"))]
    if cfg.get("out_column"):
        cmd += ["--out_column", str(cfg.get("out_column"))]
    return run_cmd(cmd, print_commands, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLIP-HOT: Orchestrate optimization, formation energy, and hull distance tasks via YAML config",
    )
    parser.add_argument("-c", "--config", type=str, help="Path to YAML config file (relative paths are resolved against the config directory)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and skip execution")
    parser.add_argument("--print-commands", action="store_true", help="Print the exact commands executed")
    # Skip feature removed: tasks always run per `task` selection or full pipeline

    # Global overrides
    parser.add_argument("--task", choices=["pipeline", "optimize", "form", "hull"], default=None, help="Override task selection (defaults to config file value or 'pipeline')")
    parser.add_argument("--mpi_nproc", dest="mpi_nproc", type=int, help="Global MPI process count (MPI enabled when nproc>1)")
    parser.add_argument("--model", dest="global_model", type=str, default=None, help="Global model name used for reference DB selection; also default for optimize.model")

    # Optimize overrides
    opt_grp = parser.add_argument_group("optimize overrides")
    opt_grp.add_argument("--opt.input", "--optimize.input", dest="opt_input", type=str, help="Path to input CSV")
    opt_grp.add_argument("--opt.output", "--optimize.output", dest="opt_output", type=str, help="Output directory for optimized results")
    opt_grp.add_argument("--opt.size", "--optimize.size", dest="opt_size", type=int, help="Number of chunks for separate jobs")
    opt_grp.add_argument("--opt.rank", "--optimize.rank", dest="opt_rank", type=int, help="Chunk number for this job")
    opt_grp.add_argument("--opt.strain", "--optimize.strain", dest="opt_strain", type=str, help="Strain (scalar or 3x3 matrix string)")
    opt_grp.add_argument(
        "--opt.print_col",
        "--optimize.print_col",
        dest="opt_print_col",
        type=str,
        help="Forwarded to optimizer as --print-col (prints local row number + this column's value).",
    )
    opt_grp.add_argument("--opt.primitive_cell_conversion", "--optimize.primitive_cell_conversion", dest="opt_primitive_cell_conversion", action="store_true", help="Convert to primitive cell before optimization")
    opt_grp.add_argument(
        "--opt.fix_symmetry",
        "--optimize.fix_symmetry",
        dest="opt_fix_symmetry",
        type=str2bool,
        default=None,
        help="Enable/disable FixSymmetry constraint during optimization (True/False).",
    )
    opt_grp.add_argument("--opt.checkpoint_path", "--optimize.checkpoint_path", dest="opt_checkpoint_path", type=str, help="Checkpoint path for eqV2/esen models")
    opt_grp.add_argument("--opt.mpi_nproc", "--optimize.mpi_nproc", dest="opt_mpi_nproc", type=int, help="MPI process count for optimize stage (MPI enabled when nproc>1)")

    # Form overrides
    form_grp = parser.add_argument_group("formation overrides")
    form_grp.add_argument("--form.input", dest="form_input", type=str, help="Input optimized CSV path")
    form_grp.add_argument("--form.database_elements", dest="form_database_elements", type=str, help="Terminal elements CSV path")
    form_grp.add_argument("--form.output", dest="form_output", type=str, help="Output CSV path for formation energies")
    form_grp.add_argument("--form.composition_column_input", "--form.composition_column_compound", "--form.compound_column_compound", "--form.formula_column_compound", dest="form_composition_column_input", type=str, help="Input composition column name")
    form_grp.add_argument("--form.composition_column_elements", "--form.compound_column_elements", "--form.formula_column_elements", dest="form_composition_column_elements", type=str, help="Terminal element column name")
    form_grp.add_argument("--form.energy_column_input", "--form.energy_column_compound", dest="energy_column_input", type=str, help="Energy column name in input database")
    form_grp.add_argument("--form.energy_column_elements", dest="energy_column_elements", type=str, help="Energy column name in elements database")
    form_grp.add_argument("--form.out_column", dest="form_out_column", type=str, help="Output column name for formation energy")

    # Hull overrides
    hull_grp = parser.add_argument_group("hull overrides")
    hull_grp.add_argument("--hull.input", dest="hull_input", type=str, help="Candidate compounds CSV path")
    hull_grp.add_argument("--hull.database_convex", dest="hull_database_convex", type=str, help="Competing phases CSV path")
    hull_grp.add_argument("--hull.output", dest="hull_output", type=str, help="Output CSV path for hull distances")
    hull_grp.add_argument("--hull.composition_column_input", "--hull.composition_column_compound", "--hull.compound_column_compound", "--hull.formula_column_compound", dest="hull_composition_column_input", type=str, help="Candidate/input composition column name")
    hull_grp.add_argument("--hull.composition_column_convex", "--hull.compound_column_convex", "--hull.formula_column_convex", dest="hull_composition_column_convex", type=str, help="Convex composition column name")
    hull_grp.add_argument("--hull.formE_column_input", dest="hull_formE_column_input", type=str, help="Input formation energy column name")
    hull_grp.add_argument("--hull.formE_column_convex", dest="hull_formE_column_convex", type=str, help="Convex formation energy column name")
    hull_grp.add_argument("--hull.out_column", dest="hull_out_column", type=str, help="Output column name for hull distance")
    hull_grp.add_argument("--hull.mpi_nproc", dest="hull_mpi_nproc", type=int, help="MPI process count for hull stage (MPI enabled when nproc>1)")

    args = parser.parse_args()

    if args.config:
        config_path = resolve_path(args.config, base=os.getcwd())
        cfg = load_config(config_path)
        config_dir = os.path.dirname(config_path)
    else:
        cfg = {}
        config_dir = os.getcwd()
    # Merge global task override
    task = (args.task or str(cfg.get("task", "pipeline")).lower())

    # Global MPI merge: single key 'mpi_nproc'
    global_mpi_nproc = cfg.get("mpi_nproc", None)
    if args.mpi_nproc is not None:
        global_mpi_nproc = int(args.mpi_nproc)

    # Global model merge: single key 'model'
    global_model = cfg.get("model", None)
    if args.global_model is not None:
        global_model = str(args.global_model)

    # Normalize sections
    optimize_cfg = dict(cfg.get("optimize") or {})
    form_cfg = dict(cfg.get("form") or {})
    hull_cfg = dict(cfg.get("hull") or {})

    # Backwards compatibility: map older keys to composition_* if new keys missing
    if "composition_column_input" not in form_cfg:
        if "composition_column_compound" in form_cfg:
            form_cfg["composition_column_input"] = form_cfg["composition_column_compound"]
        elif "compound_column_compound" in form_cfg:
            form_cfg["composition_column_input"] = form_cfg["compound_column_compound"]
        elif "formula_column_compound" in form_cfg:
            form_cfg["composition_column_input"] = form_cfg["formula_column_compound"]
    if "composition_column_elements" not in form_cfg:
        if "compound_column_elements" in form_cfg:
            form_cfg["composition_column_elements"] = form_cfg["compound_column_elements"]
        elif "formula_column_elements" in form_cfg:
            form_cfg["composition_column_elements"] = form_cfg["formula_column_elements"]
    # Normalize energy column name for form stage
    if "energy_column_input" not in form_cfg and "energy_column_compound" in form_cfg:
        form_cfg["energy_column_input"] = form_cfg["energy_column_compound"]
    if "composition_column_input" not in hull_cfg:
        if "composition_column_compound" in hull_cfg:
            hull_cfg["composition_column_input"] = hull_cfg["composition_column_compound"]
        elif "compound_column_compound" in hull_cfg:
            hull_cfg["composition_column_input"] = hull_cfg["compound_column_compound"]
        elif "formula_column_compound" in hull_cfg:
            hull_cfg["composition_column_input"] = hull_cfg["formula_column_compound"]
    if "composition_column_convex" not in hull_cfg:
        if "compound_column_convex" in hull_cfg:
            hull_cfg["composition_column_convex"] = hull_cfg["compound_column_convex"]
        elif "formula_column_convex" in hull_cfg:
            hull_cfg["composition_column_convex"] = hull_cfg["formula_column_convex"]

    # No backward compatibility: expect 'formE_column_input' only

    # Auto-set reference DB paths based on global model only
    ref_paths: Dict[str, str] = {}
    if global_model:
        ref_paths = get_reference_db_paths_for_model(global_model)
    if ref_paths.get("elements") and not form_cfg.get("database_elements"):
        form_cfg["database_elements"] = ref_paths["elements"]
        # Print info for visibility when requested
        # Note: base_dir resolution is not needed; these are absolute paths
    if ref_paths.get("convex") and not hull_cfg.get("database_convex"):
        hull_cfg["database_convex"] = ref_paths["convex"]
        
    # Apply optimize overrides
    if args.opt_input:
        optimize_cfg["input"] = args.opt_input
    if args.opt_output:
        optimize_cfg["output"] = args.opt_output
    if args.opt_size is not None:
        optimize_cfg["size"] = int(args.opt_size)
    if args.opt_rank is not None:
        optimize_cfg["rank"] = int(args.opt_rank)
    if args.opt_strain is not None:
        optimize_cfg["strain"] = args.opt_strain
    if args.opt_print_col is not None:
        optimize_cfg["print_col"] = args.opt_print_col
    if args.opt_primitive_cell_conversion:
        optimize_cfg["primitive_cell_conversion"] = True
    if args.opt_fix_symmetry is not None:
        optimize_cfg["fix_symmetry"] = bool(args.opt_fix_symmetry)
    if args.opt_checkpoint_path:
        optimize_cfg["checkpoint_path"] = args.opt_checkpoint_path
    if args.opt_mpi_nproc is not None:
        optimize_cfg["mpi_nproc"] = int(args.opt_mpi_nproc)

    # Always use the global model for optimization
    if global_model:
        optimize_cfg["model"] = global_model
    else:
        # leave unset; run_optimize will enforce requirement if optimize task runs
        optimize_cfg.pop("model", None)

    # Apply form overrides
    if args.form_input:
        form_cfg["input"] = args.form_input
    if args.form_database_elements:
        form_cfg["database_elements"] = args.form_database_elements
    if args.form_output:
        form_cfg["output"] = args.form_output
    if args.form_composition_column_input:
        form_cfg["composition_column_input"] = args.form_composition_column_input
    if args.form_composition_column_elements:
        form_cfg["composition_column_elements"] = args.form_composition_column_elements
    if args.energy_column_input:   
        form_cfg["energy_column_input"] = args.energy_column_input
    if args.energy_column_elements:
        form_cfg["energy_column_elements"] = args.energy_column_elements
    if args.form_out_column:
        form_cfg["out_column"] = args.form_out_column

    # Apply hull overrides
    if args.hull_input:
        hull_cfg["input"] = args.hull_input
    if args.hull_database_convex:
        hull_cfg["database_convex"] = args.hull_database_convex
    if args.hull_output:
        hull_cfg["output"] = args.hull_output
    if args.hull_composition_column_input:
        hull_cfg["composition_column_input"] = args.hull_composition_column_input
    if args.hull_composition_column_convex:
        hull_cfg["composition_column_convex"] = args.hull_composition_column_convex
    if args.hull_formE_column_input:
        hull_cfg["formE_column_input"] = args.hull_formE_column_input
    if args.hull_formE_column_convex:
        hull_cfg["formE_column_convex"] = args.hull_formE_column_convex
    if args.hull_out_column:
        hull_cfg["out_column"] = args.hull_out_column
    if args.hull_mpi_nproc is not None:
        hull_cfg["mpi_nproc"] = int(args.hull_mpi_nproc)

    # Execute according to task
    if task == "optimize":
        print("\n" + "#" * 60)
        print("#" + " " * 58 + "#")
        print("#" + " RUNNING TASK: OPTIMIZE ".center(58) + "#")
        print("#" + " " * 58 + "#")
        print("#" * 60 + "\n")
        return run_optimize(optimize_cfg, global_mpi_nproc, args.print_commands, args.dry_run, base_dir=config_dir)

    if task == "form":
        print("\n" + "#" * 60)
        print("#" + " " * 58 + "#")
        print("#" + " RUNNING TASK: FORMATION ENERGY ".center(58) + "#")
        print("#" + " " * 58 + "#")
        print("#" * 60 + "\n")
        return run_form(form_cfg, args.print_commands, args.dry_run, base_dir=config_dir)

    if task == "hull":
        print("\n" + "#" * 60)
        print("#" + " " * 58 + "#")
        print("#" + " RUNNING TASK: CONVEX HULL DISTANCE ".center(58) + "#")
        print("#" + " " * 58 + "#")
        print("#" * 60 + "\n")
        return run_hull(hull_cfg, global_mpi_nproc, args.print_commands, args.dry_run, base_dir=config_dir)

    if task == "pipeline":
        # Enforce pipeline-only auto-wiring: input/output for form and hull cannot be set by user
        banned: List[str] = []
        if "input" in form_cfg:
            banned.append("form.input")
        if "output" in form_cfg:
            banned.append("form.output")
        if "input" in hull_cfg:
            banned.append("hull.input")
        if "output" in hull_cfg:
            banned.append("hull.output")
        if banned:
            raise ValueError(
                "In pipeline task, do not set these keys in config/CLI: "
                + ", ".join(banned)
                + ". They are auto-wired by the orchestrator."
            )

        # Run sequentially: optimize -> form -> hull
        print("\n" + "#" * 60)
        print("#" + " " * 58 + "#")
        print("#" + " RUNNING TASK: OPTIMIZE ".center(58) + "#")
        print("#" + " " * 58 + "#")
        print("#" * 60 + "\n")
        rc = run_optimize(optimize_cfg, global_mpi_nproc, args.print_commands, args.dry_run, base_dir=config_dir)
        if rc != 0:
            print(f"[ERROR] Optimize stage failed with exit code {rc}")
            return rc
        # Wire optimize output CSV into form input when possible
        # Determine optimize output path using the same naming logic
        form_input_csv = compute_form_input_from_optimize_cfg(optimize_cfg, base_dir=config_dir)
        if form_input_csv:
            form_cfg["input"] = form_input_csv
            # Derive formation energy output name consistently (size-aware)
            derived_form_out = derive_form_output_from_optimize_cfg(optimize_cfg, base_dir=config_dir)
            if derived_form_out:
                form_cfg["output"] = derived_form_out
            if args.print_commands:
                # Inform about auto-set database paths if they were applied
                if ref_paths.get("elements"):
                    print(f"[INFO] Auto-set form.database_elements = {form_cfg.get('database_elements')}")
                print(f"[INFO] Pipeline: Set form.input = {form_input_csv}")
                if derived_form_out:
                    print(f"[INFO] Pipeline: Set form.output = {derived_form_out}")
        else:
            if args.print_commands:
                print("[WARN] Pipeline: Could not determine optimize output; form.input unchanged")

        print("\n" + "#" * 60)
        print("#" + " " * 58 + "#")
        print("#" + " RUNNING TASK: FORMATION ENERGY ".center(58) + "#")
        print("#" + " " * 58 + "#")
        print("#" * 60 + "\n")
        rc = run_form(form_cfg, args.print_commands, args.dry_run, base_dir=config_dir)
        if rc != 0:
            print(f"[ERROR] Form stage failed with exit code {rc}")
            return rc
        # Wire form output into hull candidate input
        hull_candidate_csv = compute_hull_candidate_from_form_cfg(form_cfg, base_dir=config_dir)
        if hull_candidate_csv and not hull_cfg.get("input"):
            hull_cfg["input"] = hull_candidate_csv
            if args.print_commands:
                print(f"[INFO] Pipeline: Set hull.input = {hull_candidate_csv}")
        # Derive hull output name consistently (size-aware)
        derived_hull_out = derive_hull_output_from_optimize_cfg(optimize_cfg, base_dir=config_dir)
        if derived_hull_out:
            hull_cfg["output"] = derived_hull_out
            if args.print_commands:
                if ref_paths.get("convex"):
                    print(f"[INFO] Auto-set hull.database_convex = {hull_cfg.get('database_convex')}")
                print(f"[INFO] Pipeline: Set hull.output = {derived_hull_out}")

        print("\n" + "#" * 60)
        print("#" + " " * 58 + "#")
        print("#" + " RUNNING TASK: CONVEX HULL DISTANCE ".center(58) + "#")
        print("#" + " " * 58 + "#")
        print("#" * 60 + "\n")
        rc = run_hull(hull_cfg, global_mpi_nproc, args.print_commands, args.dry_run, base_dir=config_dir)
        if rc != 0:
            print(f"[ERROR] Hull stage failed with exit code {rc}")
            return rc
        print("[INFO] Pipeline completed successfully")
        return 0

    print(f"[ERROR] Unknown task: {task}. Use one of: pipeline, optimize, form, hull")
    return 1


if __name__ == "__main__":
    sys.exit(main())
