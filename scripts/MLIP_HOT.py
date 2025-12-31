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
    mpi:
      enable: false
      nproc: 4

    optimize:
      database_csv: "../example/example_data.csv"
      model: "chgnet"
      output: "../example/example_result"  # directory
      size: 1
      rank: 0
      strain: "0.0"  # scalar or matrix string
      symmetrize: false
      checkpoint_path: null
      mpi: { enable: false, nproc: 4 }  # optional override

    form:
      input: "../example/example_result/example_data_result_mp.csv"
      database_terminal: "../example/terminal_elements_mp.csv"
      output: "../example/example_result/example_data_dft.csv"
      formula_column_compound: "optimized_formula"
      formula_column_terminal: "composition"
      energy_column: "Energy (eV/atom)"
      out_column: "Formation Energy (eV/atom)"

    hull:
      database_candidate: "../example/example_result/example_data_dft.csv"
      database_convex: "../example/example_result/convex_hull_compounds_mp.csv"
      output: "../example/example_result/example_data_result_mp.csv"
      formula_column_candidate: "optimized_formula"
      formula_column_convex: "optimized_formula"
      formE_column_candidate: "Formation Energy (eV/atom)"
      formE_column_convex: "Formation Energy (eV/atom)"
      out_column: "Hull Distance (eV/atom)"
      mpi: { enable: false, nproc: 4 }  # optional override

Notes:
- Paths are resolved relative to this script if not absolute.
- MPI tasks (optimize, hull) can be launched via mpirun when enabled.
"""

import argparse
import os
import sys
import shlex
import subprocess
from typing import Dict, Any, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable or "python"


def resolve_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(SCRIPT_DIR, p))


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


def build_base_cmd(script_name: str, mpi_cfg: Optional[Dict[str, Any]]) -> List[str]:
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    mpi_enabled = bool(mpi_cfg and mpi_cfg.get("enable"))
    nproc = int(mpi_cfg.get("nproc", 1)) if mpi_cfg else 1
    if mpi_enabled:
        return ["mpirun", "-n", str(nproc), PYTHON_EXE, script_path]
    return [PYTHON_EXE, script_path]


def run_cmd(cmd: List[str], print_commands: bool, dry_run: bool) -> int:
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    if print_commands:
        print(f"[CMD] {cmd_str}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return proc.returncode


def run_optimize(cfg: Dict[str, Any], global_mpi: Optional[Dict[str, Any]], print_commands: bool, dry_run: bool) -> int:
    mpi_cfg = cfg.get("mpi", global_mpi)
    cmd = build_base_cmd("MLIP_optimize.py", mpi_cfg)
    # Required args
    if "database_csv" not in cfg or "model" not in cfg or "output" not in cfg:
        raise ValueError("optimize.database_csv, optimize.model, and optimize.output are required")
    # Resolve paths
    database_csv = resolve_path(cfg.get("database_csv"))
    output = resolve_path(cfg.get("output"))
    checkpoint_path = resolve_path(cfg.get("checkpoint_path")) if cfg.get("checkpoint_path") else None
    # Build args
    cmd += [
        "-d", database_csv,
        "-m", str(cfg.get("model")),
        "-o", output,
        "-s", str(cfg.get("size", 1)),
        "-r", str(cfg.get("rank", 0)),
    ]
    strain = cfg.get("strain")
    if strain is not None:
        cmd += ["--strain", str(strain)]
    if bool(cfg.get("symmetrize", False)):
        cmd += ["--symmetrize"]
    if checkpoint_path:
        cmd += ["--checkpoint_path", checkpoint_path]
    return run_cmd(cmd, print_commands, dry_run)


def run_form(cfg: Dict[str, Any], print_commands: bool, dry_run: bool) -> int:
    cmd = build_base_cmd("MLIP_form.py", mpi_cfg=None)  # no MPI for form stage
    # Required args
    required = ["input", "database_terminal", "output"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"form.{k} is required")
    # Resolve paths
    input_path = resolve_path(cfg.get("input"))
    terminal_path = resolve_path(cfg.get("database_terminal"))
    output_path = resolve_path(cfg.get("output"))
    # Build args
    cmd += [
        "-i", input_path,
        "-t", terminal_path,
        "-o", output_path,
    ]
    # Optional columns
    if cfg.get("formula_column_compound"):
        cmd += ["--formula_column_compound", str(cfg.get("formula_column_compound"))]
    if cfg.get("formula_column_terminal"):
        cmd += ["--formula_column_terminal", str(cfg.get("formula_column_terminal"))]
    if cfg.get("energy_column"):
        cmd += ["--energy_column", str(cfg.get("energy_column"))]
    if cfg.get("out_column"):
        cmd += ["--out_column", str(cfg.get("out_column"))]
    return run_cmd(cmd, print_commands, dry_run)


def run_hull(cfg: Dict[str, Any], global_mpi: Optional[Dict[str, Any]], print_commands: bool, dry_run: bool) -> int:
    mpi_cfg = cfg.get("mpi", global_mpi)
    cmd = build_base_cmd("MLIP_hull.py", mpi_cfg)
    # Required args
    required = ["database_candidate", "database_convex", "output"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"hull.{k} is required")
    # Resolve paths
    candidate_path = resolve_path(cfg.get("database_candidate"))
    convex_path = resolve_path(cfg.get("database_convex"))
    output_path = resolve_path(cfg.get("output"))
    # Build args
    cmd += [
        "-d", candidate_path,
        "-c", convex_path,
        "-o", output_path,
    ]
    # Optional columns
    if cfg.get("formula_column_candidate"):
        cmd += ["--formula_column_candidate", str(cfg.get("formula_column_candidate"))]
    if cfg.get("formula_column_convex"):
        cmd += ["--formula_column_convex", str(cfg.get("formula_column_convex"))]
    if cfg.get("formE_column_candidate"):
        cmd += ["--formE_column_candidate", str(cfg.get("formE_column_candidate"))]
    if cfg.get("formE_column_convex"):
        cmd += ["--formE_column_convex", str(cfg.get("formE_column_convex"))]
    if cfg.get("out_column"):
        cmd += ["--out_column", str(cfg.get("out_column"))]
    return run_cmd(cmd, print_commands, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLIP-HOT: Orchestrate optimization, formation energy, and hull distance tasks via YAML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and skip execution")
    parser.add_argument("--print-commands", action="store_true", help="Print the exact commands executed")
    parser.add_argument("--skip", nargs="*", choices=["optimize", "form", "hull"], default=[], help="Skip selected tasks")

    # Global overrides
    parser.add_argument("--task", choices=["pipeline", "optimize", "form", "hull"], help="Override task selection")
    parser.add_argument("--mpi.enable", dest="mpi_enable", action="store_true", help="Enable global MPI (overridden by per-task mpi)")
    parser.add_argument("--mpi.nproc", dest="mpi_nproc", type=int, help="Global MPI process count")

    # Optimize overrides
    opt_grp = parser.add_argument_group("optimize overrides")
    opt_grp.add_argument("--opt.database_csv", "--optimize.database_csv", dest="opt_database_csv", type=str, help="Path to input database CSV")
    opt_grp.add_argument("--opt.model", "--optimize.model", dest="opt_model", type=str, help="ML-FF model name")
    opt_grp.add_argument("--opt.output", "--optimize.output", dest="opt_output", type=str, help="Output directory for optimized results")
    opt_grp.add_argument("--opt.size", "--optimize.size", dest="opt_size", type=int, help="Number of chunks for separate jobs")
    opt_grp.add_argument("--opt.rank", "--optimize.rank", dest="opt_rank", type=int, help="Chunk number for this job")
    opt_grp.add_argument("--opt.strain", "--optimize.strain", dest="opt_strain", type=str, help="Strain (scalar or 3x3 matrix string)")
    opt_grp.add_argument("--opt.symmetrize", "--optimize.symmetrize", dest="opt_symmetrize", action="store_true", help="Symmetrize to primitive before optimization")
    opt_grp.add_argument("--opt.checkpoint_path", "--optimize.checkpoint_path", dest="opt_checkpoint_path", type=str, help="Checkpoint path for eqV2/esen models")
    opt_grp.add_argument("--opt.mpi_enable", "--optimize.mpi_enable", dest="opt_mpi_enable", action="store_true", help="Enable MPI for optimize stage")
    opt_grp.add_argument("--opt.mpi_nproc", "--optimize.mpi_nproc", dest="opt_mpi_nproc", type=int, help="MPI process count for optimize stage")

    # Form overrides
    form_grp = parser.add_argument_group("formation overrides")
    form_grp.add_argument("--form.input", dest="form_input", type=str, help="Input optimized CSV path")
    form_grp.add_argument("--form.database_terminal", dest="form_database_terminal", type=str, help="Terminal elements CSV path")
    form_grp.add_argument("--form.output", dest="form_output", type=str, help="Output CSV path for formation energies")
    form_grp.add_argument("--form.formula_column_compound", dest="form_formula_column_compound", type=str, help="Compound formula column name")
    form_grp.add_argument("--form.formula_column_terminal", dest="form_formula_column_terminal", type=str, help="Terminal formula column name")
    form_grp.add_argument("--form.energy_column", dest="form_energy_column", type=str, help="Energy column name")
    form_grp.add_argument("--form.out_column", dest="form_out_column", type=str, help="Output column name for formation energy")

    # Hull overrides
    hull_grp = parser.add_argument_group("hull overrides")
    hull_grp.add_argument("--hull.database_candidate", dest="hull_database_candidate", type=str, help="Candidate compounds CSV path")
    hull_grp.add_argument("--hull.database_convex", dest="hull_database_convex", type=str, help="Competing phases CSV path")
    hull_grp.add_argument("--hull.output", dest="hull_output", type=str, help="Output CSV path for hull distances")
    hull_grp.add_argument("--hull.formula_column_candidate", dest="hull_formula_column_candidate", type=str, help="Candidate formula column name")
    hull_grp.add_argument("--hull.formula_column_convex", dest="hull_formula_column_convex", type=str, help="Convex formula column name")
    hull_grp.add_argument("--hull.formE_column_candidate", dest="hull_formE_column_candidate", type=str, help="Candidate formation energy column name")
    hull_grp.add_argument("--hull.formE_column_convex", dest="hull_formE_column_convex", type=str, help="Convex formation energy column name")
    hull_grp.add_argument("--hull.out_column", dest="hull_out_column", type=str, help="Output column name for hull distance")
    hull_grp.add_argument("--hull.mpi_enable", dest="hull_mpi_enable", action="store_true", help="Enable MPI for hull stage")
    hull_grp.add_argument("--hull.mpi_nproc", dest="hull_mpi_nproc", type=int, help="MPI process count for hull stage")

    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    # Merge global task override
    task = (args.task or str(cfg.get("task", "pipeline")).lower())

    # Global MPI merge
    global_mpi = cfg.get("mpi", None)
    if args.mpi_enable or args.mpi_nproc is not None:
        if not global_mpi:
            global_mpi = {"enable": False, "nproc": 1}
        if args.mpi_enable:
            global_mpi["enable"] = True
        if args.mpi_nproc is not None:
            global_mpi["nproc"] = int(args.mpi_nproc)

    # Normalize sections
    optimize_cfg = dict(cfg.get("optimize", {}))
    form_cfg = dict(cfg.get("form", {}))
    hull_cfg = dict(cfg.get("hull", {}))

    # Apply optimize overrides
    if args.opt_database_csv:
        optimize_cfg["database_csv"] = args.opt_database_csv
    if args.opt_model:
        optimize_cfg["model"] = args.opt_model
    if args.opt_output:
        optimize_cfg["output"] = args.opt_output
    if args.opt_size is not None:
        optimize_cfg["size"] = int(args.opt_size)
    if args.opt_rank is not None:
        optimize_cfg["rank"] = int(args.opt_rank)
    if args.opt_strain is not None:
        optimize_cfg["strain"] = args.opt_strain
    if args.opt_symmetrize:
        optimize_cfg["symmetrize"] = True
    if args.opt_checkpoint_path:
        optimize_cfg["checkpoint_path"] = args.opt_checkpoint_path
    if args.opt_mpi_enable or args.opt_mpi_nproc is not None:
        o_mpi = dict(optimize_cfg.get("mpi", {})) if optimize_cfg.get("mpi") else {"enable": False, "nproc": global_mpi["nproc"] if global_mpi else 1}
        if args.opt_mpi_enable:
            o_mpi["enable"] = True
        if args.opt_mpi_nproc is not None:
            o_mpi["nproc"] = int(args.opt_mpi_nproc)
        optimize_cfg["mpi"] = o_mpi

    # Apply form overrides
    if args.form_input:
        form_cfg["input"] = args.form_input
    if args.form_database_terminal:
        form_cfg["database_terminal"] = args.form_database_terminal
    if args.form_output:
        form_cfg["output"] = args.form_output
    if args.form_formula_column_compound:
        form_cfg["formula_column_compound"] = args.form_formula_column_compound
    if args.form_formula_column_terminal:
        form_cfg["formula_column_terminal"] = args.form_formula_column_terminal
    if args.form_energy_column:
        form_cfg["energy_column"] = args.form_energy_column
    if args.form_out_column:
        form_cfg["out_column"] = args.form_out_column

    # Apply hull overrides
    if args.hull_database_candidate:
        hull_cfg["database_candidate"] = args.hull_database_candidate
    if args.hull_database_convex:
        hull_cfg["database_convex"] = args.hull_database_convex
    if args.hull_output:
        hull_cfg["output"] = args.hull_output
    if args.hull_formula_column_candidate:
        hull_cfg["formula_column_candidate"] = args.hull_formula_column_candidate
    if args.hull_formula_column_convex:
        hull_cfg["formula_column_convex"] = args.hull_formula_column_convex
    if args.hull_formE_column_candidate:
        hull_cfg["formE_column_candidate"] = args.hull_formE_column_candidate
    if args.hull_formE_column_convex:
        hull_cfg["formE_column_convex"] = args.hull_formE_column_convex
    if args.hull_out_column:
        hull_cfg["out_column"] = args.hull_out_column
    if args.hull_mpi_enable or args.hull_mpi_nproc is not None:
        h_mpi = dict(hull_cfg.get("mpi", {})) if hull_cfg.get("mpi") else {"enable": False, "nproc": global_mpi["nproc"] if global_mpi else 1}
        if args.hull_mpi_enable:
            h_mpi["enable"] = True
        if args.hull_mpi_nproc is not None:
            h_mpi["nproc"] = int(args.hull_mpi_nproc)
        hull_cfg["mpi"] = h_mpi

    # Execute according to task
    if task == "optimize":
        if "optimize" in args.skip:
            print("[INFO] Optimize task skipped by --skip")
            return 0
        return run_optimize(optimize_cfg, global_mpi, args.print_commands, args.dry_run)

    if task == "form":
        if "form" in args.skip:
            print("[INFO] Form task skipped by --skip")
            return 0
        return run_form(form_cfg, args.print_commands, args.dry_run)

    if task == "hull":
        if "hull" in args.skip:
            print("[INFO] Hull task skipped by --skip")
            return 0
        return run_hull(hull_cfg, global_mpi, args.print_commands, args.dry_run)

    if task == "pipeline":
        # Run sequentially: optimize -> form -> hull (unless skipped)
        if "optimize" not in args.skip:
            rc = run_optimize(optimize_cfg, global_mpi, args.print_commands, args.dry_run)
            if rc != 0:
                print(f"[ERROR] Optimize stage failed with exit code {rc}")
                return rc
        else:
            print("[INFO] Optimize task skipped by --skip")

        if "form" not in args.skip:
            rc = run_form(form_cfg, args.print_commands, args.dry_run)
            if rc != 0:
                print(f"[ERROR] Form stage failed with exit code {rc}")
                return rc
        else:
            print("[INFO] Form task skipped by --skip")

        if "hull" not in args.skip:
            rc = run_hull(hull_cfg, global_mpi, args.print_commands, args.dry_run)
            if rc != 0:
                print(f"[ERROR] Hull stage failed with exit code {rc}")
                return rc
        else:
            print("[INFO] Hull task skipped by --skip")
        print("[INFO] Pipeline completed successfully")
        return 0

    print(f"[ERROR] Unknown task: {task}. Use one of: pipeline, optimize, form, hull")
    return 1


if __name__ == "__main__":
    sys.exit(main())
