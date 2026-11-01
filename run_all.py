"""Run the notebooks headless, in order, and check the environment before a long run.

Every notebook is executed in its own process with nbconvert. The executed copy is written
next to the source as ``<name>_out.ipynb`` and the log goes to ``logs/``.

    python run_all.py --preflight                 environment and data check, no training
    python run_all.py                             00-08, then AAA01-AAA10 and AAA_summary (stops at the first failing main notebook)
    python run_all.py --synthetic                 the 10 synthetic notebooks (AAA01 ... AAA10), then AAA_summary
    python run_all.py --everything                the same list, resumable through the .done markers (use it for chained cluster jobs)
    python run_all.py --only AAA03 AAA08          chosen notebooks by prefix
    python run_all.py --stage neural              only the neural stage of notebook 04 (THESIS_SHARD allowed)

Environment variables: THESIS_STAGE (set here for notebook 04), THESIS_RESUME_ONLY (set for
the assemble stage), THESIS_SHARD (neural stage only), THESIS_ARTIFACT_DIR and
THESIS_MAX_WORKERS and THESIS_RAW_DATA_DIR (read in run_support). THESIS_KERNEL (optional) names the Jupyter kernel
that runs the notebooks; unset, every notebook uses the kernel stored in it.
"""
import argparse
import glob
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_support import ARTIFACT_DIR, RAW_DATA_DIR, lsm_path, resolve_max_workers

ORDER = ["00", "01", "02", "03", "04", "05", "06", "07", "08"]

# Cell indices of 04_continuous_conditional_copulas.ipynb (0-based, markdown cells count). Keep in sync with the
# notebook: 4 imports/settings, 6 load folds, 8 weight decay + Acar cache set-up, 9 Acar bandwidth tuning,
# 11 parametric stage, 13 Acar fits, 15 nonparametric fits, 17 neural batch. The stage notebook is checked against these markers below.
STAGE_CELLS = {
    "param": {0, 4, 6, 11},
    "acar": {0, 4, 6, 8, 9, 13},
    "nonparam": {0, 4, 6, 15},
    "neural": {0, 4, 6, 8, 17},
}
STAGE_MARKERS = {          # a substring that must appear in the selected code cells, so a shifted notebook fails loudly
    11: "run_param_stage", 13: "build_acar_jobs", 15: "build_nonparametric_jobs", 17: "build_fold_jobs", 9: "choose_acar_bandwidths", 8: "fixed_wd_by_model",
    6: "load_all_folds", 4: "run_support",
}


def preflight():
    """Check imports, worker settings and the raw data before a long run.

    Prints the version of every required package, the resolved worker counts and whether the
    raw data files of notebook 00 are present and readable. Does not start any model training.
    Exits the interpreter with status 1 when a package or a data file is missing.
    """
    print("Python:", sys.version.split()[0]); print("Host:", os.uname().nodename)
    print("Working directory:", Path.cwd())
    # Only the dependencies of the nbconvert workflow block the launch.
    packages = ["numpy", "scipy", "pandas", "torch", "xarray", "pyvinecopulib", "arch",
                "model_confidence_set", "nbconvert", "ipykernel", "matplotlib", "statsmodels",
                "numba", "tqdm", "joblib"]
    missing = []
    for name in packages:
        try:
            mod = importlib.import_module(name)
            print(f"OK      {name:<16} {getattr(mod, '__version__', 'installed')}")
        except Exception as exc:
            print(f"MISSING {name:<16} {type(exc).__name__}: {exc}")
            missing.append(name)
    print("THESIS_ARTIFACT_DIR:", os.getenv("THESIS_ARTIFACT_DIR", "./artifacts_smoke" if os.getenv("THESIS_SMOKE") == "1" else "./artifacts"))
    print("THESIS_MAX_WORKERS:", os.getenv("THESIS_MAX_WORKERS", "unset = automatic CPUs minus two"))
    print("CPUs in process affinity:", len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unavailable')
    print("Resolved workers for neural jobs (>=62 tasks):", resolve_max_workers(n_tasks=144))
    print("Resolved workers for Acar fits (28 tasks):", resolve_max_workers(n_tasks=28))
    print("Resolved workers for Acar CV (56 tasks):", resolve_max_workers(n_tasks=56))
    print("Notebooks:", len(list(Path('.').glob('*.ipynb'))))
    # Notebook 00 reads this folder (run_support.RAW_DATA_DIR). Fail early rather than after starting a long run.
    raw = RAW_DATA_DIR
    print('Notebook 00 raw data directory:', raw)
    lsm = lsm_path(raw)
    if not lsm.is_file():
        print('MISSING lsm.nc (or lsm_sfc_<date>.nc) at the notebook 00 data path')
        missing.append('raw-data:lsm.nc')
    else:
        try:
            import xarray as xr
            with xr.open_dataset(lsm):
                print('OK      can open', lsm.name, 'using xarray')
        except Exception as exc:
            print(f'MISSING functioning NetCDF backend: {type(exc).__name__}: {exc}')
            missing.append('raw-data:NetCDF-backend')
    for pat in ('precip_sfc_*.nc', 'uwnd_10m_*.nc', 'vwnd_10m_*.nc', 'st_sfc_*.nc'):
        if not any(raw.glob(pat)):
            print('MISSING expected data files:', pat)
            missing.append('raw-data:' + pat)
        else:
            print('OK      raw-data:', pat)
    if missing:
        print("Missing imports:", ', '.join(missing))
        sys.exit(1)


def find(prefix):
    """Find the source notebooks that belong to a prefix.

    Matches ``<prefix>.ipynb``, ``<prefix>_*.ipynb`` and ``<prefix>[a-z]_*.ipynb`` in the
    working directory. Executed copies (``_out``), shard copies and stage notebooks are ignored.

    Parameters:
        prefix (str): notebook prefix such as ``"04"`` or ``"AAA05"``.

    Returns:
        list[str]: matching notebook paths, empty if none (a warning is printed then).
    """
    hits = sorted(glob.glob(f"{prefix}.ipynb")) + sorted(glob.glob(f"{prefix}_*.ipynb")) + sorted(glob.glob(f"{prefix}[a-z]_*.ipynb"))
    hits = [h for h in hits if "_stage_" not in Path(h).stem and not Path(h).stem.endswith("_out")
            and "_out_shard_" not in Path(h).stem]
    if not hits:
        print(f"WARNING: no notebook found for prefix {prefix!r} in {Path.cwd()}", flush=True)
    return hits


def prepare_stage_notebook(nb, stage):
    """Write a self-contained notebook that holds only the cells of one stage of notebook 04.

    The selected cells are those listed in STAGE_CELLS. Each is checked against STAGE_MARKERS
    first, so an edit that shifted the cell indices fails loudly instead of running the wrong cells.

    Parameters:
        nb (str): path of notebook 04.
        stage (str): key of STAGE_CELLS (``"param"``, ``"acar"``, ``"nonparam"`` or ``"neural"``).

    Returns:
        str: path of the written ``<name>_stage_<stage>[_shard_<id>].ipynb``.

    Raises:
        RuntimeError: if a selected cell does not contain its marker.
    """
    import nbformat
    src = nbformat.read(nb, as_version=4)
    out = nbformat.v4.new_notebook(metadata=src.metadata)
    for i in STAGE_CELLS[stage]:
        if i in STAGE_MARKERS and STAGE_MARKERS[i] not in src.cells[i].source:
            raise RuntimeError(f"STAGE_CELLS is out of date: cell {i} of {nb} does not contain {STAGE_MARKERS[i]!r}. "
                               "Update STAGE_CELLS/STAGE_MARKERS in run_all.py after editing the notebook.")
    out.cells = [cell for i, cell in enumerate(src.cells) if i in STAGE_CELLS[stage]]
    shard_id = os.getenv("THESIS_SHARD", "").replace("/", "_of_")
    suffix = f"_shard_{shard_id}" if shard_id else ""
    dest = Path(nb).with_name(Path(nb).stem + f"_stage_{stage}{suffix}.ipynb")
    nbformat.write(out, dest)
    return str(dest)


def run(nb, stage=None):
    """Execute one notebook headless with nbconvert.

    The executed copy is ``<name>_out.ipynb`` and the log goes to ``logs/``. With a SLURM shard
    set in THESIS_SHARD both names carry the shard id, so concurrent tasks never write to the
    same file. A shard of the neural stage that ends in IncompleteBatch counts as success.

    Parameters:
        nb (str): notebook path.
        stage (str | None): stage of notebook 04 to run (``"param"``, ``"acar"``, ``"nonparam"``, ``"neural"``,
            ``"assemble"``); sets THESIS_STAGE, and THESIS_RESUME_ONLY for ``"assemble"``.

    Returns:
        bool: True if the notebook ran to the end, False otherwise (the log tail is printed).
    """
    if stage and Path(nb).name.startswith("04_") and stage in STAGE_CELLS:
        nb = prepare_stage_notebook(nb, stage)
    out = nb.replace(".ipynb", "_out.ipynb")
    shard_label = os.getenv("THESIS_SHARD", "").replace("/", "_of_")
    if shard_label:
        out = out.replace(".ipynb", f"_shard_{shard_label}.ipynb")
    print(f"=== {nb} stage={stage or 'default'} shard={shard_label or 'none'}", flush=True)
    Path("logs").mkdir(exist_ok=True)
    logname = Path("logs") / (Path(nb).stem + (f"_shard_{shard_label}" if shard_label else "")
                                 + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    env = os.environ.copy()
    if stage:
        env["THESIS_STAGE"] = stage
    if stage == "assemble":
        env["THESIS_RESUME_ONLY"] = "1"
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=-1", nb, "--output", out]
    if os.getenv("THESIS_KERNEL"):
        cmd.insert(-3, f"--ExecutePreprocessor.kernel_name={os.environ['THESIS_KERNEL']}")
    print(f"    notebook log: {logname}", flush=True)
    # The log goes to disk, so a long run does not hold its output in memory.
    with logname.open("w", encoding="utf-8") as handle:
        r = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    if r.returncode == 0:
        return True
    tail = logname.read_text(encoding="utf-8", errors="replace")[-6000:]
    if stage == "neural" and os.environ.get("THESIS_SHARD") and "IncompleteBatch" in tail:
        print("    neural shard completed its assignment (IncompleteBatch), continuing")
        return True
    print(tail, flush=True)
    return False


def _marker(nb):
    """Return the path of the status record of a notebook, creating ``<artifact dir>/.done/``.

    Parameters:
        nb (str): notebook path; only its stem is used.

    Returns:
        pathlib.Path: ``<artifact dir>/.done/<stem>.status``.
    """
    d = Path(ARTIFACT_DIR) / ".done"
    d.mkdir(parents=True, exist_ok=True)
    return d / (Path(nb).stem + ".status")


def synthetic_prefixes():
    """List the prefixes of the synthetic notebooks (AAA01, AAA02, ...) found in the working directory.

    Returns:
        list[str]: sorted prefixes, executed copies and stage notebooks excluded.
    """
    return sorted({Path(p).name.split("_")[0] for p in glob.glob("AAA*_synthetic_*.ipynb")
                   if "_out" not in p and "_stage_" not in p})


def _run_fingerprint(nb):
    """Digest the notebook, Python sources and execution settings used by this run.

    Parameters:
    nb (str): Notebook path.

    Returns:
    str: Hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    paths = sorted(Path('.').glob('*.py')) + [Path(nb)]
    if Path(nb).stem[:2] not in ("00", "01", "02", "03"):
        calibration = Path(ARTIFACT_DIR) / "margin_penalty_calibrated.json"
        if calibration.is_file():
            paths.append(calibration)
    for p in paths:
        h.update(str(p).encode())
        h.update(p.read_bytes())
    for name in ("THESIS_SMOKE", "THESIS_MARGIN_PENALTY", "THESIS_CAL_DELTA", "THESIS_CAL_LAMBDAS", "THESIS_CAL_WDS",
                 "THESIS_SYNTH_N", "THESIS_ARTIFACT_DIR", "THESIS_RAW_DATA_DIR", "THESIS_KERNEL"):
        h.update(f"{name}={os.getenv(name, '')}".encode())
    return h.hexdigest()


def run_everything(fresh=False):
    """Run notebooks 00 to 08 in order (03 is the penalty calibration, which the later notebooks read), then the synthetic notebooks AAA01-AAA10 and AAA_summary.

    A failure in the main pipeline stops the synthetic notebooks too. Within the synthetic study,
    failed notebooks are recorded and other designs can still run if the main pipeline passed.

    A successful notebook is skipped on the next call only when its source/code/settings digest
    still matches. Failed notebooks are retried. Individual model fits resume from their batch files.

    Parameters:
        fresh (bool): ignore and clear the records of earlier calls. Use it for the first call
            after any code change.

    Returns:
        list[tuple[str, str]]: ``(notebook, status)`` for every notebook found.
    """
    if fresh:
        for m in _marker("x").parent.glob("*.status"):
            m.unlink()

    def go(nb):
        """Run one notebook unless an unchanged successful run is recorded, and record the outcome.

        Parameters:
        nb (str): Notebook path.

        Returns:
        tuple: (success flag, status text).
        """
        m = _marker(nb)
        fingerprint = _run_fingerprint(nb)
        if m.exists():
            try:
                old = json.loads(m.read_text())
            except (ValueError, TypeError):
                old = {}
            if old.get("status") == "ok" and old.get("fingerprint") == fingerprint:
                print(f"=== {nb}: unchanged successful run, skipped", flush=True)
                return True, "ok (earlier call)"
            print(f"=== {nb}: changed or previously failed; rerunning", flush=True)
        try:
            ok = run(nb)
        except Exception as exc:
            print(f"runner error in {nb}: {exc}", flush=True)
            ok = False
        m.write_text(json.dumps({"status": "ok" if ok else "FAILED", "fingerprint": fingerprint}))
        return ok, "ok" if ok else "FAILED"

    results, main_ok = [], True
    for p in ORDER:
        for nb in find(p):
            if not main_ok:
                results.append((nb, "skipped (main pipeline stopped earlier)")); continue
            ok, st = go(nb)
            results.append((nb, st))
            main_ok = main_ok and ok
    if not main_ok:
        print("\nMain pipeline failed; synthetic notebooks are not started.", flush=True)
    else:
        for p in synthetic_prefixes():
            for nb in find(p):
                results.append((nb, go(nb)[1]))
        if all(st.startswith("ok") for nb, st in results):
            for nb in find("AAA_summary"):
                results.append((nb, go(nb)[1]))
        else:
            print("\nA synthetic design failed; AAA_summary is not started.", flush=True)
    print("\nSummary", flush=True)
    for nb, st in results:
        print(f"  {st:<45} {nb}", flush=True)
    return results


def main():
    """Parse the command line and run the requested notebooks.

    Exits with status 1 if a notebook fails or an option combination is invalid.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="check environment and raw data, then exit")
    ap.add_argument("--only", nargs="*", help="notebook prefixes, e.g. 03 04 AAA05")
    ap.add_argument("--synthetic", action="store_true", help="run the 10 synthetic notebooks and AAA_summary")
    ap.add_argument("--everything", action="store_true", help="resumable run of everything (00-08, then all AAA notebooks)")
    ap.add_argument("--fresh", action="store_true", help="with --everything: ignore and clear the records of earlier calls")
    ap.add_argument("--stage", choices=("param", "acar", "nonparam", "neural", "assemble"),
                    help="Run only this stage of notebook 04; only neural supports THESIS_SHARD")
    a = ap.parse_args()
    if a.preflight:
        preflight()
        sys.exit(0)
    if a.everything:
        res = run_everything(fresh=a.fresh)
        sys.exit(0 if all(s.startswith("ok") for _, s in res) else 1)
    if os.environ.get("THESIS_SHARD") and a.stage != "neural":
        ap.error("THESIS_SHARD is only allowed with --stage neural; never shard param, Acar, nonparametric or assembly")
    if a.stage and (a.only or a.synthetic):
        ap.error("--stage cannot be combined with --only or --synthetic")
    if not a.stage and not a.only and os.environ.get("THESIS_SHARD"):
        ap.error("sharded execution requires --stage neural")
    if a.stage:
        prefixes = ["04"]
    elif a.only:
        prefixes = a.only
    elif a.synthetic:
        prefixes = synthetic_prefixes() + ["AAA_summary"]
    else:
        prefixes = ORDER + synthetic_prefixes() + ["AAA_summary"]
    for p in prefixes:
        for nb in find(p):
            if not run(nb, stage=a.stage):
                sys.exit(f"failed: {nb}")


if __name__ == "__main__":
    main()
