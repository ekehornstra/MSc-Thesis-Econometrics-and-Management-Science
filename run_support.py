"""run_support.py
Run support: environment switches and logging (THESIS_SMOKE, THESIS_MAX_WORKERS, THESIS_SHARD), worker-count policy, and the artifact store (save/load with provenance).
"""
import logging
import os
import sys
import hashlib
import json
import pickle
import platform
import time
from importlib import metadata
from pathlib import Path
import numpy as np


SMOKE = os.environ.get("THESIS_SMOKE", "0") == "1"
MAX_WORKERS = int(os.environ["THESIS_MAX_WORKERS"]) if os.environ.get("THESIS_MAX_WORKERS") else None
SHARD = os.environ.get("THESIS_SHARD")            # "i/n", 1-based, or None


def get_logger(name="thesis"):
    """Library modules log through this instead of print: INFO to stdout, one line per message.

    Parameters:
    name (str): Logger name. Default "thesis".

    Returns:
    logging.Logger: The logger, with a stdout handler attached on first use.
    """
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


# Smoke runs never touch the real artifacts: they default to their own folder.
ARTIFACT_DIR = Path(os.environ.get("THESIS_ARTIFACT_DIR", "./artifacts_smoke" if SMOKE else "./artifacts"))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Folder with the raw model output read by notebook 00 and checked by run_all.py --preflight.
# Default: the JASMIN group-workspace folder of the model output if it exists there, otherwise ./data.
_JASMIN_RAW = Path("/gws/pw/j07/greyzone/users/ekarl20/data/moudp135_sfc_d/std")
RAW_DATA_DIR = Path(os.environ.get("THESIS_RAW_DATA_DIR", str(_JASMIN_RAW if _JASMIN_RAW.is_dir() else "./data")))


def lsm_path(raw_dir=None):
    """Land-sea mask file inside the raw data folder: lsm.nc if present, else the dated file lsm_sfc_<date>.nc.

    Parameters:
    raw_dir (str or Path): Raw data folder. Default RAW_DATA_DIR.

    Returns:
    Path: The mask file (lsm.nc if nothing matches, so a missing file is reported under that name).
    """
    raw = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if (raw / "lsm.nc").is_file():
        return raw / "lsm.nc"
    found = sorted(raw.glob("lsm_sfc_*.nc"))
    return found[0] if found else raw / "lsm.nc"


def provenance():
    """Where a result came from: package versions, a SHA-1 of the project source, run flags, time.

    Returns:
    dict: saved_at, python, platform, packages, code_sha1 (first 12 hex characters) and smoke.
    """
    pkgs = {}
    for p in ("numpy", "scipy", "pandas", "torch", "pyvinecopulib", "arch", "statsmodels",
              "model-confidence-set"):
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    here = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for f in sorted(here.glob("*.py")):
        h.update(f.read_bytes())
    return dict(saved_at=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0],
                platform=platform.platform(), packages=pkgs, code_sha1=h.hexdigest()[:12],
                smoke=os.environ.get("THESIS_SMOKE", "0") == "1")


def save_artifact(name, **arrays_or_objects):
    """Save numpy arrays with np.savez and everything else with pickle.

    A `<name>_provenance.json` next to the files records package versions, a hash of the project
    source, the smoke flag and the time.

    Parameters:
    name (str): Base name used to construct the output file paths.
    **arrays_or_objects: Named values to save. Values that are numpy arrays
        are written to `<name>_arrays.npz`; everything else is written to
        `<name>_objects.pkl`.
    """
    arrs = {k: v for k, v in arrays_or_objects.items() if isinstance(v, np.ndarray)}
    other = {k: v for k, v in arrays_or_objects.items() if k not in arrs}
    if arrs:
        np.savez(ARTIFACT_DIR / f"{name}_arrays.npz", **arrs)
    if other:
        with open(ARTIFACT_DIR / f"{name}_objects.pkl", "wb") as f:
            pickle.dump(other, f)
    with open(ARTIFACT_DIR / f"{name}_provenance.json", "w") as f:
        json.dump(provenance(), f, indent=1)

def load_artifact(name):
    """Load a previously saved artifact, merging arrays and objects into one dict.

    Parameters:
    name (str): Base name used when the artifact was saved.

    Returns:
    dict: Combined dictionary of the arrays and objects saved under this name.
    """
    out = {}
    arr_path = ARTIFACT_DIR / f"{name}_arrays.npz"
    obj_path = ARTIFACT_DIR / f"{name}_objects.pkl"
    if arr_path.exists():
        out.update(dict(np.load(arr_path)))
    if obj_path.exists():
        with open(obj_path, "rb") as f:
            out.update(pickle.load(f))
    return out


def resolve_max_workers(n_tasks=None, reserve=2, floor=1):
    """Pick a worker-process count that respects the CPU allocation of the job.

    Call this fresh at (or just before) every point that opens a
    ProcessPoolExecutor / joblib.Parallel pool -- don't compute it once
    and reuse a stale number. A fresh call each time is what lets this
    track a changed allocation between cells and functions; there is no
    single "checkpoint" in the pipeline where checking once would be
    enough, since each parallel call site is independent.

    Why not os.cpu_count(): on a shared/SLURM node it reports the
    physical core count of the *node*, not what's actually been granted
    to this job's cgroup -- so "cpu_count() - reserve" can ask for MORE
    cores than you were given, which is the opposite of what this is
    for. os.sched_getaffinity(0) reflects the CPU set this process is
    actually bound to (SLURM/cgroup-aware) and is used first.
    SLURM_CPUS_PER_TASK / SLURM_JOB_CPUS_PER_NODE are checked as a
    fallback for the rare case sched_getaffinity is unavailable (e.g.
    non-Linux) or reports the full node anyway. os.cpu_count() is the
    last resort, not the primary source.

    This only sizes the *process* pool. If workers themselves call into
    numpy/PyTorch code that uses OpenMP/MKL threading internally, each
    worker can spawn its own extra threads on top -- N-2 processes x
    several BLAS threads each can still oversubscribe the node even
    though this function did its job. Set OMP_NUM_THREADS=1 /
    torch.set_num_threads(1) inside worker processes (e.g. via each
    ProcessPoolExecutor's initializer) if you haven't already, especially
    once max_workers stops being a small fixed number.

    Default: usable cores (affinity) minus `reserve`. THESIS_MAX_WORKERS=N, if set, caps this amount.

    Parameters:
    n_tasks (int): If given, cap the result at n_tasks -- no point
        opening more workers than there is work, and it keeps this from
        claiming cores it won't use. Optional.
    reserve (int): Cores to leave free for the login/head process and
        everyone else sharing the node. Default 2.
    floor (int): Never return fewer than this. Default 1.

    Returns:
    int: Number of worker processes to use right now.
    """
    available = None
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    if not available:
        for var in ('SLURM_CPUS_PER_TASK', 'SLURM_JOB_CPUS_PER_NODE'):
            val = os.environ.get(var)
            if val:
                try:
                    available = int(val.split(',')[0].split('(')[0])
                    break
                except ValueError:
                    continue
    if not available:
        available = os.cpu_count() or 1

    n = max(floor, available - reserve)
    explicit = os.environ.get('THESIS_MAX_WORKERS')
    if explicit:
        n = min(n, max(floor, int(explicit)))
    if n_tasks is not None:
        n = min(n, max(floor, int(n_tasks)))
    return n

def resolve_pool_workers(n_tasks=None, *, max_workers=None, worker_gib=None):
    """Allocate one independent process pool, bounded by CPU affinity, tasks and current RAM.

    This is a thin wrapper around resolve_max_workers and resolve_memory_workers.
    Pass max_workers only to set a LOWER explicit cap (e.g. for fragile preprocessing).
    The memory estimate is not a measurement: set a stage-specific worker_gib
    based on observed peak RSS if the default allocation is insufficient.

    Parameters:
    n_tasks (int): Number of tasks; caps the worker count. Optional.
    max_workers (int): Explicit upper limit on the workers. Optional.
    worker_gib (float): Estimated RAM per worker in GiB. Optional.

    Returns:
    int: Number of worker processes.
    """
    cpu = resolve_max_workers(n_tasks=n_tasks)
    if max_workers is not None:
        cpu = min(cpu, max(1, int(max_workers)))
    return resolve_memory_workers(cpu, worker_gib=worker_gib)


def resolve_memory_workers(cpu_workers, *, worker_gib=None, reserve_gib=None):
    """Conservative cgroup-aware process cap for independent worker pools.

    Reads the actual cgroup allocation and its *current* usage, so a Jupyter
    kernel that already occupies most RAM cannot start dozens of new workers.
    Defaults are an initial safety budget, not measurements of model peak RSS.
    Tune THESIS_NEURAL_WORKER_GIB after a representative one-worker run.
    On systems without a readable finite cgroup limit, use SLURM_MEM_PER_NODE
    when available; otherwise fail safe to one worker and warn.

    Parameters:
    cpu_workers (int): Worker count allowed by the CPU allocation (see resolve_max_workers).
    worker_gib (float): Estimated RAM per worker in GiB. Default: THESIS_NEURAL_WORKER_GIB, else 1.5. Optional.
    reserve_gib (float): RAM in GiB to leave free. Default: THESIS_MEMORY_RESERVE_GIB, else 3. Optional.

    Returns:
    int: min(cpu_workers, number of workers that fit in the memory headroom); 1 if the headroom cannot be read.

    Raises:
    ValueError: If the memory budget is not finite or not positive.
    MemoryError: If not even one worker fits in the headroom.
    """
    import math
    import warnings
    gb = 1024 ** 3
    per = float(worker_gib if worker_gib is not None else
                os.environ.get('THESIS_NEURAL_WORKER_GIB', '1.5'))
    reserve = float(reserve_gib if reserve_gib is not None else
                    os.environ.get('THESIS_MEMORY_RESERVE_GIB', '3'))
    if not (math.isfinite(per) and math.isfinite(reserve) and per > 0 and reserve >= 0):
        raise ValueError('Pool memory budget must be finite; per-worker > 0 and reserve >= 0')
    candidates = [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                  ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                   '/sys/fs/cgroup/memory/memory.usage_in_bytes')]
    limit = used = None
    for lpath, upath in candidates:
        try:
            raw = Path(lpath).read_text().strip()
            lim = int(raw) if raw != 'max' else None
            if lim is not None and lim < (1 << 60):
                limit, used = lim, int(Path(upath).read_text().strip())
                break
        except (OSError, ValueError):
            continue
    if limit is not None and used is not None:
        # memory.current counts page cache, which the kernel frees on demand. Count real process memory only.
        try:
            stat = {}
            for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines():
                key, val_ = line.split()
                stat[key] = int(val_)
            used = stat['anon'] + stat.get('kernel', 0) + stat.get('shmem', 0)
        except (OSError, KeyError, ValueError):
            pass   # keep the memory.current reading
    if limit is None:
        val = os.environ.get('SLURM_MEM_PER_NODE')
        if val:
            try:
                limit = int(val) * 1024 ** 2
            except ValueError:
                pass
        if limit is not None:
            try:
                import psutil
                used = max(0, limit - psutil.virtual_memory().available)
            except ImportError:
                used = None
    if limit is None or used is None:
        warnings.warn('Cannot identify current memory headroom; limiting pool to one worker.', RuntimeWarning)
        return 1
    headroom = (limit - used) / gb
    slots = math.floor((headroom - reserve) / per)
    if slots < 1:
        raise MemoryError(
            f'Pool not started: memory headroom {headroom:.2f} GiB, '
            f'reserve {reserve:.2f} GiB, estimated {per:.2f} GiB per worker. '
            'Stop orphaned workers / restart kernel or request more memory. '
            'Set THESIS_NEURAL_WORKER_GIB only after measuring peak worker RAM.')
    return min(int(cpu_workers), slots)
