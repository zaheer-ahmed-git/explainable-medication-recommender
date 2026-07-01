# Calculco Platform — ResearchModule

Generic ULCO Calculco (CalcULCO) notes for this project. **Account-specific paths,
usernames, and setup status live in gitignored local files** — not in this
document.

**Platform migration (2026):** CalcULCO is being moved to a new hardware platform
(phase II, summer 2026). During the transition, the login hostname may be
`ritchie.univ-littoral.fr` while compute nodes are renamed (`orval*` →
`chimay*`). The name `calculco` will remain as an alias once migration is
complete. See [Platform migration (2026)](#platform-migration-2026) below.

## Before You Run Commands

1. Confirm `PROJECT_HOME` and `DATASET_ROOT` are exported (see
   `Documentation/Environment.md`).
2. On Calculco, read `Documentation/CalculcoSetup.local.md` if present (create
   from `Documentation/CalculcoSetup.example.md`).

## Platform Overview

Calculco is the ULCO scientific computing platform (OAR scheduler, NFS storage,
environment modules).

| Item | Value |
|------|-------|
| Login host (stable name) | `calculco.univ-littoral.fr` |
| Login host (during migration) | `ritchie.univ-littoral.fr` — use when the old front-end has no compute nodes |
| Transfer host (large I/O) | `pcsdata.univ-littoral.fr` |
| Scheduler | **OAR** (`oarsub`, not Slurm) |
| Web docs | [www-calculco.univ-littoral.fr](https://www-calculco.univ-littoral.fr) |
| Transitional docs archive | [PCSBox essentials archive](https://pcsbox.univ-littoral.fr/d/a6d7b17d78694755974b/) |

## Storage Roles (typical)

| Role | Pattern | Use |
|------|---------|-----|
| Home | `/nfs/home/<lab>/<user>` | Code, configs, small results |
| Protected | `/nfs/data/protected/<lab>/<user>` | Licensed clinical data |
| Unprotected | `/nfs/data/unprotected/<lab>/<user>` | Large non-sensitive outputs |
| Workdir | `/workdir/<lab>/<user>` | Ephemeral job scratch (purged) |
| Node scratch | `/scratch` on compute nodes | Intra-node heavy I/O (purged) |

**Rules:** never commit raw clinical rows; do not run heavy jobs on the login node;
copy important results from scratch to protected storage after jobs finish.

## Recommended Split Layout

```text
$PROJECT_HOME/              # git clone (home)
$DATASET_ROOT/              # MIMIC/eICU (protected NFS)
$DATASET_ROOT/processed/    # cohorts, extracts, harmonized tables
$PROJECT_HOME/reports/      # aggregate manifests only
$WORK_SCRATCH/runs/         # per-job temp I/O
```

Set `PROJECT_HOME`, `DATASET_ROOT`, and `WORK_SCRATCH` via `~/.bashrc` or
`.env.calculco` (gitignored). See `.env.example`.

## Python on Calculco

System Python may differ from the repo's Python 3.13 target. Prefer `uv` in home:

```bash
cd "$PROJECT_HOME"
uv sync
uv run python -V
```

Conda module path is documented on the Calculco website (*Environnements virtuels
Python*) if needed.

## OAR Jobs (extraction and harmonization)

Version-controlled scripts: `scripts/calculco/`. Submit from `$PROJECT_HOME`.
Pass log paths at submit time so they stay machine-specific:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_mimic.sh"

oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_harmonize_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/harmonize.sh"
```

Repository CPU extraction scripts request `gpudevice='-1'` so jobs do not land on
GPU nodes during the migration. You do not need to pin a host name; request
cores and walltime only unless you have a specific hardware need.

Preflight (aggregate gates):

```bash
test -f "$DATASET_ROOT/processed/cohorts/cohort_stays.parquet"
test -f "$PROJECT_HOME/reports/quality_profile.json"
```

## Useful Commands

```bash
whoami
hostname
module avail
oarstatmon.py          # platform usage at a glance (preferred during migration)
oarstat
oarstat -u
purgelist
df -h "$HOME"
```

## Platform migration (2026)

CalcULCO admins are transferring nodes in batches from the legacy `orval*` names
to the new `chimay*` platform. Expect reduced capacity and changing hostnames
through summer 2026.

### Access during transition

| Phase | What to expect |
|-------|----------------|
| CPU batches | Chimay 1–20 online on the new front-end; older `orval04`–`orval17` nodes follow in further CPU batches |
| GPU batch (last) | GPU nodes (`orval39`–`orval43`) migrate after CPU batches; short windows where only `ritchie` has runnable nodes |
| After migration | `calculco.univ-littoral.fr` remains the familiar alias for `ritchie` |

SSH to **`ritchie.univ-littoral.fr`** when `calculco` cannot schedule jobs. First
login may prompt you to accept an updated host key — follow the on-screen
instructions.

Monitoring and refreshed web documentation are still being rolled out; the
[PCSBox archive](https://pcsbox.univ-littoral.fr/d/a6d7b17d78694755974b/) holds
essential usage notes from the legacy site.

### Node naming and OAR properties

**Do not** pin jobs with legacy host names such as `-p host='orval18'`. Nodes are
renamed (for example `orval34` → `chimay19`, `orval35` → `chimay20`).

Prefer homogeneous selectors when you need a specific CPU family:

| Node family | Example legacy names | New names (partial) | OAR property options |
|-------------|---------------------|---------------------|----------------------|
| Intel Xeon Gold 6348, 512 GB RAM | orval18, orval19, … | chimay 5–12 | `-p nodemodel='HPE_DL360'` or `-p cputype='Xeon_Gold_6348'` |
| AMD EPYC 7643, 512 GB RAM | orval27–orval33 | chimay 13–18 | `-p nodemodel='HPE_DL365'` or `-p cputype='EPYC_7643'` |

For **CPU-only** jobs (including this project's extraction scripts), always add:

```bash
-p gpudevice='-1'
```

You may also request resources by core count only; the scheduler will place jobs on
suitable 512 GB / AVX-512 nodes. The main rule from admins: do not omit
`gpudevice='-1'` on CPU work, or jobs may occupy GPU nodes that others need.

**Project-dedicated nodes:** `orval34` (RUPTURE / LISIC) and `orval35` (FAAR /
LPCA) became `chimay19` and `chimay20`. Project members have default access;
others may run there in best-effort mode (killable by project members).

### Planned summer outage

A CGU-wide power event for solar-panel grid connection is expected in the **first
half of July 2026** (exact date not yet announced). Even a short mains interruption
will trigger a **platform shutdown of at least 24 hours** for storage rewiring and
Infiniband cleanup. Plan long OAR jobs and data transfers accordingly.

### New accounts

Request access via **calculco-admins@liste.univ-littoral.fr** if you need an
account form during the transition.

## Account-Specific Details

Record usernames, quotas, transfer history, and checklist status only in
**`Documentation/CalculcoSetup.local.md`** (gitignored). Template:
`Documentation/CalculcoSetup.example.md`.
