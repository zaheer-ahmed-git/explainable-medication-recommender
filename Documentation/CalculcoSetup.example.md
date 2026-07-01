# Calculco Setup (Local) — Template

Copy to `Documentation/CalculcoSetup.local.md` on the Calculco server. That file
is gitignored so account-specific paths do not confuse agents on local machines.

```bash
cp Documentation/CalculcoSetup.example.md Documentation/CalculcoSetup.local.md
```

Fill in your account values. Do not commit the `.local.md` file.

---

# Calculco Server Setup — ResearchModule (local copy)

Operational reference for **your** ULCO Calculco account. Last verified: _YYYY-MM-DD_.

## Account Identity

| Field | Value |
|-------|-------|
| Username | `<your-calculco-username>` |
| Lab | `<lab>` |
| SSH (stable) | `ssh <username>@calculco.univ-littoral.fr` |
| SSH (during 2026 migration) | `ssh <username>@ritchie.univ-littoral.fr` if `calculco` has no compute nodes |
| Transfer host | `pcsdata.univ-littoral.fr` |

## Storage Layout (your paths)

| Role | Path |
|------|------|
| Home (code) | `/nfs/home/<lab>/<username>/ResearchModule` |
| Protected data | `/nfs/data/protected/<lab>/<username>/ResearchModule` |
| Dataset root | `$DATA_PROTECTED/Dataset` |
| Workdir scratch | `/workdir/<lab>/<username>` |

## Environment Variables (`~/.bashrc`)

```bash
export PROJECT_HOME=$HOME/ResearchModule
export DATA_PROTECTED=/nfs/data/protected/<lab>/<username>/ResearchModule
export DATASET_ROOT=$DATA_PROTECTED/Dataset
export WORK_SCRATCH=/workdir/<lab>/<username>
```

Optional per-repo file instead: copy `.env.example` to `.env.calculco` and source it.

## Setup Checklist

| Step | Status |
|------|--------|
| SSH login works | |
| Storage directories created | |
| `PROJECT_HOME` / `DATASET_ROOT` exported | |
| `uv sync` in `$PROJECT_HOME` | |
| First OAR test job | |
| Clinical data on protected NFS | |

## OAR Job Submission

From `$PROJECT_HOME`:

```bash
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_mimic.sh"
```

See `WORKFLOWS.md` and generic `Documentation/CalculcoSetup.md` for platform details.
