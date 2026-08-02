# Runtime Environment — ULCO Calculco

Development runs on **ULCO Calculco** with code in home storage and licensed
clinical data on protected NFS. Resolve paths through `pipeline/config.py` and
environment variables; do not hard-code account-specific NFS paths in committed
source.

## Required Path Variables

Export these before pipeline CLIs or OAR jobs (typically in `~/.bashrc`,
`.env.calculco`, or `scripts/calculco/common.local.sh`):

| Variable | Purpose |
|----------|---------|
| `PROJECT_HOME` | Repository root (code and default reports) |
| `DATASET_ROOT` | Licensed clinical data root (protected NFS) |
| `DATA_PROTECTED` | Optional parent when `DATASET_ROOT` is unset; resolves to `$DATA_PROTECTED/Dataset` |
| `REPORTS_ROOT` | Optional aggregate report output directory |
| `WORK_SCRATCH` | Ephemeral job I/O on Calculco (OAR scripts; not read by `config.py`) |

Phase D protected GNN preparation additionally requires
`GNN_CROSSFIT_MIN_FREE_GIB`. It is not a path setting: it records the
operator-reviewed minimum free-space threshold for the five physical
fold-excluded cache trees. The preparation code also derives an estimate from
the full cache and requires the larger value. Keep this setting in the
gitignored GNN job env, not shared configuration.

See `.env.example` and `Documentation/CalculcoSetup.example.md` for templates.

## Gitignored Machine-Specific Files

These files are **not in Git** so account paths stay off shared clones. Copy
from the committed `*.example.*` templates on the server.

| File | Purpose |
|------|---------|
| `.env.calculco` | Path overrides for Calculco |
| `environment.calculco.sh` | Shell `export` snippets |
| `Documentation/CalculcoSetup.local.md` | Account paths, quotas, setup checklist |
| `Documentation/Environment.local.md` | Optional free-form machine notes for agents |
| `scripts/calculco/common.local.sh` | Optional extra exports for OAR jobs |

Committed references (safe for all clones):

- `.env.example` — path variable template
- `Documentation/CalculcoSetup.md` — generic Calculco platform notes (no account paths)
- `Documentation/CalculcoSetup.example.md` — template for `CalculcoSetup.local.md`

## Agent Checklist

1. Confirm `PROJECT_HOME` and `DATASET_ROOT` are exported (or read
   `Documentation/CalculcoSetup.local.md` if present).
2. Read `Documentation/CalculcoSetup.md` for platform and OAR conventions.
3. Resolve runtime paths only through `pipeline/config.py` and environment variables.
4. Run lightweight checks (`uv run pytest`, `uv run ruff check .`) on the login node;
   submit heavy extraction via OAR (`scripts/calculco/`).
