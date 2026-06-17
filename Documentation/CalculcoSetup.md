# Calculco Server Setup — ResearchModule

Operational reference for the ULCO Calculco HPC account used by this project.
Last verified on the cluster: **2026-06-17**.

## Platform Overview

**Calculco** (CalcULCO) is the scientific computing platform of Université du
Littoral Côte d'Opale (ULCO), managed by the Pôle Calcul Scientifique (SCoSI).
It provides shared CPU/GPU compute nodes, networked storage, licensed software,
and the **OAR** batch scheduler.

| Item | Value |
|------|-------|
| Login host (SSH) | `calculco.univ-littoral.fr` |
| File transfer host (large data) | `pcsdata.univ-littoral.fr` (preferred for heavy I/O) |
| Web documentation | [www-calculco.univ-littoral.fr](https://www-calculco.univ-littoral.fr) |
| Job scheduler | **OAR** (`oarsub`, not Slurm) |
| Lab | LISIC |
| OS on login node | Debian Linux, kernel 6.1 (x86_64) |

Official usage guide (login required): **Utilisation → Accéder** and related
sections on the Calculco website. English summary: **Calculco Quick Reference
(gb)** in the same menu.

## Account Identity

The **active SSH and web login username is `zahmed`**.

| Field | Value |
|-------|-------|
| Username | `zahmed` |
| Display name | Zaheer Ahmed |
| Lab | `lisic` |
| Email | `dev.zaheer.ahmad@gmail.com` |
| Shell | `/bin/bash` |

**SSH command (Windows PowerShell, macOS, Linux):**

```bash
ssh zahmed@calculco.univ-littoral.fr
```

**Web portal login:** same username and password at
[www-calculco.univ-littoral.fr](https://www-calculco.univ-littoral.fr) →
**Identifiant** = `zahmed`, **Mot de passe** = LDAP password.

## Storage Layout

Calculco exposes four storage roles. Only **home** and **protected data** are
nightly backed up.

| Role | Path | Quota / size | Backed up | Purge | Use for |
|------|------|--------------|-----------|-------|---------|
| **Home** | `/nfs/home/lisic/zahmed` | 102400 MB (~100 GB) per user | Yes | No | Code, scripts, configs, small results |
| **Protected data** | `/nfs/data/protected/lisic/zahmed` | Lab pool (~15 TB total, 2.1 TB used) | Yes | No | Licensed clinical data (MIMIC/eICU), derived artifacts |
| **Unprotected data** | `/nfs/data/unprotected/lisic/zahmed` | Lab pool (~167 TB total) | No | No | Large non-sensitive outputs |
| **Workdir (BeeGFS)** | `/workdir/lisic/zahmed` | Shared ~28 TB scratch | No | **180 days** | Temporary job I/O |
| **Node scratch** | `/scratch` on compute nodes | Local per node | No | **30 days** | Intra-node heavy I/O |

Login MOTD also shows convenience entries in home:

```text
data-protected   → symlink to protected data area
data-unprotected → symlink to unprotected data area
```

### Recommended ResearchModule layout

```text
/nfs/home/lisic/zahmed/
  ResearchModule/          # git clone of this repository
  scripts/                 # OAR job scripts

/nfs/data/protected/lisic/zahmed/ResearchModule/
  Dataset/                 # MIMIC-IV, MIMIC-IV-Note, eICU (licensed)
  derived/                 # cohorts, features, labels
  outputs/                 # final modeling artifacts

/nfs/data/unprotected/lisic/zahmed/
  public_outputs/          # non-sensitive large exports only

/workdir/lisic/zahmed/
  runs/                    # ephemeral per-job scratch
```

**Rules:**

- Never commit or expose raw clinical rows from `Dataset/`.
- Do not run heavy jobs on the login node; submit via OAR.
- Copy important results from `workdir` to `protected` after jobs finish.
- Use `purgelist` to check upcoming purge dates for scratch/workdir.

## Setup Completed (2026-06-17)

The following steps were completed on Calculco.

### 1. First SSH login

```bash
ssh zahmed@calculco.univ-littoral.fr
```

Login banner confirmed:

- Home usage: 0 MB / 102400 MB
- Purge: `/scratch` 30 days, `/workdir` 180 days
- Hostname: `calculco`

### 2. Directory structure created

```bash
mkdir -p ~/ResearchModule
mkdir -p ~/scripts
mkdir -p /workdir/lisic/zahmed/runs
mkdir -p /nfs/data/protected/lisic/zahmed/ResearchModule/{Dataset,derived,outputs}
mkdir -p /nfs/data/unprotected/lisic/zahmed/public_outputs
```

Resulting home listing:

```text
data-protected  data-unprotected  ResearchModule  scripts
```

Current working directory: `/nfs/home/lisic/zahmed`

### 3. Environment variables (session)

These were exported in the shell; persist them in `~/.bashrc` on the server:

```bash
export PROJECT_HOME=$HOME/ResearchModule
export DATA_PROTECTED=/nfs/data/protected/lisic/zahmed/ResearchModule
export WORK_SCRATCH=/workdir/lisic/zahmed
```

### 4. System inspection

| Check | Result |
|-------|--------|
| `whoami` | `zahmed` |
| `hostname` | `calculco` |
| `python3 --version` | Python 3.11.2 (system) |
| `which uv` | not installed yet |
| `which oarsub` | `/usr/bin/oarsub` |
| `nvidia-smi` on login node | not available (GPUs on compute nodes only) |

### 5. Disk space at setup time

| Mount | Size | Used | Avail | Use% |
|-------|------|------|-------|------|
| Home (`pcsdata:/home`) | 19T | 12T | 6.0T | 67% |
| Protected lab | 15T | 2.1T | 13T | 15% |
| Unprotected lab | 167T | 50T | 114T | 31% |
| Workdir (BeeGFS) | 28T | 64G | 28T | 1% |

### 6. Software modules available (sample)

Environment modules are managed with Lmod under `/nfs/opt/apps/modulefiles/`.

Notable modules for this project:

| Module | Notes |
|--------|-------|
| `conda/23.7` | Conda environments (Calculco tutorial path) |
| `fidle/pytorch-2.7-et-cie-gpu` | GPU PyTorch stack |
| `fidle/pytorch-2.7-et-cie-cpu` (default) | CPU PyTorch |
| `fidle/tensorflow-keras` | TensorFlow/Keras |
| `matlab/R2025b` | MATLAB (token/license rules apply) |
| `cuda/cuda-12.1` (default) | CUDA toolkit |
| `gcc/gcc-14.3.0` (default) | GCC compiler |
| `dmtcp/4.0.0` (default) | Checkpointing for long OAR jobs |

Discover more:

```bash
module avail
module spider python
module keyword pytorch
```

## Shell Configuration To Add on Server

Add to `~/.bashrc` on Calculco (if not already present):

```bash
# ResearchModule paths
export PROJECT_HOME=$HOME/ResearchModule
export DATA_PROTECTED=/nfs/data/protected/lisic/zahmed/ResearchModule
export WORK_SCRATCH=/workdir/lisic/zahmed

# Optional: quick navigation
alias cproj='cd $PROJECT_HOME'
alias cdata='cd $DATA_PROTECTED'
alias cscratch='cd $WORK_SCRATCH/runs'
```

Reload: `source ~/.bashrc`

## SSH Client Configuration (Local Windows PC)

Create or edit `%USERPROFILE%\.ssh\config`:

```text
Host calculco
    Hostname calculco.univ-littoral.fr
    User zahmed

Host pcsdata
    Hostname pcsdata.univ-littoral.fr
    User zahmed
```

Then connect with `ssh calculco` and transfer with `pcsdata` as the host alias.

## Next Step: Transfer ResearchModule

Clinical data must go only to **protected** storage. Code goes to **home**.

### Option A — Git (recommended for code)

On Calculco:

```bash
cd ~
git clone <repository-url> ResearchModule
cd ResearchModule
```

Do not clone licensed datasets into a public remote. Keep `Dataset/` out of Git.

### Option B — rsync from Windows (code)

From PowerShell on the local machine (with OpenSSH/rsync available):

```powershell
rsync -azuv --exclude Dataset --exclude .git --exclude __pycache__ `
  C:\ZaheerWork\Research\ResearchModule\ `
  zahmed@pcsdata.univ-littoral.fr:~/ResearchModule/
```

Use `pcsdata` for large transfers. The `--exclude Dataset` flag keeps clinical
data off the code sync; transfer datasets separately to protected paths.

### Option C — rsync clinical data to protected storage

```powershell
rsync -azuv --progress `
  C:\ZaheerWork\Research\ResearchModule\Dataset\ `
  zahmed@pcsdata.univ-littoral.fr:/nfs/data/protected/lisic/zahmed/ResearchModule/Dataset/
```

Only run this if PhysioNet license terms are satisfied and data stays on
Calculco protected storage.

### Option D — scp (small transfers only)

```powershell
scp -Cr C:\ZaheerWork\Research\ResearchModule\pyproject.toml `
  zahmed@calculco.univ-littoral.fr:~/ResearchModule/
```

Prefer `rsync` for anything large or resumable.

## After Transfer: Python Environment on Calculco

The repository declares Python 3.13 and `uv` locally. On Calculco, system Python
is 3.11.2. Choose one path:

1. **Conda module** (Calculco standard — see website tutorial *Environnements
   virtuels Python*):

   ```bash
   module load conda/23.7
   conda create -n researchmodule python=3.13 -y
   conda activate researchmodule
   ```

2. **Install uv in home** (matches repo `AGENTS.md`):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   cd ~/ResearchModule
   uv sync
   ```

Verify after setup:

```bash
uv run python -V
uv run pytest   # when tests exist in the tree
```

## First OAR Test Job (after code is present)

Create `~/scripts/test_job.sh`:

```bash
#!/bin/bash
#OAR -n test_hello
#OAR -l /nodes=1/core=1,walltime=00:10:00
#OAR -O ~/test_%jobid%.out
#OAR -E ~/test_%jobid%.err

echo "Job on $(hostname) at $(date)"
echo "OAR_WORKDIR=$OAR_WORKDIR"
```

Submit:

```bash
chmod +x ~/scripts/test_job.sh
oarsub -S ~/scripts/test_job.sh
oarstat
```

## Useful Commands Reference

```bash
# Identity and paths
whoami
pwd
quota -s
df -h ~
df -h /nfs/data/protected/lisic/zahmed
df -h /nfs/data/unprotected/lisic/zahmed
df -h /workdir/lisic/zahmed
du -sh /nfs/data/protected/lisic/zahmed/*

# Purge schedule
purgelist

# Modules
module avail
module load conda/23.7

# Jobs
oarsub -S ~/scripts/my_job.sh
oarstat
```

## Documentation Map (Calculco Website)

Read in this order after account setup:

1. **Accéder** — SSH, storage, transfers
2. **Calculco Quick Reference (gb)** — English cheat sheet
3. **Environnement logiciel** — modules and software
4. **Lancer un calcul** — OAR job submission
5. **Tutoriaux** — Python/conda, checkpointing, VS Code remote

External tutorial repo:
[gogs.univ-littoral.fr/PoleCalcul/tutoriaux_calculco](https://gogs.univ-littoral.fr/PoleCalcul/tutoriaux_calculco)

## Security Reminders

- Username is `zahmed`; do not confuse with `zrahmed1122` from the welcome email.
- Never commit passwords, SSH private keys, or raw clinical rows.
- Change the initial LDAP password when `passwd` or the web portal allows it.
- Prefer SSH keys (`ssh-keygen -t ed25519`) over password-only login.
- MIMIC/eICU data stays on **protected** storage only.

## Status Checklist

| Step | Status |
|------|--------|
| SSH login works | Done |
| Storage directories created | Done |
| System inspected (modules, OAR, disk) | Done |
| Environment variables defined | Done (add to `~/.bashrc`) |
| Password changed | Pending (`passwd` error — contact support) |
| SSH keys configured | Not started |
| ResearchModule code transferred | **Next** |
| Clinical data transferred to protected | Pending |
| Python/uv or conda environment | Pending |
| First OAR test job | Pending |
