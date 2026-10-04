# Calculco notes 

Replace `<user>` and `<lab>` with your login and lab.

---

## What Calculco is

Calculco (also called CalcULCO) is Université du Littoral Côte d'Opale's shared
scientific computing platform. SCoSI / Pôle Calcul Scientifique run it.

You get:

- a login machine (the front-end / "frontal")
- CPU and GPU compute nodes you cannot SSH to directly
- NFS home and data storage, plus scratch space
- environment modules for compilers, CUDA, conda, MATLAB, etc.
- **OAR** as the batch scheduler (`oarsub`, `oarstat`, `oardel`)

The cluster currently has about 35 compute nodes (`chimay01` … `chimay35`),
roughly 2300 cores, and GPU machines on `chimay31`–`chimay35` (A100, H100, and
a GH200 node). Names used to be `orval*`. Those are gone. Do not pin
`host='orval18'` or similar.

---

## How you log in

The real front-end is `ritchie.univ-littoral.fr`. `calculco.univ-littoral.fr`
is the familiar alias. During the 2026 hardware move, if `calculco` cannot
schedule jobs, use `ritchie`.

```bash
ssh <user>@ritchie.univ-littoral.fr
# or
ssh <user>@calculco.univ-littoral.fr
```

Windows: OpenSSH in PowerShell, WSL, Bitvise, or PuTTY (port 22). Same
username/password as the web site.

First login: `passwd` to change the random LDAP password they mailed you. Then
put an SSH key on the machine so you stop typing it:

```bash
# on your laptop
ssh-keygen -t ed25519 -o -a 64
ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@ritchie.univ-littoral.fr
```

Optional `~/.ssh/config` on your laptop:

```text
Host calculco
    Hostname ritchie.univ-littoral.fr
    User <user>

Host pcsdata
    Hostname pcsdata.univ-littoral.fr
    User <user>
```

Then `ssh calculco` is enough.

Web docs (same login): [www-calculco.univ-littoral.fr](https://www-calculco.univ-littoral.fr).
Start with **Utilisation → Accéder** and **Lancer un calcul**.

---

## The one rule that matters

**Do not run heavy work on the login node.** Compile there if you must. Tests
that finish in seconds are fine. Anything that hammers CPU, RAM, disk, or GPU
goes through OAR.

You also cannot `ssh chimay31` yourself. OAR places you on a node. From the
front-end, attach to a running job with `oarsub -C <jobid>` or
`OAR_JOB_ID=<jobid> oarsh chimayXX`.

---

## Disk layout

| Where | Path | Backed up? | Use |
|-------|------|------------|-----|
| Home | `/nfs/home/<lab>/<user>` | yes | code, configs, small results. Quota ~100 GB (`quota -s`) |
| Protected data | `/nfs/data/protected/<lab>/<user>` | yes (nightly) | licensed / irreplaceable data and final results |
| Unprotected data | `/nfs/data/unprotected/<lab>/<user>` | **no** | large non-sensitive outputs |
| Workdir (BeeGFS) | `/workdir/<lab>/<user>` | **no**, purged | shared scratch during jobs |
| Node scratch | `/scratch` on the compute node | **no**, purged | local heavy I/O. Not shared across nodes |

Home has convenience symlinks `data-protected` and `data-unprotected`.

Protected is for keeping things, not as a working directory while a job is
rewriting huge files. Put bursty I/O on `/workdir` or `/scratch`, then copy
what you need to keep.

`purgelist` tells you when scratch and workdir will be wiped. Do not treat
those trees as storage.

Licensed clinical data stays on **protected**. Do not copy it to a laptop, to
Git, or into chat logs.

---

## Copying files

Large transfers go through `pcsdata.univ-littoral.fr`, not the login node.

```bash
# code to home
rsync -azuv --exclude .git ./myproject/ <user>@pcsdata.univ-littoral.fr:~/myproject/

# recover results
rsync -azuv <user>@pcsdata.univ-littoral.fr:/nfs/data/unprotected/<lab>/<user>/out/ ./out/
```

`scp` is fine for a few small files. `rsync` for anything that might drop
mid-copy. On Windows, Bitvise / WinSCP / FileZilla also work over SSH.

---

## Software on the cluster

List modules:

```bash
module avail
module spider python
module keyword pytorch
```

Load what you need **on the compute node**, after OAR has placed you. Modules
loaded on the front-end do not follow you into `oarsub -I`.

In a **batch** script, source the module environment first:

```bash
source /nfs/opt/env/env.sh
module load cuda/cuda-12.1
```

Python: system Python on the login node is older than what many projects want.
Two common setups:

1. Calculco conda (`module load conda/23.7`, then a conda env).
2. `uv` installed in your home (`~/.local/bin/uv`). That is what we use on
   ResearchModule.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
cd ~/your-repo
uv sync
uv run python -V
```

There is no GPU on the login node. `nvidia-smi` only makes sense inside a GPU
job.

---

## OAR in 30 seconds

Submit from the front-end (`ritchie` / `calculco`).

| What | Command |
|------|---------|
| Interactive shell on a node | `oarsub -I …` |
| Batch script | `oarsub -S ./job.sh` |
| Cluster occupancy | `oarstatmon.py` |
| Your jobs | `oarstat -u` |
| One job, full details | `oarstat -j <id> -f` |
| Kill a job | `oardel <id>` |
| Attach to a running job | `oarsub -C <id>` |

Default walltime if you omit it: **2 hours**. Always set `walltime`. Format is
`HH:MM:SS` (or `H:MM` on some examples). When it expires, OAR kills the job.

Resource request `-l /nodes=2/cpu=2/core=4` **multiplies**: 2 nodes × 2 CPUs ×
4 cores = 16 cores.

Job states you will see: `W` waiting, `R` running, `T` terminated, `E` error.
`oarstat -j <id> -f` prints `assigned_hostnames`. Check that before you trust a
long GPU run.

Stdout/stderr: pass `-O` and `-E` at submit time, or put `#OAR -O` / `#OAR -E`
in the script. `%jobid%` is replaced by OAR.

```bash
oarsub -O "$HOME/logs/test_%jobid%.out" \
       -E "$HOME/logs/test_%jobid%.err" \
       -S ~/scripts/job.sh
```

Mail at the end (sometimes delayed / filtered):

```bash
oarsub --notify "[END,ERROR]mail:you@univ-littoral.fr" -S ./job.sh
```

---

## Interactive vs batch

Use **interactive** while you debug (`python`, `htop`, `nvidia-smi`). Use
**batch** once the command is known; you can disconnect.

Interactive, one core, default 2 h:

```bash
oarsub -I
```

Interactive, 8 CPU cores, 4 h 30, **not** on a GPU node:

```bash
oarsub -I -l /core=8,walltime=4:30 -p "gpudevice='-1'"
```

Interactive, one GPU:

```bash
oarsub -I -p "gpu > 0" -l host=1/gpu=1,walltime=4:00:00
```

You land on a `chimay*` node. `exit` frees the reservation. If you close the
laptop without `exit`, the reservation keeps running until walltime.

Batch script must be executable (`chmod +x job.sh`). Directives are lines
starting with `#OAR`. OAR parses **every** `#OAR` line in the file, including
commented-out ones. If you want to disable a directive, delete the `#OAR`
prefix, do not leave `#OAR -p ...` sitting in a comment.

```bash
#!/bin/bash
#OAR -n hello_cpu
#OAR -l /nodes=1/core=4,walltime=00:20:00
#OAR -p gpudevice='-1'
#OAR -O ~/logs/hello_%jobid%.out
#OAR -E ~/logs/hello_%jobid%.err

echo "host=$(hostname) date=$(date)"
echo "OAR_JOB_ID=$OAR_JOB_ID"
```

```bash
chmod +x ~/scripts/hello_cpu.sh
oarsub -S ~/scripts/hello_cpu.sh
oarstat -u
```

---

## CPU jobs

Ask for cores (and walltime). Tell OAR you do **not** want a GPU:

```text
-p gpudevice='-1'
```

If you skip that, the scheduler may park a DuckDB/Python job on `chimay31+`
and block people who actually need the cards. Admins are explicit about this.

Typical batch header:

```bash
#OAR -n my_cpu_job
#OAR -l /nodes=1/core=8,walltime=24:00:00
#OAR -p gpudevice='-1'
```

Whole node:

```bash
oarsub -I -l /nodes=1,walltime=8:00:00 -p "gpudevice='-1'"
```

Need a CPU family (optional):

```bash
# Intel Xeon Gold 6348 / HPE DL360
-p "gpudevice='-1' and cputype='Xeon_Gold_6348'"

# AMD EPYC 7643 / HPE DL365
-p "gpudevice='-1' and cputype='EPYC_7643'"
```

Do **not** pin leftover `orval*` names.

A few CPU nodes are project-dedicated (`chimay19` / `chimay20` at the time I
wrote this). Project members have priority; other jobs there can be killed.
Stay off them unless you are on that project.

---

## GPU jobs

GPU nodes: **chimay31–chimay35**. Requesting a GPU also reserves the CPU cores
tied to that card.

Official pattern:

```bash
# one GPU, any GPU node
oarsub -I -p "gpu > 0" -l host=1/gpu=1,walltime=8:00:00

# two GPUs on the same node
oarsub -I -p "gpu > 0" -l host=1/gpu=2,walltime=8:00:00

# four GPUs (chimay33 or chimay34)
oarsub -I -p "gpu > 0" -l host=1/gpu=4,walltime=8:00:00

# pin a host
oarsub -I -p "gpu > 0 AND host='chimay34'" -l host=1/gpu=1,walltime=8:00:00
```

In our scripts we also force a real GPU device, because `/nodes=1/gpu=1`
**without** a property has landed on a CPU host (`chimay01`) during the
migration. Batch header we actually use:

```bash
#OAR -n my_gpu_job
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR -p gpudevice<>'-1'
```

No extra quotes around that `-p` expression inside `#OAR` lines. Extra quotes
have made `oarsub -S` fail with "There are not enough resources" /
`OAR_JOB_ID=-5`.

Optional pin (H100 / CornelIA on chimay34; A100 on chimay31–32 — confirm with
`oarstatmon.py` before you depend on a name):

```bash
#OAR -p network_address='chimay34' and gpudevice<>'-1'
```

After the job starts:

```bash
nvidia-smi
hostname
oarstat -j $OAR_JOB_ID -f | grep assigned_hostnames
```

If `nvidia-smi` is missing or you are on `chimay01`–`chimay20`, you did not get
a GPU. Delete the job and fix the `-p` / `-l` lines.

PyTorch (once you are on the node):

```bash
source /nfs/opt/env/env.sh
# either a Calculco module, e.g. fidle/pytorch-2.7-et-cie-gpu
# or your own env with CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Queues: default, long, besteffort

| Queue / type | When I use it |
|--------------|----------------|
| `default` (no `-q`) | normal jobs |
| `-q long` | long walltimes, if your account is allowed |
| `-t besteffort` | fill idle nodes. **Can be killed** by any non-besteffort job |

Besteffort is useful for short retries when the cluster is empty. Do not use it
for a 48 h train you cannot restart. `-t besteffort -t idempotent` asks OAR to
resubmit if it gets killed; still write your own checkpoints.

Check occupancy **before** you grab a whole node interactively. Besteffort jobs
die if your interactive reservation needs those cores.

```bash
oarstatmon.py
```

---

## Watching jobs and connecting back

```bash
oarstat -u
oarstat -j 12345 -f
tail -f ~/logs/hello_12345.out
```

Connect to the node while it runs:

```bash
oarsub -C 12345
# then: htop -u $USER
# GPU: nvidia-smi
```

Leave with `exit` (that does not kill the batch job if you attached with `-C`).

`oardel 12345` stops it.

If "There are not enough resources":

1. `oarstatmon.py` — are CPU nodes Alive, or only GPUs?
2. CPU job with `gpudevice='-1'` on a day when CPU nodes are Dead will not
   start. Log in on `ritchie` if `calculco` still points at a dead front-end.
3. GPU request with a bad `-p` (extra quotes, old hostname) also fails this way.

Tiny probe, then delete it:

```bash
oarsub -l /nodes=1/core=4,walltime=1:00:00 -t besteffort \
       -p "gpudevice='-1'" \
       -O /tmp/oar_probe_%jobid%.out -E /tmp/oar_probe_%jobid%.err \
       "hostname; echo ok"
# oardel <jobid> when you have seen it run
```

---

## Python / our lab layout (ResearchModule)

If you are working on this repo, keep code in home and licensed data on
protected NFS:

```bash
export PROJECT_HOME=$HOME/ResearchModule
export DATA_PROTECTED=/nfs/data/protected/<lab>/<user>/ResearchModule
export DATASET_ROOT=$DATA_PROTECTED/Dataset
export WORK_SCRATCH=/workdir/<lab>/<user>
```

Put those in `~/.bashrc`. Copy `.env.example` to `.env.calculco` (gitignored)
or `scripts/calculco/common.local.sh`. OAR `-S` jobs start with a **clean**
environment; login-shell `export` is **not** inherited. Wrappers write a job
`.env` file for that reason.

Login node: `uv run pytest`, `uv run ruff check .`. Extraction, training,
graph builds: `scripts/calculco/`.

CPU examples (always `gpudevice='-1'` in the script):

```bash
cd "$PROJECT_HOME"
oarsub -O "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.out" \
       -E "$PROJECT_HOME/scripts/calculco/logs/rm_mimic_%jobid%.err" \
       -S "$PROJECT_HOME/scripts/calculco/extract_mimic.sh"
```

GPU examples: `submit_phase8_p0_neural_training.sh`,
`submit_phase8_p0_gnn_training.sh`, `submit_phase8_p0_paired_oof.sh`. Those
request `/nodes=1/gpu=1` plus `gpudevice<>'-1'`.

Do not paste patient rows, note text, or identifiers anywhere. Reports under
`reports/` are aggregate on purpose.

Full pipeline procedures: `WORKFLOWS.md`. Account-specific paths (not for
email): gitignored `Documentation/CalculcoSetup.local.md` on the server.

---

## Mistakes I keep seeing

1. Running training on `ritchie` because "it has Python". It will annoy
   everyone and get you yelled at.
2. CPU job without `gpudevice='-1'`, sitting on a GPU node.
3. GPU job that requested `/gpu=1` with no property, then quietly ran on CPU.
4. Pinning `orval*`.
5. Putting licensed data in home Git clones or `rsync` to a laptop.
6. Leaving huge files on `/scratch` or `/workdir` and being surprised they
   vanished.
7. `#OAR` left in a "commented" line. OAR still reads it.
8. Assuming `export` in your SSH session reaches `oarsub -S`. It does not.
9. Loading modules on the front-end, then `oarsub -I`, and wondering why they
   disappeared.

---

## Commands I actually type

```bash
whoami
hostname
quota -s
df -h "$HOME"
df -h /workdir/<lab>/<user>
purgelist
module avail
oarstatmon.py
oarstat -u
oarstat -j <id> -f
oardel <id>
```

---
