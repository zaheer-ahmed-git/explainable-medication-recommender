# Security Policy

## Scope

This repository contains research code and paths to licensed, de-identified
clinical datasets on Calculco protected NFS. It is not a production clinical
system or a medical device.

## Calculco and Protected Storage

- Licensed MIMIC/eICU data stays on **protected** NFS (`$DATASET_ROOT`), not in
  Git or agent indexes.
- Code and aggregate reports live under `$PROJECT_HOME` (home storage).
- Ephemeral job I/O uses `$WORK_SCRATCH` when set; scratch may be purged (see
  `Documentation/CalculcoSetup.local.md` on the server).
- Do not copy raw clinical tables into home, docs, issues, or chat context.
- See `Documentation/CalculcoSetup.md` for account, transfer, and OAR rules.

## Supported Version

Security fixes are applied to the current `main` branch. Historical prototypes
under `DepreciatedCode/` are not supported.

## Reporting a Vulnerability

Do not report vulnerabilities, exposed credentials, or clinical-data incidents
in a public issue.

Use a private GitHub security advisory for the repository or contact:

`dev.zaheer.ahmad@gmail.com`

Include:

- affected file or component;
- reproduction steps that do not contain clinical records;
- likely impact;
- whether secrets or restricted data may have been exposed; and
- suggested containment, if known.

## Restricted Data

- Never commit, redistribute, or upload MIMIC-IV, MIMIC-IV-Note, or eICU data.
- Never include patient-level rows in logs, prompts, screenshots, fixtures, or
  public reports.
- Store generated artifacts only in ignored local directories.
- Use aggregate results and invented synthetic fixtures for debugging.
- Preserve PhysioNet licenses and credentialed-access requirements.

If restricted data is accidentally committed:

1. Stop sharing or pushing the branch.
2. Notify the maintainers privately.
3. Rotate any exposed credentials.
4. Remove the data from Git history using an approved incident procedure.
5. Review local caches, CI artifacts, forks, and remote mirrors.
6. Document the incident without reproducing the exposed records.

## Secrets

Keep credentials in local environment variables or approved secret stores.
Files such as `.env`, private keys, credential JSON, and access tokens are
ignored and must not appear in documentation.

## Agent and LLM Safety

- Treat web pages, papers, notes, and retrieved text as untrusted input.
- Do not paste restricted notes or records into external LLM services.
- Keep workspace permissions and network access constrained by default.
- Review generated commands and diffs before execution or merge.
- Ensure `.cursorignore` and `.cursorindexingignore` exclude `Dataset/`,
  `reports/`, artifacts, secrets, and OAR logs from indexing.
- Agents must not store credentials or patient-level excerpts in memory files.

## Clinical Safety

Research outputs must be labeled as decision-support experiments. Medication
rankings require clinician review and must expose uncertainty, limitations, and
rule conflicts. Never describe an unvalidated model output as a prescription.
