# GitHub release plan

## Recommended repository identity

- Working title: `so101-robot-learning`.
- Scope: one umbrella repository with separate ACT, Diffusion Policy, and
  SmolVLA experiment sections.
- Visibility: public.
- License: Apache-2.0, matching the upstream LeRobot code used by the included
  patch.

The local directory remains `so101-two-cube-ordering` so existing scripts and
absolute local artifacts do not need to move. A GitHub repository can use the
broader public name without renaming this directory.

## Included public evidence

- Root project overview and explicit claim boundaries.
- Three compressed real-robot GIFs with source-trial provenance.
- Config templates, guarded command launchers, dataset/QC scripts, manifests,
  and safety protocols.
- Curated ACT and SmolVLA reports.
- Diffusion experiment note and the tested action-chunk blending patch.

## Excluded material

- `rig.local.yaml`, calibration files, serial mappings, tokens, and login state.
- Raw datasets, full videos, checkpoints, W&B files, caches, and `.runtime`.
- `CHAT_TRANSCRIPT.md`, `RESUME.md`, `PROGRESS.md`, `PROJECT_PLAN.md`, and
  `AGENTS.md`.
- Full environment exports and package freezes.
- The shared local LeRobot checkout and its unrelated dirty changes.

## Required checks before push

1. Initialize a local Git repository and stage only non-ignored files.
2. Scan staged content for tokens, credentials, private keys, local SSH hosts,
   absolute home paths, and unexpected large files.
3. Run Python compilation, shell syntax checks, manifest validators, Markdown
   link checks, GIF decode checks, and `git diff --check`.
4. Review the staged file list and README rendering.
5. Confirm repository name, visibility, and license. Completed: public
   `IptiharPolat/so101-robot-learning`, Apache-2.0.
6. Create the GitHub repository, commit locally, and push only after that final
   confirmation.

## Media budget

| Asset | Size | Duration |
|---|---:|---:|
| ACT fixed-order success | 3.94 MB | 13.01 s |
| SmolVLA Pink -> Cyan | 5.13 MB | 16.88 s |
| SmolVLA Cyan -> Pink | 6.25 MB | 20.88 s |
| Total | 15.32 MB | 50.77 s |

This is small enough for a normal GitHub repository and avoids Git LFS for the
public preview assets.
