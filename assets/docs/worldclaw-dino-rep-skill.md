# DINO-REP COSMOS3 Repair Skill

## Time

2026-06-16

## Target

COSMOS3 video-to-world rollout quality under Plan 2.

## Dataset And Evidence

The node uses the Plan 2 replay and metric bundle, including qualitative replay, PSNR, SSIM, LPIPS, FID, and WorldArena-style metrics.

## Failure Diagnosis

DINO-REP helped depth and motion but exposed a remaining appearance gap: background consistency and subject consistency did not improve at the same rate.

## Intervention

Use train-test aligned DINO-REP conditioning as an additional representation signal while preserving the base video generation objective.

## Next Patch Rule

If geometry and motion improve while appearance weakens, the next patch should target appearance retention, REP injection position, or data replay focused on complex backgrounds and fast-moving subjects.
