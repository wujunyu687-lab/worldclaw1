# RoboLab120 Policy-RL Reward Skill

## Time

2026-07-22

## Target

Improve a Cosmos3-DROID policy in RoboLab120 block-stacking rollouts.

## Reward Contract

```text
R = 10 * terminal_success
  + 2 * verified_progress
  - bounded invalid_action_penalty
  - bounded trusted_safety_penalty
```

## Constraints

- Terminal success comes from RoboLab, not a VLM or video score.
- Progress comes from verified subtask/progress fields.
- Failed trajectories are capped below successful trajectories.
- Safety and audit events enter the final quality gate.
- Video and VLM checks are audit-only and do not enter the training reward.

## Evidence Use

The published block-stacking videos are qualitative audit artifacts. The training reward is defined by environment success and verified progress.
