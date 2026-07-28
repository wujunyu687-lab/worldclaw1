# FRANKA Real-Robot Transfer Skill

## Time

2026-07-27

## Target

Compare original COSMOS3 and improved COSMOS3 on FRANKA real-robot tasks.

## Task Set

- Stack three blocks into one tower
- Fold the white tablecloth
- Move a polyhedron from a bowl into a drawer and close the drawer

## Observed Failure Modes

- The original stack rollout is disrupted by the blue bowl and fails late-stage grasping.
- The original tablecloth rollout uses a longer, less direct path.
- The original drawer task fails when the target object is occluded inside the bowl.

## Improved Behavior

The improved COSMOS3 policy completes the stack, shortens the tablecloth path, and completes the occluded pick-place-close drawer sequence.

## Page Encoding

The public videos are accelerated 4x and converted to H.264 for browser compatibility. The acceleration is presentation-only and does not change the task label.
