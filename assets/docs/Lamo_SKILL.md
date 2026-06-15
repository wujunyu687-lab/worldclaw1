# CtrlWorld + LaMo Fusion Training Skill

> **Version**: v1.0
> **Date**: 2026-06-15 8:00
> **Purpose**: Integrate LaMo (Latent Motion Prior) into CtrlWorld (SVD-UNet backbone) to enhance physical consistency in action-conditioned world models via hierarchical learning rates and multi-objective training.

---

## 1. Overview

This skill describes how to modify CtrlWorld's training pipeline to incorporate LaMo's self-supervised latent motion priors. The key insight is that **latent-space frame differences carry physical motion signals** that can break the 'copy-current-frame' collapse inherent in pure MSE denoising objectives.

### 1.1 Problem Diagnosis from Previous Training

Previous training with `lr=3e-6` on all UNet parameters (1428 tensors) alongside a tiny `action_encoder` (6 tensors) resulted in:
- Initial loss `0.11` with no downward trend
- Loss oscillation: `0.076` (step 40) → `0.118` (step 180)
- `motion_prior` completely outside computation graph when `use_lamo_train=False`
- Pre-trained SVD distribution being corrupted by uniform LR across all layers

**Root cause**: Action signal drowned in 1428 UNet parameters. The model learned to 'cheat' by copying spatial content rather than responding to action conditioning.

---

## 2. Architecture Modifications

### 2.1 Data Flow

```
INPUT LAYER:
  History Frames (7)     Action Sequence (15)     Target Future (5)
       |                        |                          |
       v                        v                          |
   VAE Encoder           action_encoder                    |
   (frozen)              (3-layer MLP)                   |
       |                        |                          |
       v                        v                          v
  LATENT SPACE:
    z_history [B,7,3,24,40]  action_emb [B,5,1024]   z_target [B,5,3,24,40]
       |                         |
       +------------+------------+
                    |
                    v
  CTRLWORLD UNET (SVD backbone):
    Input: concat[z_history, z_noised_future] → [B,12,3,24,40]
    Cross-Attention: action_emb injected per-frame via spatial transformer
    Memory Retrieval: pose-conditioned sparse history frames
    Output: x0_pred [B,5,3,24,40]
                    |
                    v
  LAMO BRANCH (NEW):
    motion_prior (f_φ): ~3.4M params, lightweight CNN
      Input: (z_i, text_condition, τ=2)
      Output: Δz_pred (predicted latent motion)
    Drift Readout: channel-wise mean → macro drift vector d [B,C]
    Field Readout: spatially-resolved → micro motion field [B,C,H,W]
```

### 2.2 Component Specifications

#### Action Encoder (Inherited from CtrlWorld)

```python
# 3-layer MLP, newly initialized (not from SVD checkpoint)
action_hidden = self.action_encoder(action_cartesian)  # [B,T,7] → [B,T,1024]

# Temporal downsample: 15 steps → 5 steps to match video frame rate
action_emb = temporal_downsample(action_hidden)  # [B,5,1024]
```

**Note**: Cartesian-space actions (end-effector pose) are more stable than joint angles. CtrlWorld already performs this conversion.

#### Motion Prior Network (New, from LaMo)

```python
class MotionPrior(nn.Module):
    """Predicts latent-space motion delta_z given current frame and condition."""
    def __init__(self, latent_dim, cond_dim, hidden_dim=512):
        super().__init__()
        # Lightweight: ~3.4M parameters
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1),
        )

    def forward(self, z_i, cond, tau=2):
        # z_i: [B, C, H, W] — current frame latent
        # cond: [B, D] — text/action embedding
        feat = self.encoder(z_i)  # [B, hidden_dim, H, W]
        cond_feat = self.cond_proj(cond)[:, :, None, None].expand(-1, -1, H, W)
        fused = torch.cat([feat, cond_feat], dim=1)
        delta_z = self.predictor(fused)  # [B, C, H, W]
        return delta_z
```

#### Drift Readout (Macro — Training Time)

```python
def compute_drift(delta_z):
    """Channel-wise motion drift: describes overall latent change rate."""
    return delta_z.mean(dim=[2, 3])  # Spatial average → [B, C]
```

#### Field Readout (Micro — Sampling Time)

```python
def compute_field(delta_z):
    """Spatially-resolved motion field: per-location change vectors."""
    return delta_z  # [B, C, H, W] — preserves spatial structure
```

---

## 3. Training Objectives

### 3.1 Combined Loss Function

```
L_total = L_mse + λ_drift · L_drift + λ_phi · L_phi

  L_mse        L_drift        L_phi
 (denoising)  (motion drift)  (action consistency)
 CtrlWorld     LaMo (new)     Extension (new)
  original      training        training
```

### 3.2 L_mse — Base Diffusion Denoising Loss

```python
# Standard diffusion training (inherited from CtrlWorld)
# x0: clean target latent, t: diffusion timestep, noise: Gaussian
xt = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
pred = unet(xt, t, encoder_hidden_states=action_emb, ...)
loss_mse = F.mse_loss(pred, x0)  # x0-prediction parameterization
```

**Purpose**: Maintain base video generation quality.

### 3.3 L_drift — Latent Motion Drift Loss (LaMo Core)

```python
# Compute ground-truth latent difference from VAE-encoded clean latents
z_current = vae.encode(frame_i)           # [B, C, H, W]
z_future  = vae.encode(frame_{i+τ})       # [B, C, H, W]
delta_z_gt = z_future - z_current          # [B, C, H, W]

# Motion Prior prediction
delta_z_pred = motion_prior(z_current, text_cond, tau=2)

# Drift: channel-wise spatial mean
d_gt   = delta_z_gt.mean(dim=[2, 3])       # [B, C]
d_pred = delta_z_pred.mean(dim=[2, 3])     # [B, C]

# Scale-normalized drift loss (key LaMo design)
# Normalizes per-channel to prevent dominant channels from overwhelming others
d_gt_norm   = d_gt   / (d_gt.std() + 1e-6)
d_pred_norm = d_pred / (d_gt.std() + 1e-6)
loss_drift = F.mse_loss(d_pred_norm, d_gt_norm)
```

**Key Design Choices** (from LaMo ablation studies):

| Design | Description | Recommended |
|--------|-------------|-------------|
| Temporal lag τ | Adjacent vs. skip-frame | **τ=2** (superior to τ=1) |
| Target type | Channel-mean vs. per-pixel | **Channel-mean** (macro drift) |
| Loss formulation | L2 vs. scale-normalized | **Scale-normalized** |

**Purpose**: Forces the model to learn plausible motion rates instead of static copying.

### 3.4 L_phi — Action-Conditioned Motion Consistency (Extension)

```python
# Optional: Constrain motion prior with action conditioning
# Same action across different scenes should produce similar latent changes

action_emb = action_encoder(action)  # [B, D]
delta_z_pred = motion_prior(z_current, action_emb, tau=2)

# Action consistency: same action → similar drift
loss_phi = contrastive_loss(
    anchor=delta_z_pred,
    positive=delta_z_same_action,    # Same action, different scene
    negative=delta_z_diff_action,    # Different action
    temperature=0.07
)
```

**Purpose**: Ensure action genuinely drives motion, not scene content.

### 3.5 Loss Weight Schedule

```python
# Recommended initial configuration (derived from ablation experiments)
lambda_drift = 0.4    # LaMo paper recommendation
lambda_phi = 0.1      # Start from 0, increase gradually

# Progressive training strategy
# Phase 1 (0-20k steps):    λ_drift=0.4, λ_phi=0
#   → Learn base motion priors first
# Phase 2 (20k-50k steps):  λ_drift=0.4, λ_phi=0.1
#   → Introduce action consistency constraint
# Phase 3 (50k+ steps):     λ_drift=0.4, λ_phi=0.2
#   → Strengthen action-motion coupling
```

---

## 4. Hierarchical Learning Rate Strategy

### 4.1 Core Principle

Based on diagnostic results from previous training, **action-related modules must have 100-1000× higher LR than UNet backbone**.

### 4.2 Specific Configuration

```python
from torch.optim import AdamW

param_groups = [
    # Group 1: Action Encoder (trained from scratch, needs fast convergence)
    {"params": action_encoder.parameters(), "lr": 1e-3, "weight_decay": 0.01},

    # Group 2: Motion Prior (new, trained from scratch)
    {"params": motion_prior.parameters(), "lr": 5e-4, "weight_decay": 0.01},

    # Group 3: UNet Cross-Attention Layers (fine-tuned for action adaptation)
    {"params": [p for n, p in unet.named_parameters()
                if "attn" in n and "cross" in n],
     "lr": 1e-5, "weight_decay": 0.01},

    # Group 4: UNet Mid Block (fine-tuned for spatiotemporal fusion)
    {"params": unet.mid_block.parameters(), "lr": 5e-6, "weight_decay": 0.01},

    # Group 5: Other UNet layers (conservative fine-tuning)
    {"params": [p for n, p in unet.named_parameters()
                if any(x in n for x in ["down_blocks.2", "up_blocks.0"])],
     "lr": 1e-6, "weight_decay": 0.01},
]

optimizer = AdamW(param_groups)

# Frozen layers (explicitly excluded from training)
for param in vae.parameters(): param.requires_grad = False
for param in text_encoder.parameters(): param.requires_grad = False
for param in image_encoder.parameters(): param.requires_grad = False

# Freeze UNet head/tail layers (protect pre-trained priors)
for block in unet.down_blocks[:2]:
    for param in block.parameters(): param.requires_grad = False
for block in unet.up_blocks[-2:]:
    for param in block.parameters(): param.requires_grad = False
```

### 4.3 Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

warmup_steps = 5000
total_steps = 100000

# Independent scheduling per parameter group
for group in optimizer.param_groups:
    peak_lr = group["lr"]
    # Warmup: 0 → peak_lr
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01,
        end_factor=1.0, total_iters=warmup_steps
    )
    # Cosine Decay: peak_lr → peak_lr/10
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=peak_lr / 10
    )
```

---

## 5. Training Pipeline

### 5.1 Data Preparation (DROID)

```python
def build_training_sample(trajectory, current_idx):
    """Construct training sample from DROID trajectory."""
    # 1. History frames (7 frames, 1-2s interval)
    history_indices = sample_history(current_idx, k=7, interval=1.5)
    history_frames = [trajectory[i]["image"] for i in history_indices]
    history_poses = [trajectory[i]["cartesian_pose"] for i in history_indices]

    # 2. Target future frames (5 frames)
    future_indices = range(current_idx + 1, current_idx + 6)
    target_frames = [trajectory[i]["image"] for i in future_indices]

    # 3. Action sequence (15 steps, temporal downsample to 5)
    action_sequence = [trajectory[i]["action"]
                       for i in range(current_idx + 1, current_idx + 16)]
    action_cartesian = joint_to_cartesian(action_sequence)  # 7-DOF → Cartesian

    # 4. LaMo: latent difference pairs (for motion prior)
    # τ=2: difference between current frame and frame at i+2
    z_i = vae.encode(history_frames[-1])  # Current frame latent
    z_i_plus_2 = vae.encode(trajectory[current_idx + 2]["image"])
    delta_z_gt = z_i_plus_2 - z_i

    return {
        "history_frames": history_frames,      # [7, 3, H, W]
        "history_poses": history_poses,        # [7, 7]
        "target_frames": target_frames,        # [5, 3, H, W]
        "action_sequence": action_cartesian,   # [15, 7]
        "z_current": z_i,                      # [C, H, W]
        "delta_z_gt": delta_z_gt,              # [C, H, W]
        "text_prompt": trajectory.get("language_instruction", ""),
    }
```

### 5.2 Training Loop

```python
for step in range(total_steps):
    # -- 1. Data loading --
    batch = next(dataloader)

    # -- 2. VAE encoding (frozen) --
    with torch.no_grad():
        z_history = vae.encode(batch["history_frames"])   # [B, 7, 3, 24, 40]
        z_target = vae.encode(batch["target_frames"])       # [B, 5, 3, 24, 40]

    # -- 3. Action encoding --
    action_emb = action_encoder(batch["action_sequence"])   # [B, 5, 1024]

    # -- 4. Diffusion forward (CtrlWorld) --
    t = torch.randint(0, num_timesteps, (B,))
    noise = torch.randn_like(z_target)
    z_noised = sqrt_alpha_bar[t] * z_target + sqrt_one_minus_alpha_bar[t] * noise

    # UNet input: [history, noised_future]
    model_input = torch.cat([z_history, z_noised], dim=1)   # [B, 12, 3, 24, 40]

    # UNet prediction
    pred_x0 = unet(
        model_input,
        t,
        encoder_hidden_states=action_emb,
    )

    # -- 5. LaMo Motion Prior --
    z_current = z_history[:, -1]  # Last history frame [B, 3, 24, 40]
    delta_z_pred = motion_prior(
        z_current,
        text_cond=clip_encode(batch["text_prompt"]),
        tau=2
    )

    # Drift readout
    d_pred = delta_z_pred.mean(dim=[2, 3])       # [B, C]
    d_gt = batch["delta_z_gt"].mean(dim=[2, 3])  # [B, C]

    # -- 6. Loss computation --
    loss_mse = F.mse_loss(pred_x0, z_target)

    # Scale-normalized drift loss
    d_gt_norm = d_gt / (d_gt.std(dim=-1, keepdim=True) + 1e-6)
    d_pred_norm = d_pred / (d_gt.std(dim=-1, keepdim=True) + 1e-6)
    loss_drift = F.mse_loss(d_pred_norm, d_gt_norm)

    # Optional: action consistency
    loss_phi = 0  # Disabled in Phase 1
    if step > 20000:
        loss_phi = compute_action_consistency_loss(delta_z_pred, action_emb)

    # Total loss
    loss = loss_mse + lambda_drift * loss_drift + lambda_phi * loss_phi

    # -- 7. Backpropagation --
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % 20 == 0:
        print("Step", step, "loss=", loss.item(), "mse=", loss_mse.item(),
              "drift=", loss_drift.item(), "phi=", loss_phi)
```

### 5.3 Key Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Total steps | 100k | CtrlWorld original config |
| Batch size | 64 | Original config (2×8 H100) |
| Resolution | 192×320 | 3-camera joint prediction |
| History frames | 7 | 1-2s interval |
| Target frames | 5 | Prediction target |
| Action chunk | 15 steps | ~1s DROID action |
| τ (LaMo lag) | 2 | Skip-frame difference |
| λ_drift | 0.4 | LaMo recommendation |
| λ_phi | 0→0.1→0.2 | Progressive increase |
| Gradient clip | 1.0 | Prevent bs=7 anomaly gradients |
| EMA decay | 0.9999 | Stable sampling |

---

## 6. Inference and Sampling

### 6.1 Standard Diffusion Sampling (CtrlWorld)

```python
def sample_ctrlworld(initial_frame, action_sequence, num_steps=50):
    """Standard CtrlWorld sampling pipeline."""
    z = vae.encode(initial_frame)
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((B,), t)
        pred_x0 = unet(z, t_tensor, action_emb)
        z = denoise_step(z, pred_x0, t)
    return vae.decode(z)
```

### 6.2 LaMo Motion Prior Guidance (Enhanced Sampling)

```python
def sample_with_lamo_guidance(initial_frame, action_sequence, num_steps=50):
    """Inject LaMo Motion Prior Guidance during sampling."""
    z = vae.encode(initial_frame)
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((B,), t)

        # 1. Standard UNet prediction
        pred_x0 = unet(z, t_tensor, action_emb)

        # 2. LaMo Micro Motion Field Guidance
        # active_window_ratio ρ=0.8: enable guidance for first 80% steps
        if t > num_steps * (1 - 0.8):
            with torch.no_grad():
                delta_z_field = motion_prior(z, text_cond, tau=2)
                guidance_scale = 25.0  # λ_guide
                noise_correction = compute_noise_guidance(delta_z_field, t)
                pred_x0 = pred_x0 + guidance_scale * noise_correction

        # 3. Denoising step
        z = denoise_step(z, pred_x0, t)
    return vae.decode(z)
```

### 6.3 Long-Horizon Rollout (Policy-in-the-Loop)

```python
def rollout_policy_in_loop(policy, world_model, initial_obs, max_steps=100):
    """Closed-loop interaction between policy and world model."""
    observations = [initial_obs]
    actions = []
    for step in range(max_steps):
        action = policy(observations[-7:])  # Based on last 7 frames
        actions.append(action)

        future_frames = world_model.sample(
            initial_frame=observations[-1],
            action_sequence=actions[-15:],
        )
        observations.append(future_frames[0])

        # Optional: Memory Retrieval (CtrlWorld)
        if step % memory_interval == 0:
            retrieved = memory_bank.retrieve_similar_pose(observations[-1])
            observations.extend(retrieved)
    return observations
```

---

## 7. Experiments and Evaluation

### 7.1 Training Monitoring Metrics

| Metric | Frequency | Healthy Threshold |
|--------|-----------|-------------------|
| loss_mse | Every step | Steady decrease, final < 0.05 |
| loss_drift | Every step | Synchronized with loss_mse |
| loss_phi | Every step (Phase 2+) | Slow decrease, no oscillation |
| action_encoder grad_l2 | Every 100 steps | > 0.01 |
| motion_prior grad_l2 | Every 100 steps | > 0.001 |
| UNet cross-attn grad_l2 | Every 100 steps | > 0.1 |
| FVD (generation quality) | Every 5k steps | No degradation vs. original SVD |
| Long-horizon consistency | Every 10k steps | No drift in 10s rollout |

### 7.2 Ablation Study Design

```
Exp 1: Baseline (CtrlWorld original)
  - L_mse only, full UNet LR=1e-5

Exp 2: + LaMo Drift Loss
  - L_mse + λ_drift·L_drift, hierarchical LR

Exp 3: + LaMo Guidance (sampling time)
  - Based on Exp 2, add motion field guidance during inference

Exp 4: Full (Drift + Guidance + Action Consistency)
  - L_mse + λ_drift·L_drift + λ_phi·L_phi
  - Hierarchical LR + progressive training

Exp 5: Train from scratch vs. resume from checkpoint
  - Compare SVD original weights vs. 240-step trained checkpoint
```

### 7.3 Evaluation Benchmarks

| Benchmark | Metrics | Purpose |
|-----------|---------|---------|
| DROID Validation | MSE, PSNR, SSIM | Reconstruction quality |
| Long-horizon Rollout (10s) | Human rating, FVD | Temporal consistency |
| Policy Evaluation | Success rate (vs. real world) | Policy compatibility |
| VideoPhy / VideoPhy2 | SA, PC | Physical plausibility |

---

## 8. Troubleshooting

### 8.1 Loss Not Decreasing

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| loss_mse stuck at 0.11 | SVD pre-training at lower bound | Add L_drift, use hierarchical LR |
| loss_drift not decreasing | motion_prior gradient is zero | Check computation graph connectivity |
| loss_phi oscillating | λ_phi too large | Start from 0.01, increase gradually |

### 8.2 Generation Quality Degradation

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| Blurry/corrupted frames | UNet LR too high | Reduce to 1e-6, freeze head/tail layers |
| Action not followed | action_encoder LR too low | Increase to 1e-3 |
| Long-horizon drift | Missing temporal constraint | Enable L_phi, add memory retrieval |

### 8.3 Training Instability

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| Loss NaN | Gradient explosion | gradient clip=1.0, check action value range |
| Severe oscillation | bs=7 noise too large | Increase EMA decay, or increase effective bs |
| Pre-trained capability lost | Full UNet unfrozen | Freeze down/up blocks, only open mid+cross-attn |

---

## 9. Implementation Checklist

```
[ ] CODE LEVEL
  [ ] action_encoder output correctly wired to UNet cross-attention
  [ ] motion_prior participates in computation graph (requires_grad=True, no detach)
  [ ] Hierarchical optimizer correctly configured (5 parameter groups)
  [ ] Frozen layers explicitly set requires_grad=False
  [ ] Gradient clipping (max_norm=1.0)
  [ ] EMA update logic

[ ] DATA LEVEL
  [ ] DROID action correctly converted to Cartesian space
  [ ] Temporal downsample (15→5 steps)
  [ ] τ=2 latent difference pairs correctly constructed
  [ ] VAE encoder output has no anomalies

[ ] TRAINING LEVEL
  [ ] Warmup starts from 0 (first 5k steps)
  [ ] Cosine decay to peak_lr/10
  [ ] Loss weights progressive: λ_phi starts from 0
  [ ] Regular checkpoint saving (every 5k steps)
  [ ] Per-module gradient norm monitoring

[ ] EVALUATION LEVEL
  [ ] Comparison with original CtrlWorld (same seed)
  [ ] Long-horizon rollout video saving
  [ ] FVD computation script ready
```

---

## 10. References

- **CtrlWorld**: Guo et al., "A Controllable Generative World Model for Robot Manipulation", ICLR 2026
- **LaMo**: Jiang et al., "Self-Supervised Latent Motion Priors for Physical Realism in Video Generation", 2026
- **SVD**: Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets", 2023
- **DROID**: Khazatsky et al., "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset", 2024

---

> **Note**: This skill is based on CtrlWorld's official implementation and LaMo's paper methodology. Specific implementations should be adjusted according to actual codebase. It is recommended to validate each component on a small dataset (1k clips) before scaling to full DROID training.