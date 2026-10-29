"""MOOD-style finetuning-free guided sampling on DRAKES MDLM (planned 2026-05-01).

Recipe: Lee, Jo, Hwang. "Exploring Chemical Space with Score-based
Out-of-distribution Generation" (MOOD), ICML 2023. Two added gradient terms,
no fine-tuning of the diffusion model:

  (1) OOD push   : damp the unconditional score by (1 − √λ).
                   On MDLM categorical posterior this is a mix of
                   p_θ(x_0|x_t) with a uniform-over-{A,C,G,T} distribution.
  (2) Property g.: α_t · ∇P_φ via Tweedie x̂_0 estimate, where α_t is the
                   auto-balanced ratio (Eq 11) ‖score‖ / ‖∇P_φ‖ * r.

P_φ is DRAKES's `reward_oracle_ft.ckpt` -- the GUIDANCE oracle, loaded as
`oracle.get_gosai_oracle(mode='train')` (frozen 3-head Enformer-style
regressor over [hepg2, k562, sknsh]; the head is selected by --cell_type).

It is a DIFFERENT trained model from `reward_oracle_eval.ckpt`
(`mode='eval'`), which this script loads separately as `oracle_eval` and
uses ONLY to score finished sequences (the `gold_mu` field in
summary.json). The two checkpoints share all 285 tensor names and shapes,
yet 0/285 tensors are numerically identical (mean |diff| 768.9;
global_step 8910 vs 5460). Keeping guidance and evaluation on separate
oracles is what makes the reported Pred-activity non-circular, so
oracle_eval must never be passed to `mood_sample` as P_φ.

Optional `--use_masked_separator true`: replace P_φ(x) with mask-localized
oracle(x · M + bkg · (1−M)) using the frozen V-REx masked separator
loaded from --ms_ckpt.

Run (M1 sanity — λ=0, α_ratio=0 must reproduce SVDD plain ≈ 0.34):
    python -m scripts.drakes_backbone.mood_guided_sample_mdlm \
        --name sanity_uncond --lam 0.0 --alpha_ratio 0.0 \
        --num_samples 64 --batch_size 32

Plan: runs/result_checkpoints/2026-05-01_planned_mdlm_switch/notes.md
"""
import sys
import os

# Make DRAKES drakes_dna importable. Set LOCOGEN_DRAKES_REPO to the path of
# DRAKES `drakes_dna/` (clone from https://github.com/ChenyuWang-Monica/DRAKES).
DRAKES_REPO = os.environ.get("LOCOGEN_DRAKES_REPO")
if DRAKES_REPO is None:
    raise RuntimeError(
        "LOCOGEN_DRAKES_REPO environment variable is not set. "
        "Clone DRAKES (https://github.com/ChenyuWang-Monica/DRAKES) and "
        "export LOCOGEN_DRAKES_REPO=/path/to/DRAKES/drakes_dna"
    )
if DRAKES_REPO not in sys.path:
    sys.path.insert(0, DRAKES_REPO)

# grelu schema-shift workaround (same as the finetune script).
import grelu.lightning as _grelu_lightning
_orig_on_load = _grelu_lightning.LightningModel.on_load_checkpoint
def _patched_on_load(self, checkpoint):
    checkpoint.setdefault('data_params', {})
    checkpoint.setdefault('performance', {})
    return _orig_on_load(self, checkpoint)
_grelu_lightning.LightningModel.on_load_checkpoint = _patched_on_load

import diffusion_gosai_update
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
import oracle
# Set LOCOGEN_DRAKES_DATA to the directory containing `data_and_model/`.
oracle.base_path = os.environ.get(
    "LOCOGEN_DRAKES_DATA",
    "./data/DRAKES_data/data_and_model/"
)

import argparse
import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import str2bool, set_seed

# Our masked-separator (loaded frozen if --use_masked_separator true).
from locogen.model.masked_separator import MaskedSeparatorModel  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

NUC = ['A', 'C', 'G', 'T']

# DRAKES multi-task oracle head index per `enhancer_gosai_dataset.py:14` and
# `train_oracle.py:35-36` (both reward_oracle_ft and reward_oracle_eval ckpts
# are 3-head Enformer regressors trained on [hepg2, k562, sknsh] columns).
ORACLE_HEAD = {'hepg2': 0, 'k562': 1, 'sknsh': 2}


def load_masked_separator(ms_config_path, ms_ckpt_path, device, cell_type=None):
    if ms_config_path is None or ms_ckpt_path is None:
        raise SystemExit(
            'This run needs the masked separator (--use_masked_separator, '
            '--sil_strength > 0 or --ood_sil_localized), so both --ms_config and '
            '--ms_ckpt are required. Released checkpoints: '
            'https://huggingface.co/DabinJeong/LocoGen')

    cfg = OmegaConf.load(ms_config_path)

    # The oracle head comes from --cell_type while the separator comes from
    # --ms_config/--ms_ckpt. Nothing downstream couples them, so a mismatched
    # pair would silently score one cell line's sequences with another cell
    # line's separator. Fail loudly instead.
    if cell_type is not None:
        cfg_target = cfg.get('masked_separator', {}).get('target_idx', None)
        expected = ORACLE_HEAD[cell_type]
        if cfg_target is None:
            raise SystemExit(
                f'{ms_config_path} sets no masked_separator.target_idx, so it '
                f'cannot be checked against --cell_type {cell_type}.')
        if int(cfg_target) != expected:
            inverse = {v: k for k, v in ORACLE_HEAD.items()}
            raise SystemExit(
                f'separator/oracle cell mismatch: --cell_type {cell_type} '
                f'selects oracle head {expected}, but {ms_config_path} has '
                f'target_idx {cfg_target} ({inverse.get(int(cfg_target), "?")}). '
                f'Pass the --ms_config and --ms_ckpt for {cell_type}.')

    model = MaskedSeparatorModel(cfg).to(device)
    payload = torch.load(ms_ckpt_path, map_location=device)
    model.load_state_dict(payload['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[masked-sep] loaded {ms_ckpt_path} (epoch={payload.get('epoch', '?')})")
    return model


def property_phi(x_hat_0, oracle_model, ms_model=None, head_idx=0):
    """P_φ(x̂_0) -> scalar reward per sample (B,).

    x_hat_0: (B, L, 4) soft one-hot, gradient flows through.
    head_idx: which oracle head to read off (0=hepg2, 1=k562, 2=sknsh).
    """
    if ms_model is not None:
        # Mask-localized: replace non-mask positions with mask-token embedding,
        # then score the masked sequence. The mask itself is also differentiable
        # in x_hat_0 via the separator's CNN, but separator weights are frozen.
        M = ms_model.separator(x_hat_0)                          # (B, L)
        M3 = M.unsqueeze(-1)
        bkg = ms_model.mask_embed.view(1, 1, -1)                 # (1, 1, 4)
        x_for_oracle = x_hat_0 * M3 + bkg * (1.0 - M3)           # (B, L, 4)
    else:
        x_for_oracle = x_hat_0

    # DRAKES oracle expects (B, 4, L)
    preds = oracle_model(x_for_oracle.transpose(1, 2)).squeeze(-1)  # (B, 3) or (B,)
    if preds.ndim == 2:
        return preds[:, head_idx]
    return preds


def mood_step(model, oracle_model, x, t, dt, lam, r, ms_model=None, log_list=None,
              step_idx=None, alpha_cap=1.0, head_idx=0, sil_strength=0.0,
              use_phi_masking=False, sil_t_threshold=1.0,
              ood_sil_localized=False, ood_sil_threshold=0.5,
              ood_sil_invert=False, ms_constant_rho=None,
              ood_global=False):
    """One MOOD-guided MDLM reverse step.

    Math:
      log_p_x0 = MDLM denoiser  (B, L, V=5)
      x̂_0     = softmax(log_p_x0[:, :, :-1])   # Tweedie clean estimate
      r_φ     = P_φ(x̂_0)                       # reward
      ∇r/∇x̂  = autograd.grad(r_φ.sum(), x̂_0)
      α_t     = r · ‖log_p_x0[:,:,:4]‖ / ‖∇r/∇x̂‖   # Eq 11 magnitude match

      # MOOD Eq 6 (OOD push) on categorical posterior:
      q_xs over nucleotides  ←  damp toward uniform by (1−√λ)
      # MOOD Eq 9 (property guidance):
      log q_xs[:, :, :4]    +=  α_t · ∇r/∇x̂

    The mask-channel q_xs[:, :, mask_idx] = move_chance_s is left untouched —
    it controls *whether* to unmask, the {A,C,G,T} channels control *what to*.
    Same convention as `_ddpm_update` in DRAKES diffusion_gosai_update.py.
    """
    if x.ndim == 2 or x.shape[-1] != model.vocab_size:
        x = F.one_hot(x, num_classes=model.vocab_size).to(torch.float32)

    sigma_t, _ = model.noise(t)
    sigma_s, _ = model.noise(t - dt)
    if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
    move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
    move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]
    unet_conditioning = sigma_t

    # ----- denoiser forward (with grad through inputs for property guidance) -----
    use_grad = (r > 0.0)
    if use_grad:
        x_in = x.detach().clone().requires_grad_(True)
        log_p_x0 = model.forward(x_in, unet_conditioning)        # (B, L, V=5)

        # === Sanity (gated, first 5 steps only): is log_p_x0 already log-prob? ===
        if step_idx is not None and step_idx < 5:
            raw = log_p_x0[:, :, :-1].detach()
            raw_min = raw.min().item(); raw_max = raw.max().item()
            raw_abs_mean = raw.abs().mean().item()
            normalized = F.log_softmax(raw, dim=-1)
            norm_min = normalized.min().item(); norm_max = normalized.max().item()
            norm_abs_mean = normalized.abs().mean().item()
            print(f"[sanity step={step_idx}] raw: min={raw_min:.3f} max={raw_max:.3f} abs_mean={raw_abs_mean:.3f} | "
                  f"log_softmax: min={norm_min:.3f} max={norm_max:.3f} abs_mean={norm_abs_mean:.3f}")

        # Tweedie x̂_0 over the 4 nucleotide channels (drop mask channel).
        x_hat_0 = log_p_x0[:, :, :-1].exp()                       # (B, L, 4)
        # Renormalize (in case mass leaks via the mask channel)
        x_hat_0 = x_hat_0 / (x_hat_0.sum(-1, keepdim=True) + 1e-12)

        # property_phi uses ms_model for input-masking ONLY when explicitly enabled.
        # ms_model may also be loaded for SIL gradient gating (different mechanism).
        ms_for_phi = ms_model if use_phi_masking else None
        reward = property_phi(x_hat_0, oracle_model, ms_model=ms_for_phi, head_idx=head_idx)  # (B,)
        grad_reward = torch.autograd.grad(reward.sum(), x_hat_0)[0]      # (B, L, 4)

        # MOOD Eq 11: auto-balance α_t = r · ‖score‖ / ‖∇P_φ‖
        # 'score' analog on MDLM = log p_x0[:, :, :4] norm.
        # Cap the ratio to avoid exp() overflow when grad is tiny early in
        # sampling — empirically the unbounded α_t can push log-prob shifts
        # past 50 and `q_xs.exp()` blows up to inf, breaking categorical sampling.
        # MOOD Eq.11 magnitude match. Two fixes layered on top of the original
        # `.norm()` formulation:
        # (1) per-element mean (not norm) — invariant to B*L*V element count.
        # (2) exclude DRAKES sentinel values (-1e6 entries injected to force
        #     prob=0 on already-decided positions). Sentinels otherwise
        #     dominate any summary statistic and pin alpha_t at the cap 100%
        #     of the time (diagnostics 2026-05-01/02).
        # Sanity step=0 (all-MASK): valid log-probs are in [-1.5, -1.2],
        # log(0.25) ≈ -1.386 baseline.
        nuc_lp = log_p_x0[:, :, :-1].detach()
        valid = nuc_lp > -1e5
        score_mag = nuc_lp[valid].abs().mean() if valid.any() else nuc_lp.abs().mean()
        grad_mag = grad_reward.abs().mean()
        alpha_t_raw = r * (score_mag / (grad_mag + 1e-12))
        alpha_t = torch.clamp(alpha_t_raw, max=alpha_cap)
        if log_list is not None:
            log_list.append({
                'step': step_idx,
                't': float(t.mean().item()),
                'alpha_t_raw': float(alpha_t_raw.item()),
                'alpha_t_clamped': float(alpha_t.item()),
                'score_mag': float(score_mag.item()),
                'grad_mag': float(grad_mag.item()),
            })
        log_p_x0 = log_p_x0.detach()
        x = x.detach()
    else:
        with torch.no_grad():
            log_p_x0 = model.forward(x, unet_conditioning)
            # Tweedie x̂_0 is still needed downstream for separator-based OOD
            # localization (incl. the constant-mask ablation) even when property
            # gradient is off (r=0). Compute without grad tracking.
            x_hat_0 = log_p_x0[:, :, :-1].exp()
            x_hat_0 = x_hat_0 / (x_hat_0.sum(-1, keepdim=True) + 1e-12)
        grad_reward = None
        alpha_t = 0.0

    # ----- DRAKES posterior over nucleotides (q_xs without guidance) -----
    # q_xs[:, :, k] = p_x0[k] * (move_chance_t - move_chance_s) for k in {A,C,G,T}
    # q_xs[:, :, mask_idx] = move_chance_s
    q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)       # (B, L, 5)
    q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]

    # ----- MOOD Eq 6 (discrete analogue): position-aware logit damping (v3pos) -----
    # Hypothesis B: damp logits ONLY at currently-masked positions, where the
    # property gradient is active. Already-unmasked positions keep original
    # logits (sentinel-marked at non-active tokens; copy through via copy_flag).
    # v3logit (uniform damping) was the prior version; in principle equivalent
    # at unmasked positions (sentinel dominates softmax either way) but tested
    # explicitly here to isolate damping to the gradient-active subset.
    # λ=0 short-circuits — identical to un-damped q_xs above.
    if lam > 0.0:
        sqrt_lam = float(np.sqrt(lam))
        is_masked = (x[:, :, model.mask_index] > 0.5)             # (B, L) bool
        # ood_global=True: paper-faithful MOOD (Eq 4) — apply (1-√λ) globally to
        # ALL positions, not gated by is_masked. v3logit equivalent. At already-
        # unmasked positions, log_p_x0 has sentinel-marked logits so categorical
        # sampling is dominated by the active token regardless (sentinel pinning).
        # Only the lam>0 block path differs; downstream LocoGen separator gate
        # composes correctly because damp_locations = is_masked & sil_background
        # → ones & sil_background = sil_background when ood_global is on.
        if ood_global:
            is_masked = torch.ones_like(is_masked, dtype=torch.bool)
        # v6 (anchor + background OOD): apply OOD damping ONLY at SIL-background
        # positions (M < threshold). Functional motifs preserved, background pushed
        # off train manifold. Time-conditional via sil_t_threshold (separator
        # unreliable on soft x_hat_0 at high t).
        damp_locations = is_masked    # default v3pos (all currently-masked)
        t_scalar_ood = float(t.mean().item())
        if (ood_sil_localized and (ms_model is not None or ms_constant_rho is not None)
                and t_scalar_ood < sil_t_threshold):
            with torch.no_grad():
                if ms_constant_rho is not None:
                    # Constant-mask ablation: bypass separator forward.
                    sil_mask_ood = torch.full(
                        x_hat_0.shape[:2],
                        float(ms_constant_rho),
                        device=x_hat_0.device,
                        dtype=x_hat_0.dtype,
                    )
                else:
                    sil_mask_ood = ms_model.separator(x_hat_0.detach())  # (B, L)
            sil_background = sil_mask_ood < ood_sil_threshold        # bool
            if ood_sil_invert:
                # Inverted-mask ablation ("attack" ablation in the main text): damp at
                # FOREGROUND (motifs) instead of background.
                # Sanity inverse — should hurt reward / atac_h4 if localization matters.
                sil_background = ~sil_background
            damp_locations = is_masked & sil_background
            if log_list is not None and len(log_list) > 0:
                log_list[-1]['ood_sil_localized_active'] = True
                log_list[-1]['ood_sil_inverted'] = bool(ood_sil_invert)
                log_list[-1]['ood_damp_frac'] = float(damp_locations.float().mean().item())
        elif ood_sil_localized and log_list is not None and len(log_list) > 0:
            log_list[-1]['ood_sil_localized_active'] = False
        damping_factor = torch.where(
            damp_locations.unsqueeze(-1),                          # (B, L, 1)
            torch.tensor(1.0 - sqrt_lam, device=log_p_x0.device, dtype=log_p_x0.dtype),
            torch.tensor(1.0,            device=log_p_x0.device, dtype=log_p_x0.dtype),
        )                                                          # (B, L, 1)
        log_p_nuc    = log_p_x0[:, :, :4]                          # (B, L, 4)
        log_p_damped = log_p_nuc * damping_factor                  # (B, L, 4)
        p_damped     = log_p_damped.softmax(dim=-1)
        nuc_mass     = (move_chance_t - move_chance_s)
        q_xs[:, :, :4] = p_damped * nuc_mass
        q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]
        # Mass conservation: q_xs.sum(-1) == move_chance_t.
        ms_sum = q_xs.sum(dim=-1)                                  # (B, L)
        ref    = move_chance_t.squeeze(-1).expand_as(ms_sum)
        assert torch.allclose(ms_sum, ref, atol=1e-5), \
            f"OOD damping broke mass conservation: max diff {(ms_sum - ref).abs().max().item():.2e}"

    # ----- SIL gradient gating (Sequence Invariant Learning) ------------------
    # Multiply property gradient by SIL mask M so gradient flows only through
    # functional positions identified by the trained separator. alpha_t is
    # computed BEFORE gating, so gradient magnitude at invariant positions is
    # unchanged; non-invariant positions are zeroed (or attenuated by 1-strength).
    # Verified motif-aligned by post-hoc analysis (3/3 PASS in
    # runs/sil_post_hoc/verdict.json).
    #
    # Time-conditional gating: separator was trained on clean one-hot Train
    # sequences but x_hat_0 is soft (~uniform at t≈1). Apply SIL gating only
    # when t < sil_t_threshold (late sampling, x_hat_0 nearly one-hot).
    # sil_t_threshold=1.0 means always apply (legacy behavior).
    t_scalar = float(t.mean().item())
    sil_active = (sil_strength > 0.0 and ms_model is not None
                  and t_scalar < sil_t_threshold)
    if grad_reward is not None and sil_active:
        with torch.no_grad():
            sil_mask = ms_model.separator(x_hat_0.detach())     # (B, L), in [0,1]
        gate = sil_strength * sil_mask + (1.0 - sil_strength)   # >=1-strength
        grad_reward = grad_reward * gate.unsqueeze(-1)
        if log_list is not None and len(log_list) > 0:
            log_list[-1]['sil_mask_mean'] = float(sil_mask.mean().item())
            log_list[-1]['sil_gate_mean'] = float(gate.mean().item())
            log_list[-1]['sil_t_active'] = True
    elif sil_strength > 0.0 and log_list is not None and len(log_list) > 0:
        log_list[-1]['sil_t_active'] = False

    # ----- MOOD Eq 9: property guidance shift on log q_xs nucleotide channels -----
    if grad_reward is not None and float(alpha_t) > 0.0:
        log_q = torch.log(q_xs[:, :, :-1] + 1e-12)
        log_q_guided = log_q + alpha_t * grad_reward
        q_xs_nuc = log_q_guided.exp()
        # Renormalize so the {A,C,G,T} channels still sum to (move_chance_t - move_chance_s)
        nuc_mass = (move_chance_t - move_chance_s)                # (B, 1, 1)
        q_xs_nuc = q_xs_nuc / (q_xs_nuc.sum(-1, keepdim=True) + 1e-12) * nuc_mass
        q_xs[:, :, :-1] = q_xs_nuc

    # ----- categorical sample. DRAKES `q_xs` does NOT sum to 1 along the
    # vocab axis (sum = move_chance_t), so torch.distributions.Categorical's
    # simplex check rejects it. Use multinomial which normalizes internally
    # (and tolerates non-simplex weights as long as they're >= 0 and finite). -----
    if not torch.isfinite(q_xs).all():
        # Defensive: if any inf/nan slipped through, fall back to model posterior
        # without guidance for this step rather than crashing the whole sweep.
        print("[mood] non-finite q_xs detected; falling back to model posterior")
        q_xs = (log_p_x0.exp() * (move_chance_t - move_chance_s))
        q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]
    q_xs = q_xs.clamp_min(0.0)
    B, L, V = q_xs.shape
    _x_flat = torch.multinomial(q_xs.reshape(B * L, V), num_samples=1)  # (B*L, 1)
    _x = _x_flat.reshape(B, L)
    _x = F.one_hot(_x, num_classes=model.vocab_size).float()

    # copy_flag: positions already unmasked stay; masked positions take _x
    copy_flag = (1 - x[:, :, model.mask_index]).unsqueeze(-1)
    return copy_flag * x + (1 - copy_flag) * _x


def mood_sample(model, oracle_model, num_steps, lam, r, batch_size, ms_model=None,
                eps=1e-5, log_list=None, alpha_cap=1.0, head_idx=0, sil_strength=0.0,
                use_phi_masking=False, sil_t_threshold=1.0,
                ood_sil_localized=False, ood_sil_threshold=0.5,
                ood_sil_invert=False, ms_constant_rho=None,
                step_times_ms=None, ood_global=False):
    """Run a full MOOD-guided MDLM trajectory; return final argmax token ids (B, L).

    If `step_times_ms` is a list, append per-step wall time (ms) for compute-cost
    instrumentation. CUDA sync is applied around each step for accurate timing.
    """
    L = model.config.model.length
    x = model._sample_prior(batch_size, L).to(model.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps
    for i in range(num_steps):
        t = timesteps[i] * torch.ones(batch_size, 1, device=model.device)
        if step_times_ms is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter() if step_times_ms is not None else None
        x = mood_step(model, oracle_model, x, t, dt, lam, r, ms_model=ms_model,
                      log_list=log_list, step_idx=i, alpha_cap=alpha_cap,
                      head_idx=head_idx, sil_strength=sil_strength,
                      use_phi_masking=use_phi_masking,
                      sil_t_threshold=sil_t_threshold,
                      ood_sil_localized=ood_sil_localized,
                      ood_sil_threshold=ood_sil_threshold,
                      ood_sil_invert=ood_sil_invert,
                      ms_constant_rho=ms_constant_rho,
                      ood_global=ood_global)
        if step_times_ms is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times_ms.append((time.perf_counter() - _t0) * 1000.0)
    # Noise-removal: argmax the denoised distribution at the last step (DRAKES default)
    if model.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(batch_size, 1, device=model.device)
        unet_conditioning = model.noise(t)[0]
        with torch.no_grad():
            logits = model.forward(x, unet_conditioning)
        return logits[:, :, :-1].argmax(dim=-1)
    return x.argmax(dim=-1)


def tokens_to_dna(tokens):
    return [''.join(NUC[i] for i in row.tolist()) for row in tokens]


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--base_path', type=str,
                    default=os.environ.get("LOCOGEN_DRAKES_DATA",
                                            "./data/DRAKES_data/data_and_model/"))
    ap.add_argument('--out_root', type=str,
                    default='./runs/mdlm_mood/')
    ap.add_argument('--name', type=str, required=True)
    ap.add_argument('--num_samples', type=int, default=640)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_steps', type=int, default=128)
    ap.add_argument('--lam', type=float, default=0.0,
                    help='MOOD OOD-push hyperparameter; 0=vanilla, ->1=more OOD')
    ap.add_argument('--alpha_ratio', type=float, default=0.0,
                    help='MOOD Eq 11 magnitude ratio r; 0=no property guidance')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--use_masked_separator', type=str2bool, default=False,
                    help='If true, P_φ(x) becomes oracle(x·M + bkg·(1-M))')
    ap.add_argument('--ms_ckpt', type=str, default=None,
                    help='Path to V-REx separator checkpoint. No default: pass the checkpoint '
                         'for the cell line given by --cell_type. Released checkpoints: '
                         'https://huggingface.co/DabinJeong/LocoGen . Note the `epoch` field '
                         'stored inside a payload is a resumed run index, not the real epoch.')
    ap.add_argument('--ms_config', type=str, default=None,
                    help='Separator config matching --ms_ckpt. No default; its '
                         'masked_separator.target_idx must agree with --cell_type.')
    ap.add_argument('--diagnostic', action='store_true',
                    help='One-batch diagnostic mode: log alpha_t per reverse step. Skips eval/save.')
    ap.add_argument('--alpha_cap', type=float, default=1.0,
                    help='Upper cap on auto-balanced α_t. v1 used hardcoded 1.0; v2 score_mag/grad_mag fix lets cap go higher (5–20 range).')
    ap.add_argument('--sil_strength', type=float, default=0.0,
                    help='SIL gradient gating strength in [0,1]. 0=off, 1=fully gate by SIL mask. '
                         'Loads the same separator ckpt as --use_masked_separator (sets ms_model). '
                         'Gradient at position i is multiplied by (sil_strength*M_i + 1-sil_strength).')
    ap.add_argument('--sil_t_threshold', type=float, default=1.0,
                    help='Apply SIL gating only when sampling time t < threshold. '
                         '1.0 = always (legacy). 0.3 = only late sampling (x_hat_0 nearly one-hot, '
                         'separator gives meaningful masks). 0.0 = never. '
                         'Shared between SIL gradient gating and v6 OOD localized damping.')
    ap.add_argument('--ood_sil_localized', type=str2bool, default=False,
                    help='v6: apply MOOD OOD damping (lam > 0) ONLY at SIL-background '
                         'positions (mask < ood_sil_threshold). Functional motifs preserved.')
    ap.add_argument('--ood_sil_threshold', type=float, default=0.5,
                    help='SIL mask threshold; positions with M < threshold treated as background.')
    ap.add_argument('--ood_sil_invert', type=str2bool, default=False,
                    help='Ablation: flip the localization gate so OOD damping happens at FOREGROUND '
                         '(motif positions) instead of background. This is the "attack" ablation reported in '
                         'the main text. Sanity inverse — should hurt reward / atac_h4 '
                         'if localization matters. Requires --ood_sil_localized true.')
    ap.add_argument('--ood_global', type=str2bool, default=False,
                    help='Paper-faithful MOOD: apply (1-√λ) globally to ALL positions, not gated by is_masked. '
                         'Direct discrete analogue of MOOD Eq 4. With --ood_sil_localized true the localization '
                         'gate composes as `M < threshold` (no `is_masked` AND).')
    ap.add_argument('--ms_random', type=str2bool, default=False,
                    help='Ablation: re-initialize the MaskedSeparatorModel weights randomly after loading '
                         'the ckpt. Tests whether the trained separator carries useful information.')
    ap.add_argument('--ms_constant_rho', type=float, default=None,
                    help='Ablation: bypass the separator and use a constant mask M = rho everywhere. '
                         'Tests whether adaptive masking matters vs uniform background damping. '
                         'Set to e.g. 0.3 to mimic the locked init_rho. Disables ms_model entirely.')
    ap.add_argument('--debug_log', action='store_true',
                    help='Dump per-step alpha_t / score_mag / grad_mag CSV to logs/alpha_diagnostic_v2.csv even on full runs.')
    ap.add_argument('--cell_type', type=str, required=True, choices=list(ORACLE_HEAD.keys()),
                    help='Oracle head to use as reward (and for gold eval). 0=hepg2, 1=k562, 2=sknsh. '
                         'Required: there is no default, because the separator passed via '
                         '--ms_config/--ms_ckpt must target the same cell line.')
    args = ap.parse_args()
    print(args)
    head_idx = ORACLE_HEAD[args.cell_type]

    set_seed(args.seed, use_cuda=True)
    CKPT_PATH = os.path.join(args.base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.join(DRAKES_REPO, 'configs_gosai'), job_name='load_model')
    cfg = compose(config_name='config_gosai.yaml')
    cfg.eval.checkpoint_path = CKPT_PATH

    model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    oracle_model = oracle.get_gosai_oracle(mode='train').to(model.device)
    oracle_model.eval()
    for p in oracle_model.parameters():
        p.requires_grad_(False)

    oracle_eval = oracle.get_gosai_oracle(mode='eval').to(model.device)
    oracle_eval.eval()

    ms_model = None
    if args.ms_constant_rho is not None:
        # Constant-mask ablation overrides the separator entirely.
        print(f"[ablation] ms_constant_rho={args.ms_constant_rho} — separator bypassed, "
              f"mask = constant.")
    elif args.use_masked_separator or args.sil_strength > 0.0 or args.ood_sil_localized:
        ms_model = load_masked_separator(args.ms_config, args.ms_ckpt, model.device,
                                         cell_type=args.cell_type)
        if args.ms_random:
            # Random-separator ablation: reinitialize all weights, then refreeze.
            for p in ms_model.parameters():
                p.requires_grad_(True)
            for m in ms_model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            for p in ms_model.parameters():
                p.requires_grad_(False)
            ms_model.eval()
            print("[ablation] ms_random=True — separator weights re-initialized randomly.")
        if args.sil_strength > 0.0:
            print(f"[sil] gradient gating active, strength={args.sil_strength:.2f}")
        if args.ood_sil_localized:
            print(f"[v6] OOD damping localized to SIL background (lam={args.lam}, "
                  f"threshold={args.ood_sil_threshold}, t_thresh={args.sil_t_threshold}, "
                  f"invert={args.ood_sil_invert})")

    if args.diagnostic:
        log_list = []
        bsz = 32
        print(f"[diagnostic] one batch bsz={bsz} steps={args.num_steps} "
              f"lam={args.lam} alpha_ratio={args.alpha_ratio} alpha_cap={args.alpha_cap} "
              f"ms={int(args.use_masked_separator)}")
        _ = mood_sample(model, oracle_model, args.num_steps, args.lam, args.alpha_ratio,
                        bsz, ms_model=ms_model, log_list=log_list, alpha_cap=args.alpha_cap,
                        head_idx=head_idx, sil_strength=args.sil_strength,
                        use_phi_masking=args.use_masked_separator,
                        sil_t_threshold=args.sil_t_threshold,
                        ood_sil_localized=args.ood_sil_localized,
                        ood_sil_threshold=args.ood_sil_threshold,
                        ood_sil_invert=args.ood_sil_invert,
                        ms_constant_rho=args.ms_constant_rho,
                        ood_global=args.ood_global)
        log_path = './logs/alpha_diagnostic.csv'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        df = pd.DataFrame(log_list)
        df.to_csv(log_path, index=False)
        n = len(df)
        clamp_frac = float((df['alpha_t_raw'] > args.alpha_cap).mean())
        a_mean = float(df['alpha_t_raw'].mean())
        a_std = float(df['alpha_t_raw'].std())
        a_min = float(df['alpha_t_raw'].min())
        a_max = float(df['alpha_t_raw'].max())
        sm_mean = float(df['score_mag'].mean())
        gm_mean = float(df['grad_mag'].mean())
        print(f"[diagnostic] N={n} steps | cap={args.alpha_cap} | "
              f"clamp_active_frac={clamp_frac:.3f}")
        print(f"[diagnostic] alpha_t_raw mean={a_mean:.4f} std={a_std:.4f} "
              f"min={a_min:.4f} max={a_max:.4f}")
        print(f"[diagnostic] score_mag mean={sm_mean:.4f} | grad_mag mean={gm_mean:.4f}")
        print(f"[diagnostic] log -> {log_path}")
        return

    # ----- generate -----
    log_list = [] if args.debug_log else None
    step_times_ms = []  # populated by mood_sample for compute-cost instrumentation
    all_tokens = []
    n_done = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _gen_t0 = time.perf_counter()
    while n_done < args.num_samples:
        bsz = min(args.batch_size, args.num_samples - n_done)
        tok = mood_sample(model, oracle_model, args.num_steps, args.lam, args.alpha_ratio,
                          bsz, ms_model=ms_model, log_list=log_list, alpha_cap=args.alpha_cap,
                          head_idx=head_idx, sil_strength=args.sil_strength,
                          use_phi_masking=args.use_masked_separator,
                          sil_t_threshold=args.sil_t_threshold,
                          ood_sil_localized=args.ood_sil_localized,
                          ood_sil_threshold=args.ood_sil_threshold,
                          ood_sil_invert=args.ood_sil_invert,
                          ms_constant_rho=args.ms_constant_rho,
                          step_times_ms=step_times_ms,
                          ood_global=args.ood_global)
        all_tokens.append(tok.cpu())
        n_done += bsz
        print(f"  generated {n_done}/{args.num_samples}")
    _gen_wall_sec = time.perf_counter() - _gen_t0
    _peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
    tokens = torch.cat(all_tokens, dim=0)[:args.num_samples]
    seqs = tokens_to_dna(tokens)
    if args.debug_log and log_list:
        log_path = './logs/alpha_diagnostic.csv'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        pd.DataFrame(log_list).to_csv(log_path, index=False)
        print(f"[debug_log] per-step trace -> {log_path}")

    # ----- quick gold-oracle eval (no plausibility yet — that's M2's eval pipeline) -----
    with torch.no_grad():
        x_oh = F.one_hot(tokens.to(model.device), num_classes=4).float().transpose(1, 2)
        gold = oracle_eval(x_oh).squeeze(-1)
        if gold.ndim == 2:
            gold = gold[:, head_idx]
        gold_mu = float(gold.mean())
        gold_top = float(torch.topk(gold, k=min(100, gold.numel())).values.mean())

    # ----- save -----
    run_name = (f'lam{args.lam}_r{args.alpha_ratio}_ms{int(args.use_masked_separator)}'
                f'_{args.name}_{args.cell_type}_seed{args.seed}')
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    # Use column name 'seq' to match the eval pipeline's default (and the
    # gosai train CSV); the eval script's --seq_col flag is shared between
    # generated and train data, so they must agree.
    pd.DataFrame({'seq': seqs, 'gold_reward': gold.cpu().numpy()}).to_csv(
        os.path.join(out_dir, 'sequences.csv'), index=False
    )
    # Per-step latency stats
    if step_times_ms:
        _arr = np.array(step_times_ms, dtype=np.float64)
        step_stats = {
            'step_time_ms_mean': float(_arr.mean()),
            'step_time_ms_median': float(np.median(_arr)),
            'step_time_ms_p95': float(np.percentile(_arr, 95)),
            'n_steps_recorded': int(_arr.size),
        }
    else:
        step_stats = {}

    summary = {
        'run': run_name,
        'cell_type': args.cell_type,
        'head_idx': head_idx,
        'seed': args.seed,
        'lam': args.lam,
        'alpha_ratio': args.alpha_ratio,
        'use_masked_separator': args.use_masked_separator,
        'ood_sil_localized': args.ood_sil_localized,
        'ood_sil_invert': args.ood_sil_invert,
        'ood_global': args.ood_global,
        'ms_random': args.ms_random,
        'ms_constant_rho': args.ms_constant_rho,
        'ms_ckpt': args.ms_ckpt,
        'num_steps': args.num_steps,
        'n': args.num_samples,
        'gold_mu': gold_mu,
        'gold_top100': gold_top,
        # Compute cost instrumentation (added 2026-05-06 for ablation):
        'gen_wall_sec': float(_gen_wall_sec),
        'throughput_samples_per_sec': float(args.num_samples / _gen_wall_sec),
        'peak_mem_gb': float(_peak_mem_gb),
        **step_stats,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"[done] {run_name}: gold_mu={gold_mu:.4f} gold_top100={gold_top:.4f}")
    print(f"       saved to {out_dir}")


if __name__ == '__main__':
    main()
