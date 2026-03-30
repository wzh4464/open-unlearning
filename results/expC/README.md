# Experiment C: Noise Injection Ablation

Model: Llama-3.2-1B-Instruct | Data: TOFU forget01 | K=10 | Fisher HVP

## Ablation: Removal vs Noise

| Config | m_utility | f_quality | f_Prob | f_truth_ratio |
|--------|:---------:|:---------:|:------:|:-------------:|
| Original | 0.332 | 1.000 | 0.186 | 0.770 |
| Removal-Only (ε=0) | 0.331 | 1.000 | 0.179 | 0.772 |
| Full (iso ε=100) | 0.323 | 0.990 | 0.168 | 0.776 |
| Full (sub ε=1.0, k=1) | 0.321 | 0.990 | 0.167 | 0.781 |
| Noise-Only | 0.334 | 1.000 | 0.186 | 0.769 |

## Subspace ε Sweep (k=1, β=0.001, Δ_cert=0.01)

| ε | m_utility | f_quality | f_Prob | f_truth_ratio |
|---|:---------:|:---------:|:------:|:-------------:|
| 1.0 | 0.321 | 0.990 | 0.167 | 0.781 |
| 5.0 | 0.325 | 0.990 | 0.169 | 0.778 |
| 10.0 | N/A | 1.000 | 0.176 | 0.774 |

## Key Findings

1. **Removal is the core mechanism**: Removal-Only reduces f_Prob by 4%, noise adds another 6%
2. **Noise-Only has zero effect**: identical to Original (noise alone cannot target forget set)
3. **Subspace noise enables small ε**: ε=1.0 with subspace (k=1, β=0.001) achieves same result as isotropic ε=100, without model collapse
4. **Isotropic ε=1.0 crashes the model**: noise_norm = 1502% of model_norm
5. **ε=1.0 and ε=5.0 nearly identical**: β=0.001 makes noise very small regardless of ε

## Noise Calibration

σ = 2 × Δ_cert / ε × √(2·log(2.5/δ))

| Mode | ε | Δ_cert | β | σ_∥ | noise/model | Result |
|------|---|--------|---|-----|-------------|--------|
| Isotropic | 1.0 | 0.01 | — | 0.100 | 1502% | Crash |
| Isotropic | 100 | 0.001 | — | 1e-4 | 0.5% | OK |
| **Subspace k=1** | **1.0** | **0.01** | **0.001** | **0.100** | **0.5%** | **OK** |

Subspace noise concentrates noise in k-dim subspace. With β=0.001, the orthogonal
complement gets only 0.1% of σ_∥, reducing effective noise by ~1000× vs isotropic.

## Subspace k Sweep (ε=1.0, β=0.001, Δ_cert=0.01) — FIXED projector

| k | m_utility | f_quality | f_Prob | f_truth_ratio |
|---|:---------:|:---------:|:------:|:-------------:|
| 1 | 0.321 | 0.990 | 0.167 | 0.781 |
| 2 | 0.320 | 0.990 | 0.166 | 0.782 |
| 3 | 0.320 | 0.990 | 0.167 | 0.783 |
| 4 | 0.320 | 0.990 | 0.167 | 0.781 |
| 5 | 0.320 | 1.000 | 0.166 | 0.782 |

k=1-5 results are nearly identical because β=0.001 concentrates almost all
noise in the orthogonal complement, making the subspace rank k irrelevant.
The projector fix (float64 norm instead of QR) resolved the k>=2 crash
caused by numerical instability of torch.linalg.qr at 1.24B dimensions.
