# Measured Efficiency Comparison

Hardware: 1x H200 140GB | Model: Llama-3.2-3B | Data: TOFU forget10 (400 samples)

|                                 | Retrain | GradDiff | NPO    | PDU    | UNDIAL | LMC K=10 | LMC K=50 |
|---------------------------------|---------|----------|--------|--------|--------|----------|----------|
| On-disk storage (GB)            | 0       | <1       | <1     | <1     | <1     | 42       | 42       |
| **Deletion (per request)**      |         |          |        |        |        |          |          |
| Latency (s)                     | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 15.1     | 52.8     |
| Peak GPU (GB)                   | 30.0    | 32.9     | 39.1   | 32.4   | 39.6   | 25.8     | 25.8     |
| **Total time (m requests)**     |         |          |        |        |        |          |          |
| m=1 (min)                       | 1.4     | 0.3      | 0.7    | 0.3    | 0.3    | 0.3      | 0.9      |
| m=10 (min)                      | 14.4    | 2.7      | 6.7    | 2.7    | 3.1    | 2.5      | 8.8      |
| m=100 (min)                     | 143.7   | 27.3     | 67.0   | 27.3   | 31.3   | 25.2     | 88.0     |
| m=400 (h)                       | 9.6     | 1.8      | 4.5    | 1.8    | 2.1    | 1.7      | 5.9      |
| **Amortized per req (w/ pre)**  |         |          |        |        |        |          |          |
| m=1 (s)                         | 86.2    | 111.7    | 135.5  | 111.7  | 114.1  | 1305.1   | 1342.8   |
| m=10 (s)                        | 86.2    | 25.9     | 49.7   | 25.9   | 28.3   | 144.1    | 181.8    |
| m=100 (s)                       | 86.2    | 17.4     | 41.2   | 17.4   | 19.8   | 28.0     | 65.7     |
| m=400 (s)                       | 86.2    | 16.6     | 40.4   | 16.6   | 19.0   | 18.3     | 56.0     |
| m->inf (s)                      | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 15.1     | 52.8     |
| **Speedup vs Retrain**          |         |          |        |        |        |          |          |
| Per request                     | 1.0x    | 5.3x     | 2.1x   | 5.3x   | 4.6x   | 5.7x     | 1.6x     |
| m=400 total                     | 1.0x    | 5.3x     | 2.1x   | 5.3x   | 4.6x   | 5.7x     | 1.6x     |

Notes:
- On-disk: Baselines need original training data at unlearning time (<1GB for TOFU). LMCleaner needs 42GB sparse checkpoints (7 x 6GB), no original data.
- Deletion latency: wall-clock from receiving request to releasing unlearned model. All in-GPU, warm model.
- Amortized = (pre-comp + m x per-req) / m.
