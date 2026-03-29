# Measured Efficiency Comparison

Hardware: 1x H200 140GB | Model: Llama-3.2-3B | Data: TOFU forget10 (400 samples)

|                            | Retrain | GradDiff | NPO    | PDU    | UNDIAL | LMC K=10 | LMC K=50 |
|----------------------------|---------|----------|--------|--------|--------|----------|----------|
| **Training-time overhead** |         |          |        |        |        |          |          |
| Pre-comp time (min)        | —       | —        | —      | —      | —      | 18.5     | 18.5     |
| Pre-comp overhead (%)      | —       | —        | —      | —      | —      | +62%     | +62%     |
| On-disk storage (GB)       | Full D  | Full D   | Full D | Full D | Full D | 42       | 42       |
| **Deletion (per request)** |         |          |        |        |        |          |          |
| Latency (s)                | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 15.1     | 52.8     |
| Peak GPU (GB)              | 30.0    | 32.9     | 39.1   | 32.4   | 39.6   | 25.8     | 25.8     |
| **Total time (m requests)**|         |          |        |        |        |          |          |
| m=1 (min)                  | 1.4     | 0.3      | 0.7    | 0.3    | 0.3    | 0.3      | 0.9      |
| m=10 (min)                 | 14.4    | 2.7      | 6.7    | 2.7    | 3.1    | 2.5      | 8.8      |
| m=100 (min)                | 143.7   | 27.3     | 67.0   | 27.3   | 31.3   | 25.2     | 88.0     |
| m=400 (h)                  | 9.6     | 1.8      | 4.5    | 1.8    | 2.1    | 1.7      | 5.9      |
| **Amortized per request**  |         |          |        |        |        |          |          |
| m=1 (s)                    | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 1125.1   | 1162.8   |
| m=10 (s)                   | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 126.1    | 163.8    |
| m=100 (s)                  | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 26.2     | 63.9     |
| m=400 (s)                  | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 17.9     | 55.6     |
| m->inf (s)                 | 86.2    | 16.4     | 40.2   | 16.4   | 18.8   | 15.1     | 52.8     |
| **Speedup vs Retrain**     |         |          |        |        |        |          |          |
| Per request                | 1.0x    | 5.3x     | 2.1x   | 5.3x   | 4.6x   | 5.7x     | 1.6x     |
| m=400 total                | 1.0x    | 5.3x     | 2.1x   | 5.3x   | 4.6x   | 5.7x     | 1.6x     |

Notes:
- Pre-comp = sparse checkpoint logging during training (one-time, amortized over future requests)
- Full D = requires storing original training data for unlearning
- Break-even vs Retrain: m >= 16 requests
- All latencies measured in-GPU with warm model
