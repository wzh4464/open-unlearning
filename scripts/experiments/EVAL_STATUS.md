# Evaluation Status Matrix

**Model**: Llama-3.2-1B-Instruct
**Benchmark**: TOFU (forget10/retain90)

## Status Legend
- ✅ Complete
- ⏳ In Progress
- ❌ Missing
- 🚫 No Model

## Summary Table

| Method | Epoch | Training | Basic | MIA | Complete |
|--------|-------|----------|-------|-----|----------|
| lmcleaner | 1 | ✅ | ✅ | ✅ | ✅ |
| lmcleaner | 2 | ✅ | ✅ | ✅ | ✅ |
| lmcleaner | 3 | ✅ | ✅ | ✅ | ✅ |
| lmcleaner | 4 | ✅ | ❌ | ❌ | ❌ |
| lmcleaner | 5 | ✅ | ❌ | ❌ | ❌ |
| graddiff | 1 | ✅ | ✅ | ✅ | ✅ |
| graddiff | 2 | ✅ | ✅ | ✅ | ✅ |
| graddiff | 3 | ✅ | ✅ | ✅ | ✅ |
| graddiff | 4 | ✅ | ✅ | ✅ | ✅ |
| graddiff | 5 | ✅ | ✅ | ✅ | ✅ |
| npo | 1 | ✅ | ✅ | ✅ | ✅ |
| npo | 2 | ✅ | ✅ | ✅ | ✅ |
| npo | 3 | ✅ | ✅ | ✅ | ✅ |
| npo | 4 | ✅ | ✅ | ✅ | ✅ |
| npo | 5 | ✅ | ✅ | ✅ | ✅ |
| gradasc | 1 | ✅ | ✅ | ✅ | ✅ |
| gradasc | 2 | ✅ | ✅ | ✅ | ✅ |
| gradasc | 3 | ✅ | ✅ | ✅ | ❌ |
| gradasc | 4 | ✅ | ✅ | ✅ | ❌ |
| gradasc | 5 | 🚫 | 🚫 | 🚫 | 🚫 |

## Pending Evaluations

| GPU | Task | Status |
|-----|------|--------|
| 0 | LMCleaner epoch 5 training | ⏳ |
| 2 | GradAscent 3-4 complete eval | ⏳ |
| 3 | LMCleaner epoch 4 training | ⏳ |
| - | LMCleaner epoch 4 all eval | 🔜 (after training) |
| - | LMCleaner epoch 5 all eval | 🔜 (after training) |

## Scripts

```bash
# Evaluation queue script
./scripts/experiments/eval_queue.sh <GPU> <TASK>

# Available tasks:
# lmcleaner4_basic, lmcleaner4_mia, lmcleaner4_complete
# lmcleaner5_basic, lmcleaner5_mia, lmcleaner5_complete

# Auto monitor (starts evals when training completes)
./scripts/experiments/auto_eval_monitor.sh
```
