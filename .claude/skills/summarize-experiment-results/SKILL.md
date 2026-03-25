---
name: summarize-experiment-results
description: Use when new experiments finish, user asks to summarize results, or asks to generate semantic summaries for unlearning/eval/finetune experiment logs. Trigger on "总结实验", "语义总结", "summarize results", "summarize logs".
---

# Summarize Experiment Results

Generate a `SEMANTIC_SUMMARY.md` next to each experiment's log/result file, interpreting raw metrics into human-readable insights.

## Process

1. **Detect new results** — Find experiment directories that either lack `SEMANTIC_SUMMARY.md` or whose summary is older than the latest result file:
   ```bash
   # Find dirs with results but no summary
   find "$SAVES_DIR" -name "TOFU_SUMMARY.json" -o -name "TOFU_EVAL.json" -o -name "MUSE_SUMMARY.json" -o -name "*.log" | \
     xargs -I{} dirname {} | sort -u | while read d; do
       [ ! -f "$d/SEMANTIC_SUMMARY.md" ] && echo "NEW: $d"
     done
   ```

2. **Run the generator script** — Execute the bundled `generate_summaries.py`:
   ```bash
   python ~/.claude/skills/summarize-experiment-results/generate_summaries.py [--saves-dir /workspace/saves] [--force]
   ```
   - Default `--saves-dir`: `/workspace/saves`
   - `--force`: Regenerate all summaries, even existing ones
   - Script auto-detects experiment type from directory structure and file contents

3. **Report** — Show count of generated/updated summaries per category (finetune/unlearn/eval/train_logs)

## When to Use

- After batch experiments complete (e.g., parameter sweep finishes)
- User says "总结一下结果" / "生成语义总结" / "summarize results"
- When reviewing experiment outcomes before writing a paper or report
- After `run_all_methods_eval.sh` or similar eval scripts finish

## Metric Interpretation Reference

### TOFU Benchmarks — Key Metrics

| Metric | What it measures | Good direction | Retain90 baseline (Llama-3.2-1B) |
|--------|-----------------|----------------|-----------------------------------|
| `model_utility` | Overall model capability | Higher = better | 0.591 |
| `forget_Q_A_ROUGE` | How much forget knowledge remains | Lower = better unlearning | 0.379 |
| `forget_Q_A_Prob` | Probability of generating forget answers | Lower = better | 0.116 |
| `forget_truth_ratio` | Truth ratio on forget set | Depends on context | — |
| `extraction_strength` | Vulnerability to extraction attacks | Lower = safer | 0.059 |
| `privleak` | Privacy leakage (MIA distinguishability) | More negative = better | 23.5 |
| `mia_min_k` | Min-K MIA attack AUC | Closer to 0.5 = better | 0.383 |

### Interpretation Thresholds

**model_utility**:
- `< 0.01` → model collapsed (all capability destroyed)
- `< 0.20` → severely degraded
- `< 0.35` → low (unlearning too aggressive)
- `0.35–0.42` → moderate
- `0.42–0.50` → good
- `> 0.50` → excellent

**forget_Q_A_ROUGE** (compare to retain90 = 0.379):
- Below retain90 → unlearning effective
- Above retain90 → insufficient unlearning

**privleak**:
- `< -20` → excellent privacy protection
- `-20 to -5` → good
- `-5 to 5` → neutral
- `5 to 30` → leakage concern
- `> 30` → severe leakage

### MUSE Benchmarks

| Metric | Meaning |
|--------|---------|
| `forget_verbmem_ROUGE` | Verbatim memorization of forget data |
| `forget_knowmem_ROUGE` | Knowledge-level memorization |
| `retain_knowmem_ROUGE` | Retained knowledge preservation |

### Method-Specific Known Behaviors

| Method | Known behavior |
|--------|---------------|
| CEU | Causes total model collapse (utility=0) on Llama-3.2-1B |
| GradAsc | Same collapse — too aggressive |
| GradDiff | Partial collapse, high privleak |
| SatImp ≈ SimNPO | Nearly identical results, best utility but weakest unlearning |
| WGA | Closest to ideal retrain behavior |
| RMU | Very stable across epochs |
| NPO | Highly epoch-sensitive |
| LMCleaner | K parameter has <0.5% impact; extended refinetune helps privleak most |

## Summary File Format

Each `SEMANTIC_SUMMARY.md` should contain:

```markdown
# {experiment_name} - {类型}语义总结

## 方法: {method}
## Epoch: {N}
## 数据来源: {source_file}

### 核心指标
| 指标 | 值 | 评价 |
|------|-----|------|
| model_utility | X.XXXX | {utility解读，含retain90百分比} |
| forget_Q_A_ROUGE | X.XXXX | {遗忘效果解读} |
| ... | ... | ... |

### 语义解读
{2-4句话总结该实验的意义、与基线的对比、关键发现}
```

## Notes

- Script is idempotent — safe to run multiple times
- Summaries are gitignored by convention (generated artifacts)
- The global summary at `$SAVES_DIR/FULL_RESULTS_SUMMARY.md` is maintained separately
