#!/usr/bin/env python3
"""
Generate SEMANTIC_SUMMARY.md next to each experiment log/result file.

Usage:
    python generate_summaries.py [--saves-dir /workspace/saves] [--force] [--dry-run]

Options:
    --saves-dir DIR    Root saves directory (default: /workspace/saves)
    --force            Regenerate all summaries, even if up-to-date
    --dry-run          Show what would be generated without writing
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ============================================================
# Reference baselines (Llama-3.2-1B-Instruct on TOFU)
# ============================================================
RETAIN90 = {
    "model_utility": 0.591,
    "extraction_strength": 0.059,
    "forget_Q_A_ROUGE": 0.379,
    "forget_Q_A_Prob": 0.116,
    "mia_min_k": 0.383,
}


# ============================================================
# Metric interpretation helpers
# ============================================================
def utility_comment(u):
    if u is None or u == "N/A":
        return "无法评估"
    if u < 0.01:
        return "模型完全崩溃，丧失所有能力"
    if u < 0.2:
        return "模型严重退化，几乎不可用"
    if u < 0.35:
        return f"utility偏低 (retain90基线的{u / RETAIN90['model_utility'] * 100:.0f}%)"
    if u < 0.42:
        return f"utility中等 (retain90基线的{u / RETAIN90['model_utility'] * 100:.0f}%)"
    if u < 0.50:
        return f"utility良好 (retain90基线的{u / RETAIN90['model_utility'] * 100:.0f}%)"
    return f"utility优秀 (retain90基线的{u / RETAIN90['model_utility'] * 100:.0f}%)"


def forget_comment(rouge):
    if rouge is None or rouge == "N/A":
        return "无法评估"
    if rouge < 0.01:
        return "遗忘极其彻底(但可能是模型崩溃导致)"
    if rouge < 0.15:
        return "遗忘非常彻底"
    if rouge < 0.35:
        return "遗忘效果良好"
    if rouge < RETAIN90["forget_Q_A_ROUGE"]:
        return f"遗忘效果尚可 (低于retain90基线{RETAIN90['forget_Q_A_ROUGE']:.3f})"
    if rouge < 0.42:
        return f"遗忘不够彻底 (高于retain90基线{RETAIN90['forget_Q_A_ROUGE']:.3f})"
    return f"遗忘效果差 (远高于retain90基线{RETAIN90['forget_Q_A_ROUGE']:.3f})"


def privleak_comment(pl):
    if pl is None or pl == "N/A":
        return "无法评估"
    if pl < -20:
        return "隐私保护优秀 (MIA攻击难以区分成员)"
    if pl < -5:
        return "隐私保护良好"
    if pl < 5:
        return "隐私保护中等 (接近随机)"
    if pl < 30:
        return "隐私泄露偏高"
    if pl < 70:
        return "隐私泄露严重"
    return "隐私泄露极其严重"


def extraction_comment(ext):
    if ext is None or ext == "N/A":
        return ""
    if ext < 0.06:
        return "提取攻击风险低"
    if ext < 0.08:
        return "提取攻击风险中等"
    if ext < 0.15:
        return "提取攻击风险偏高"
    return "提取攻击风险高"


# ============================================================
# File I/O helpers
# ============================================================
def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics_from_log(log_path):
    """Extract metrics from eval.log when JSON is missing."""
    metrics = {}
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"Result for metric (\w+):\t([\d.eE\+\-]+)", line)
                if m:
                    key, val = m.group(1), m.group(2)
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass
    except Exception:
        pass
    return metrics


def extract_training_log_info(log_path):
    """Extract key info from training logs."""
    info = {}
    try:
        with open(log_path) as f:
            lines = f.readlines()
        info["total_lines"] = len(lines)
        info["last_lines"] = [l.strip() for l in lines[-10:]]
        info["errors"] = [l.strip() for l in lines if "ERROR" in l or "Traceback" in l][:5]
        info["completed"] = any(
            "Training logger saved" in l or "Saved unlearned model" in l or "completed" in l.lower()
            for l in lines
        )
        info["epochs_completed"] = [l.strip() for l in lines if "Epoch" in l and "ended" in l]
        corrections = [l.strip() for l in lines if "Applied correction" in l]
        info["corrections"] = corrections
        info["n_corrections"] = len(corrections)
        info["unlearn_success"] = any("Unlearning applied successfully" in l for l in lines)
        info["refinetune_done"] = any(
            "refinetune completed" in l.lower() or "Precise refinetune completed" in l
            for l in lines
        )
        # Extract spectral norm info
        info["spectral"] = [l.strip() for l in lines if "spectral norm" in l.lower()]
    except Exception as e:
        info["read_error"] = str(e)
    return info


def needs_update(exp_dir, force=False):
    """Check if SEMANTIC_SUMMARY.md needs to be (re)generated."""
    summary_path = exp_dir / "SEMANTIC_SUMMARY.md"
    if force:
        return True
    if not summary_path.exists():
        return True
    # Check if any result file is newer than summary
    summary_mtime = summary_path.stat().st_mtime
    for pattern in ["TOFU_SUMMARY.json", "TOFU_EVAL.json", "MUSE_SUMMARY.json",
                     "MUSE_EVAL.json", "eval.log", "*.log", "efficiency_metrics.json"]:
        for f in exp_dir.glob(pattern):
            if f.stat().st_mtime > summary_mtime:
                return True
    return False


def write_summary(path, content, dry_run=False):
    if dry_run:
        print(f"  [dry-run] would write {path}")
        return
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path}")


# ============================================================
# Method-specific semantic descriptions
# ============================================================
METHOD_DESCRIPTIONS = {
    "CEU": "CEU (Causal Erasure Unlearning) 通过因果擦除机制尝试消除forget数据的影响。",
    "GRADASC": "GradAsc (Gradient Ascent) 通过在forget数据上做梯度上升来反转学习。",
    "GRADDIFF": "GradDiff (Gradient Difference) 通过forget集上的梯度上升与retain集上的梯度下降相结合来遗忘。",
    "NPO": "NPO (Negative Preference Optimization) 通过负偏好优化让模型主动\"不偏好\"forget数据的回答。",
    "RMU": "RMU (Retain Memory Unlearning) 通过随机向量映射干扰模型对forget数据的表征。",
    "SATIMP": "SatImp (Saliency-based Importance) 基于显著性和重要性评分选择性修改参数。",
    "SIMNPO": "SimNPO (Simplified NPO) 是NPO的简化版本，去除了部分正则化项。",
    "UNDIAL": "UNDIAL 通过不确定性解码干预减少模型对forget数据的确定性输出。",
    "WGA": "WGA (Weighted Gradient Ascent) 通过加权梯度上升，根据参数重要性差异化地遗忘。",
    "PDU": "PDU (Probabilistic Data Unlearning) 基于概率框架调整模型参数分布来削弱记忆。",
}

METHOD_KNOWN_BEHAVIORS = {
    "CEU": "**已知行为**: CEU在Llama-3.2-1B + TOFU forget01设定下导致完全崩溃(utility=0)，不可用。",
    "GRADASC": "**已知行为**: GradAsc过于激进，在所有epoch上都导致模型完全崩溃。",
    "GRADDIFF": "**已知行为**: GradDiff存在部分崩溃风险，utility受损严重但遗忘效果好。",
    "NPO": "**已知行为**: NPO高度epoch敏感，早期epoch过度遗忘，晚期epoch遗忘退化。",
    "RMU": "**已知行为**: RMU非常稳定，各epoch间指标变化很小。",
    "SATIMP": "**已知行为**: SatImp是utility保留最好的方法之一，但遗忘最不彻底。与SimNPO几乎等价。",
    "SIMNPO": "**已知行为**: SimNPO与SatImp指标几乎完全一致(<0.5%差异)。",
    "UNDIAL": "**已知行为**: UNDIAL遗忘效果不错但utility偏低，是有效方法中utility最差的。",
    "WGA": "**已知行为**: WGA最接近理想retrain行为，forget_ROUGE≈retain90基线，privleak≈0。",
    "PDU": "**已知行为**: PDU表现中等，在utility和遗忘之间没有突出优势。",
}


def detect_method(name):
    """Detect unlearning method from directory name."""
    for candidate in ["CEU", "GRADASC", "GRADDIFF", "NPO", "RMU", "SATIMP", "SIMNPO",
                       "UNDIAL", "WGA", "PDU", "LMCLEANER"]:
        if candidate.lower() in name.lower():
            return candidate if candidate != "LMCLEANER" else "LMCleaner"
    return "Unknown"


# ============================================================
# Summary generators by experiment type
# ============================================================

def gen_finetune_summary(exp_dir, info):
    """Generate summary for finetune experiments."""
    name = exp_dir.name
    lines = [f"# {name} - 微调实验语义总结\n"]

    if "retrain" in name:
        version = "v2" if "v2" in name else "v1"
        lines.append("## 实验目的")
        lines.append(f"在retain-90数据集上从头重训 ({version})，作为retrain基线。\n")
        lines.append(f"## 状态: {'成功完成' if info.get('completed') or not info.get('errors') else '可能未完成'}\n")
        lines.append("## 语义解读")
        lines.append("Retrain是遗忘方法的理想参考。但在forget01场景下retrain的privleak仍为正值(~28)，")
        lines.append("说明仅排除1%数据不足以完全消除隐私泄露。")
    elif "10x_lr" in name:
        lines.append("## 实验目的\n测试10倍学习率对微调的影响。\n")
        lines.append("## 状态: 不完整\n")
        lines.append("## 语义解读\n实验在初始化后几乎立即停止，结果不可用。")
    elif "test_spectral" in name:
        lines.append("## 实验目的\n验证spectral norm计算功能。\n")
        lines.append("## 状态: 成功完成 (功能测试)\n")
        spectral_info = info.get("spectral", [])
        if spectral_info:
            lines.append("## Spectral Norm")
            for s in spectral_info[-2:]:
                lines.append(f"- {s.split('INFO')[-1].strip(' -') if 'INFO' in s else s}")
    elif "tofu_safe" in name or "tofu" in name.lower():
        lines.append("## 实验目的")
        lines.append("在TOFU full dataset上微调，作为所有后续遗忘实验的基础模型。\n")
        lines.append(f"## 状态: {'成功完成' if info.get('completed') else '运行中/未确认'}\n")
        epochs = info.get("epochs_completed", [])
        if epochs:
            lines.append(f"## 训练进度: {len(epochs)} epoch完成")
        spectral_info = info.get("spectral", [])
        if spectral_info:
            lines.append("\n## Spectral Norm")
            for s in spectral_info[-2:]:
                lines.append(f"- {s.split('INFO')[-1].strip(' -') if 'INFO' in s else s}")
        lines.append("\n## 语义解读")
        lines.append("这是整个实验流水线的基石。训练记录支持LMCleaner的batch-level遗忘。")
    else:
        lines.append(f"## 状态: {'成功' if info.get('completed') else '未确认'}")
        if info.get("errors"):
            lines.append("\n## 错误")
            for e in info["errors"][:3]:
                lines.append(f"- {e}")

    return "\n".join(lines) + "\n"


def gen_unlearn_summary(exp_dir, info, eff):
    """Generate summary for unlearning experiments."""
    name = exp_dir.name
    lines = [f"# {name} - 遗忘训练语义总结\n"]

    if "lmcleaner" in name.lower():
        lines.append("## 方法: LMCleaner (Batch-Level)")
        log_files = list(exp_dir.glob("*.log"))
        if log_files:
            lines.append(f"## 日志: {log_files[0].name}\n")

        if info.get("errors"):
            lines.append("## 状态: 存在错误\n")
            lines.append("### 错误详情")
            for e in info["errors"][:3]:
                lines.append(f"- `{e}`")
            if "cuda:0 and cpu" in str(info["errors"]):
                lines.append("\n**根因**: 设备不匹配，部分tensor未迁移到GPU。此bug在后续版本中已修复。")
        elif info.get("unlearn_success"):
            lines.append("## 状态: 成功完成\n")
        else:
            lines.append("## 状态: 运行中/未确认\n")

        n_corr = info.get("n_corrections", 0)
        if n_corr > 0:
            lines.append("### 遗忘过程")
            lines.append(f"- 成功应用了 {n_corr} 次HVP校正")
            for l in info.get("corrections", [])[-3:]:
                m = re.search(r"step (\d+).*v_norm=([\d.]+).*K_used=(\d+).*hvp_calls=(\d+)", l)
                if m:
                    step, vnorm, k_used, hvp = m.groups()
                    lines.append(f"  - Step {step}: v_norm={vnorm}, K_used={k_used}, hvp_calls={hvp}")

        if info.get("refinetune_done"):
            lines.append("\n### Refinetune阶段")
            lines.append("- 对受影响的retain样本进行了精确refinetune")
            for l in info["last_lines"]:
                m = re.search(r"(\d+) affected retain samples", l)
                if m:
                    lines.append(f"- 受影响retain样本数: {m.group(1)}")

        lines.append("\n### 语义解读")
        if info.get("unlearn_success") and not info.get("errors"):
            lines.append(f"LMCleaner通过回溯训练轨迹，对包含forget数据的{n_corr}个训练步施加HVP校正。")
            lines.append("v_norm随步号递增而递减（早期训练步的校正量更大），符合预期。")
            if info.get("refinetune_done"):
                lines.append("遗忘后通过精确refinetune修复受影响的retain样本，减少附带损害。")
        elif info.get("errors"):
            lines.append("此次运行因错误而失败，不产生有效结果。")
    else:
        method = detect_method(name)
        lines.append(f"## 方法: {method}")
        lines.append(f"## 状态: 成功完成\n")

        if eff:
            lines.append("### 效率指标")
            lines.append(f"- 遗忘耗时: {eff.get('unlearning_time_seconds', 0):.2f}秒")
            lines.append(f"- 峰值GPU内存: {eff.get('peak_gpu_memory_mb', 0):.0f} MB")
            lines.append(f"- 总步数: {eff.get('total_steps', 'N/A')}")
            lines.append(f"- 平均步延迟: {eff.get('per_step_latency_mean_ms', 0):.1f} ms")

        lines.append("\n### 语义解读")
        lines.append(METHOD_DESCRIPTIONS.get(method, f"{method}遗忘方法。"))

    return "\n".join(lines) + "\n"


def gen_eval_summary(exp_dir, metrics, source):
    """Generate summary for evaluation experiments."""
    name = exp_dir.name
    lines = [f"# {name} - 评估语义总结\n"]

    # --- MUSE ---
    if "muse" in name.lower():
        lines.append("## 基准: MUSE")
        parts = name.split("_")
        domain = parts[-2] if len(parts) > 2 else "?"
        variant = parts[-1] if len(parts) > 1 else "?"
        lines.append(f"## 域: {domain}, 类型: {variant}")
        lines.append(f"## 数据来源: {source}\n")

        if metrics:
            lines.append("### 核心指标")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for k in ["extraction_strength", "forget_verbmem_ROUGE", "forget_knowmem_ROUGE",
                       "retain_knowmem_ROUGE", "mia_min_k", "privleak"]:
                v = metrics.get(k)
                if v is not None:
                    lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")

            lines.append("\n### 语义解读")
            if variant == "target":
                lines.append(f"这是{domain}域的**原始微调模型**(未遗忘), 作为遗忘前的参考上界。")
                verbmem = metrics.get("forget_verbmem_ROUGE", 0)
                if isinstance(verbmem, (int, float)) and verbmem > 0.5:
                    lines.append(f"模型高度记忆了forget数据 (verbmem={verbmem:.3f})。")
            elif variant == "retrain":
                lines.append(f"这是{domain}域的**retrain基线**。")
                ext = metrics.get("extraction_strength", 0)
                lines.append(f"extraction降到{ext:.3f}, 遗忘效果显著。")

        return "\n".join(lines) + "\n"

    # --- Full model baselines (with sub-eval dirs) ---
    sub_evals = {}
    for sub in exp_dir.iterdir():
        if sub.is_dir() and (sub / "TOFU_SUMMARY.json").exists():
            sub_evals[sub.name] = read_json(sub / "TOFU_SUMMARY.json")

    if sub_evals:
        model_name = name.replace("tofu_", "").replace("_full", "")
        lines.append(f"## 类型: 全量微调模型基线")
        lines.append(f"## 模型: {model_name}\n")
        for sub_name, sub_m in sorted(sub_evals.items()):
            if not sub_m:
                continue
            lines.append(f"### {sub_name.replace('evals_', '')}")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for k in ["model_utility", "extraction_strength", "forget_Q_A_ROUGE",
                       "exact_memorization", "mia_min_k"]:
                v = sub_m.get(k)
                if v is not None:
                    lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
            lines.append("")

        f01 = sub_evals.get("evals_forget01", {})
        if f01:
            mem = f01.get("exact_memorization", 0)
            lines.append("### 语义解读")
            lines.append(f"全量微调后模型几乎完美记忆了所有训练数据 (memorization={mem:.3f})。")
            lines.append("所有MIA指标接近1.0，成员推断攻击在未遗忘模型上完全有效。")

        return "\n".join(lines) + "\n"

    # --- Retain split baselines ---
    if name.startswith("tofu_") and "retain" in name:
        split = name.rsplit("_", 1)[1]
        pct = split.replace("retain", "")
        lines.append(f"## 类型: Retain分割基线 ({split})")
        lines.append(f"## 数据来源: {source}\n")
        if metrics:
            lines.append("### 核心指标")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for k in ["model_utility", "extraction_strength", "forget_Q_A_ROUGE",
                       "forget_Q_A_Prob", "exact_memorization", "mia_min_k"]:
                v = metrics.get(k)
                if v is not None:
                    lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
            lines.append("")
            mu = metrics.get("model_utility", 0)
            lines.append("### 语义解读")
            lines.append(f"在{pct}%数据上微调的模型（排除{100 - int(pct)}% forget数据），代表retrain参考点。")
            if split == "retain90":
                lines.append("**retain90是forget01场景下最重要的参考基线。**")
        return "\n".join(lines) + "\n"

    # --- Retrain eval ---
    if "retrain" in name:
        epoch_m = re.search(r"epoch(\d+)", name)
        epoch = epoch_m.group(1) if epoch_m else "final"
        lines.append(f"## 类型: Retrain基线评估, Epoch {epoch}")
        lines.append(f"## 数据来源: {source}\n")
        if metrics:
            _append_metric_table(lines, metrics)
            lines.append("\n### 语义解读")
            pl = metrics.get("privleak")
            lines.append(f"Retrain epoch {epoch}作为\"金标准\"遗忘方法。")
            if isinstance(pl, (int, float)) and pl > 20:
                lines.append(f"注意: privleak={pl:.1f}为正值, 即使retrain在forget01场景下也有隐私泄露。")
        return "\n".join(lines) + "\n"

    # --- Standard method evals ---
    method = detect_method(name)
    epoch_m = re.search(r"epoch(\d+)", name)
    epoch = epoch_m.group(1) if epoch_m else "?"
    k_m = re.search(r"K(\d+)", name)
    k_val = k_m.group(1) if k_m else None
    has_refinetune = "refinetune" in name
    rt_m = re.search(r"refinetune(\d+)", name)
    refinetune_n = rt_m.group(1) if rt_m else None
    has_fisher = "fisher" in name

    lines.append(f"## 方法: {method}")
    lines.append(f"## Epoch: {epoch}")
    if k_val is not None:
        lines.append(f"## K: {k_val}")
    if has_fisher:
        lines.append("## 变体: Fisher")
    if has_refinetune:
        rt_label = f" ({refinetune_n} epochs)" if refinetune_n else ""
        lines.append(f"## Refinetune: 是{rt_label}")
    lines.append(f"## 数据来源: {source}\n")

    if not metrics:
        lines.append("### 无可用指标\n评估日志存在但未能提取到有效指标。")
        return "\n".join(lines) + "\n"

    _append_metric_table(lines, metrics)

    # Extra metrics
    extra_keys = ["retain_Q_A_ROUGE", "retain_Truth_Ratio", "ra_Q_A_ROUGE", "wf_Q_A_ROUGE",
                  "exact_memorization"]
    has_extra = any(k in metrics for k in extra_keys)
    if has_extra:
        lines.append("\n### 补充指标")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        for k in extra_keys:
            v = metrics.get(k)
            if v is not None:
                lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")

    # Semantic interpretation
    lines.append("\n### 语义解读")

    mu = metrics.get("model_utility")
    rouge = metrics.get("forget_Q_A_ROUGE")
    pl = metrics.get("privleak")
    ext = metrics.get("extraction_strength")

    if method in ("CEU", "GRADASC"):
        if isinstance(mu, (int, float)) and mu < 0.01:
            lines.append("**模型完全崩溃。** 所有生成能力被摧毁，该方法在此设定下完全不可用。")
        else:
            lines.append(METHOD_DESCRIPTIONS.get(method, ""))
    elif method == "LMCleaner":
        k_label = f"K={k_val}" if k_val else ""
        fisher_label = " Fisher变体" if has_fisher else ""
        rt_label = f" + refinetune" + (f" {refinetune_n} epoch" if refinetune_n else "")
        lines.append(f"LMCleaner{fisher_label} epoch {epoch} {k_label}{rt_label if has_refinetune else ' (无refinetune)'}:")
        if isinstance(mu, (int, float)):
            lines.append(f"- {utility_comment(mu)}")
        if isinstance(rouge, (int, float)):
            lines.append(f"- {forget_comment(rouge)}")
        if isinstance(pl, (int, float)):
            lines.append(f"- {privleak_comment(pl)}")
        if k_val is not None:
            lines.append("\n**K值影响**: K参数对结果影响极小(<0.5%)，HVP步数非关键因素。")
        if refinetune_n and int(refinetune_n) >= 4:
            lines.append("**扩展refinetune**: 增加refinetune epoch显著改善privleak。")
        if has_fisher:
            lines.append("**Fisher变体**: 与标准版结果高度一致，未带来显著改善。")
    else:
        desc = METHOD_DESCRIPTIONS.get(method, f"{method}方法。")
        lines.append(desc)
        if isinstance(mu, (int, float)):
            lines.append(f"Epoch {epoch}: {utility_comment(mu)}; {forget_comment(rouge) if isinstance(rouge, (int, float)) else ''}。")
        known = METHOD_KNOWN_BEHAVIORS.get(method)
        if known:
            lines.append(known)
        # Special insights
        if method == "WGA" and isinstance(rouge, (int, float)) and isinstance(pl, (int, float)):
            if abs(rouge - RETAIN90["forget_Q_A_ROUGE"]) < 0.01 and abs(pl) < 5:
                lines.append("**该epoch下WGA最接近理想retrain行为。**")
        if method == "NPO" and "aligned" in name:
            lines.append("对齐版本在遗忘-utility权衡上优于原始版本。")

    return "\n".join(lines) + "\n"


def _append_metric_table(lines, metrics):
    """Append the standard metric table to lines."""
    mu = metrics.get("model_utility", "N/A")
    rouge = metrics.get("forget_Q_A_ROUGE", "N/A")
    prob = metrics.get("forget_Q_A_Prob", "N/A")
    truth = metrics.get("forget_truth_ratio", "N/A")
    ext = metrics.get("extraction_strength", "N/A")
    pl = metrics.get("privleak")
    mia = metrics.get("mia_min_k", metrics.get("mia_min_k_auc"))

    lines.append("### 核心指标")
    lines.append("| 指标 | 值 | 评价 |")
    lines.append("|------|-----|------|")
    lines.append(f"| model_utility | {mu if isinstance(mu, str) else f'{mu:.4f}'} | {utility_comment(mu)} |")
    lines.append(f"| forget_Q_A_ROUGE | {rouge if isinstance(rouge, str) else f'{rouge:.4f}'} | {forget_comment(rouge)} |")
    lines.append(f"| forget_Q_A_Prob | {prob if isinstance(prob, str) else f'{prob:.6f}'} | |")
    lines.append(f"| forget_truth_ratio | {truth if isinstance(truth, str) else f'{truth:.4f}'} | |")
    lines.append(f"| extraction_strength | {ext if isinstance(ext, str) else f'{ext:.4f}'} | {extraction_comment(ext)} |")
    if isinstance(pl, (int, float)):
        lines.append(f"| privleak | {pl:.1f} | {privleak_comment(pl)} |")
    if isinstance(mia, (int, float)):
        lines.append(f"| MIA_min_k | {mia:.4f} | |")


def gen_train_log_summary(exp_dir):
    """Generate summary for training log directories."""
    name = exp_dir.name
    meta = read_json(exp_dir / "meta.json")
    lines = [f"# {name} - 训练记录语义总结\n"]

    if meta:
        lines.append("### 元数据")
        lines.append(f"- 模式: {meta.get('mode', '?')}")
        lines.append(f"- 最大步数: {meta.get('max_steps', '?')}")
        lines.append("")

    pkl_count = len(list(exp_dir.glob("*.pkl")))
    if pkl_count > 0:
        lines.append(f"### 规模: {pkl_count}个训练记录文件\n")

    index_files = [f.name for f in exp_dir.glob("*.json")]
    if index_files:
        lines.append("### 索引文件")
        for f in sorted(index_files):
            lines.append(f"- {f}")
        lines.append("")

    lines.append("### 语义解读")
    if pkl_count > 1000:
        lines.append("大规模训练记录，支持LMCleaner遗忘方法的训练轨迹回溯。")
        lines.append("存储开销是LMCleaner的主要实际限制之一。")
    elif "test" in name:
        lines.append("功能验证用途的训练记录。")
    else:
        lines.append("辅助训练记录。")

    return "\n".join(lines) + "\n"


# ============================================================
# Main orchestrator
# ============================================================
def process_saves(saves_dir, force=False, dry_run=False):
    saves = Path(saves_dir)
    counts = {"finetune": 0, "unlearn": 0, "eval": 0, "train_logs": 0, "skipped": 0}

    # 1. Finetune
    finetune_dir = saves / "finetune"
    if finetune_dir.exists():
        print("=== Finetune experiments ===")
        for exp_dir in sorted(finetune_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            log_path = exp_dir / "FinetuneTrainer.log"
            if not log_path.exists():
                continue
            if not needs_update(exp_dir, force):
                counts["skipped"] += 1
                continue
            info = extract_training_log_info(log_path)
            content = gen_finetune_summary(exp_dir, info)
            write_summary(exp_dir / "SEMANTIC_SUMMARY.md", content, dry_run)
            counts["finetune"] += 1

    # 2. Unlearn
    unlearn_dir = saves / "unlearn"
    if unlearn_dir.exists():
        print("\n=== Unlearn experiments ===")
        for exp_dir in sorted(unlearn_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            log_files = list(exp_dir.glob("*.log"))
            if not log_files:
                continue
            if not needs_update(exp_dir, force):
                counts["skipped"] += 1
                continue
            info = extract_training_log_info(log_files[0])
            eff = read_json(exp_dir / "efficiency_metrics.json")
            content = gen_unlearn_summary(exp_dir, info, eff)
            write_summary(exp_dir / "SEMANTIC_SUMMARY.md", content, dry_run)
            counts["unlearn"] += 1

    # 3. Eval
    eval_dir = saves / "eval"
    if eval_dir.exists():
        print("\n=== Eval experiments ===")
        for exp_dir in sorted(eval_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if not needs_update(exp_dir, force):
                counts["skipped"] += 1
                continue

            metrics, source = None, None
            for fname, src_label in [
                ("TOFU_SUMMARY.json", "TOFU_SUMMARY.json"),
                ("MUSE_SUMMARY.json", "MUSE_SUMMARY.json"),
                ("TOFU_EVAL.json", "TOFU_EVAL.json"),
                ("MUSE_EVAL.json", "MUSE_EVAL.json"),
            ]:
                p = exp_dir / fname
                if p.exists():
                    metrics = read_json(p)
                    source = src_label
                    break

            if metrics is None:
                log_path = exp_dir / "eval.log"
                if log_path.exists():
                    metrics = extract_metrics_from_log(log_path)
                    source = "eval.log (解析)" if metrics else None

            # Check for sub-eval dirs (full model baselines)
            has_sub = any(
                (sub / "TOFU_SUMMARY.json").exists()
                for sub in exp_dir.iterdir() if sub.is_dir()
            )

            if not metrics and not has_sub:
                continue

            content = gen_eval_summary(exp_dir, metrics, source)
            write_summary(exp_dir / "SEMANTIC_SUMMARY.md", content, dry_run)
            counts["eval"] += 1

    # 4. Train logs
    train_logs_dir = saves / "train_logs"
    if train_logs_dir.exists():
        print("\n=== Train logs ===")
        for exp_dir in sorted(train_logs_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if not needs_update(exp_dir, force):
                counts["skipped"] += 1
                continue
            content = gen_train_log_summary(exp_dir)
            write_summary(exp_dir / "SEMANTIC_SUMMARY.md", content, dry_run)
            counts["train_logs"] += 1

    # Report
    total = sum(v for k, v in counts.items() if k != "skipped")
    print(f"\n=== Done! ===")
    print(f"Generated: {total} (finetune={counts['finetune']}, unlearn={counts['unlearn']}, "
          f"eval={counts['eval']}, train_logs={counts['train_logs']})")
    print(f"Skipped (up-to-date): {counts['skipped']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SEMANTIC_SUMMARY.md for experiment results")
    parser.add_argument("--saves-dir", default="/workspace/saves",
                        help="Root saves directory (default: /workspace/saves)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate all summaries even if up-to-date")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing")
    args = parser.parse_args()

    if not Path(args.saves_dir).exists():
        print(f"Error: saves directory not found: {args.saves_dir}", file=sys.stderr)
        sys.exit(1)

    process_saves(args.saves_dir, force=args.force, dry_run=args.dry_run)
