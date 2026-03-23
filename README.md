<div align="center">

  <h1>🧗 AIMER</h1>
  <h3>AIMER: Calibration-Free Task-Agnostic MoE Pruning</h3>

  <p>
    <a href="https://arxiv.org/abs/2603.18492">
      <img src="https://img.shields.io/badge/arXiv-2603.18492-b31b1b?logo=arxiv&logoColor=white" alt="Paper">
    </a>
    <a href="https://zongfangliu.github.io/AIMER_Web/">
      <img src="https://img.shields.io/badge/Project-Website-0f766e?logo=googlechrome&logoColor=white" alt="Website">
    </a>
    <a href="pyproject.toml">
      <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
    </a>
    <a href="pyproject.toml">
      <img src="https://img.shields.io/badge/transformers-5.0.0-orange" alt="Transformers 5.0.0">
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-Apache%202.0-black" alt="License: Apache 2.0">
    </a>
  </p>

  <p><strong>Absolute mean over root mean square IMportance for Expert Ranking</strong></p>

<p>
  <a href="https://arxiv.org/abs/2603.18492"><strong>Read the Paper</strong></a>
  ·
  <a href="https://zongfangliu.github.io/AIMER_Web/"><strong>Visit the Project Website</strong></a>
</p>

</div>

AIMER, short for **Absolute mean over root mean square IMportance for Expert Ranking**, is a simple, weight-only criterion for post-training pruning of mixture-of-experts (MoE) language models. Instead of collecting router statistics or activation traces on a calibration set, AIMER ranks experts directly from pretrained weights, prunes them uniformly layer by layer, saves a standard Hugging Face checkpoint, and supports downstream evaluation with the same benchmark stack used in the paper.

## 🔍 Overview

This repository gives you a minimal end-to-end pipeline:

```text
pretrained MoE checkpoint
        |
        v
  AIMER expert scoring
        |
        v
uniform layer-wise pruning
        |
        v
  pruned HF checkpoint
        |
        v
evaluation with lm-eval / EvalPlus / LiveCodeBench / EvalScope / WildBench
```

## ✨ Highlights

- Calibration-free pruning. No calibration corpus, router statistics, or activation collection is required for AIMER scoring.
- Weight-only ranking. Expert scores are computed directly from pretrained expert parameters.
- Uniform expert pruning. The released CLI prunes the same number of experts from each detected MoE layer.
- Standard checkpoint export. Pruned models are saved with `save_pretrained(...)` and can be evaluated directly.
- Paper-aligned evaluation wrappers. The repo includes scripts for multiple-choice QA, coding, math, and WildBench-style evaluation.

## 🗂️ Repository At A Glance

| Path | Purpose |
| --- | --- |
| [`src/reap/calib_free_prune.py`](src/reap/calib_free_prune.py) | AIMER pruning CLI |
| [`src/reap/eval.py`](src/reap/eval.py) | Evaluation runner |
| [`src/reap/args.py`](src/reap/args.py) | Dataclass-based eval arguments |
| [`src/reap/model_util.py`](src/reap/model_util.py) | MoE model helpers and model-name patching |
| [`experiments/calib-free-cli.sh`](experiments/calib-free-cli.sh) | End-to-end prune + evaluate wrapper |
| [`experiments/eval.sh`](experiments/eval.sh) | Evaluation wrapper |
| [`scripts/build.sh`](scripts/build.sh) | Environment + submodule bootstrap |
| [`scripts/command_calibfree.sh`](scripts/command_calibfree.sh) | Minimal example invocation |
| [`config/`](config) | WildBench config templates |
| [`third-party/`](third-party) | Pinned evaluation dependencies |


## 🧩 MoE families in this repo



| `model_type` | Example checkpoint |
| --- | --- |
| `olmoe` | `allenai/OLMoE-1B-7B-0125-Instruct` |
| `ernie4_5_moe` | `baidu/ERNIE-4.5-21B-A3B-PT` |
| `qwen3_moe` | `Qwen/Qwen3-30B-A3B-Instruct-2507` |



## 🛠️ Installation


```bash
bash scripts/build.sh
```

## ⚡ Quick Start

### ✂️ Prune a model with AIMER

Use the pruning CLI directly when you only want the pruned checkpoint and score tables:

```bash
python src/reap/calib_free_prune.py \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --metric aimer \
  --sparsity-ratio 0.25 \
  --output-dir artifacts/calib-free/Qwen3-30B-A3B-Instruct-2507/aimer_0.250_uniform \
  --device-map auto \
  --torch-dtype bfloat16 \
  --metric-device auto \
  --local-files-only false \
  --trust-remote-code true
```

Behavior worth knowing:

- Higher AIMER scores are treated as more removable, so those experts are pruned.
- Pruning is always uniform across detected MoE layers.
- `--local-files-only` defaults to `true`, so set it to `false` if the model is not already cached locally.
- `--metric-device cpu` is useful if you want scoring to run from CPU copies after model load.

### ▶️ Prune and evaluate in one command

For the paper-style workflow, use the wrapper:

```bash
bash experiments/calib-free-cli.sh \
  0,1 \
  Qwen/Qwen3-30B-A3B-Instruct-2507 \
  aimer \
  0.25 \
  42
```

This wrapper:

1. sets `CUDA_VISIBLE_DEVICES`
2. prunes the checkpoint with AIMER
3. writes the pruned model under `artifacts/calib-free/...`
4. runs the evaluation pipeline

The first positional argument is the CUDA device mask. The wrapper also derives the serving port from the first visible GPU as `8300 + first_device`.

Minimal example:

```bash
bash scripts/command_calibfree.sh
```

Important default:

- [`experiments/calib-free-cli.sh`](experiments/calib-free-cli.sh) enables all benchmark groups by default, including WildBench.



## 📊 Evaluation

You can evaluate either an original checkpoint or a pruned checkpoint directly:

```bash
python src/reap/eval.py \
  --model-name artifacts/calib-free/Qwen3-30B-A3B-Instruct-2507/aimer_0.250_uniform \
  --seed 42 \
  --use-server true \
  --vllm_port 8300 \
  --run-lm-eval true \
  --run-evalplus true \
  --run-livecodebench true \
  --run-math true \
  --run-wildbench false \
  --results_dir artifacts/calib-free/Qwen3-30B-A3B-Instruct-2507/aimer_0.250_uniform/eval_vllm
```


## 📦 Output Layout

Pruned checkpoints are saved under:

```text
artifacts/calib-free/<model-name>/<metric>_<ratio>_uniform/
```

Typical contents:

| File | Description |
| --- | --- |
| `config.json`, model weights, tokenizer files | standard saved Hugging Face checkpoint |
| `pruned_experts.json` | pruned expert ids per layer |
| `pruning_plan.json` | per-layer kept/pruned counts |
| `calib_free_metadata.json` | pruning metadata |
| `calib_free_scores.csv` | raw per-expert AIMER scores |
| `calib_free_score_table.csv` | scores, within-layer rank, and prune decision |
| `eval_vllm/` or `eval_hf/` | evaluation outputs |
| `*.log` | wrapper logs when using the shell scripts |

## 🏗️ Project Structure

```text
AIMER/
|-- config/
|-- experiments/
|-- scripts/
|-- src/reap/
|   |-- args.py
|   |-- calib_free_prune.py
|   |-- eval.py
|   `-- model_util.py
`-- third-party/
```

## 🙏 Acknowledgement

This release builds on infrastructure adapted from [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap), the official codebase for **REAP: Router-weighted Expert Activation Pruning for SMoE compression**. We thank the REAP authors for open-sourcing their codebase and evaluation pipeline, which helped make this release possible.

## 📄 License

This repository is released under the Apache 2.0 License. See [`LICENSE`](LICENSE).
