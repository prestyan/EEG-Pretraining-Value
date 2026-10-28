# Evaluating Pretrained Weights in EEG Foundation Models

This repository contains the downstream evaluation and analysis code for studying the value of pretrained weights in EEG foundation models. It compares **Frozen**, **Fine-tuned**, and **Scratch** settings on BCIC-2A, BCIC-2B, SEED-IV, and pBCIW. The resulting Cohen's kappa scores are used to calculate **PRR**, **ACS**, and **PVS**.

Supported models: BENDR, BIOT, LaBraM, EEGPT, CBraMod, CodeBrain, and SingLEM.
For more detailed results and interactive visualizations, please visit our [project website](https://prestyan.github.io/EEG-Pretraining-Pages/).

## Repository structure

```text
.
├── analysis/
│   ├── run_analysis_bcic2a.py    # Pretrained-weight shuffling on BCIC-2A
│   ├── run_analysis_pBCIW.py     # Pretrained-weight shuffling on pBCIW
│   ├── run_samples_bcic2b.py     # Training-data-ratio experiments on BCIC-2B
│   └── weight_shuffle.py        # Shared weight-shuffling implementation
├── checkpoints/                 # Place pretrained checkpoints here
├── dataset/
│   ├── dataset.py              # Data loaders, subject splits, and input preparation
│   └── sea.py                  # Euclidean alignment utilities
├── logs/                       # Text results, CSV logs, and TensorBoard outputs
├── models/                     # Model definitions and backbone components
├── eval.py                     # Conventional performance metrics and evaluation helpers
├── utils.py                    # Shared neural-network utilities
├── run_main_bcic2a.py           # Main evaluation on BCIC-2A
├── run_main_bcic2b.py           # Main evaluation on BCIC-2B
├── run_main_seediv.py           # Main evaluation on SEED-IV
└── run_main_pBCIW.py            # Main evaluation on pBCIW
```

`dataset/` contains loading code, not the EEG recordings. The current `checkpoints/` and `logs/` directories are empty; obtain the datasets and pretrained weights separately. GitHub does not retain empty directories, so create them locally if needed.

## 1. Prepare the environment and data

The scripts use Python, PyTorch, PyTorch Lightning, and a CUDA GPU. Install a PyTorch build compatible with your CUDA environment, then install the other packages imported by the code:

```bash
python -m pip install pytorch-lightning timm numpy scipy scikit-learn pandas h5py mne einops opt-einsum tqdm pyhealth linear-attention-transformer tensorboard
```

Exact dependency versions are not pinned in this code snapshot. Use your validated training environment when reproducing results; the command above lists dependencies rather than a verified version lock. All backbones are imported by the runners, so BIOT's `linear-attention-transformer` dependency is also needed when another model is selected.

Prepare the MATLAB files expected by `dataset/dataset.py`:

| Dataset | Expected files | Required fields |
| --- | --- | --- |
| BCIC-2A / BCIC-2B | `sub1_train/Data.mat`, `sub1_test/Data.mat`, ..., `sub9_train/Data.mat`, `sub9_test/Data.mat` | `x_data`: trials × channels × samples; `y_data`: one integer label per trial |
| SEED-IV | Subject files such as `1_1.mat`, ..., `1_15.mat`; the subject ID is parsed from the second underscore-separated filename component | `data`: trials × channels × samples; `label`: one label per trial |
| pBCIW | `Sub_S1_1.mat`, ..., `Sub_S1_15.mat` | `S1Data`: channels × samples × trials, or trials × channels × samples; `S1Label`: one label per trial |

Labels should use zero-based class indices matching the classifier. Preserve the channel order expected by the loaders and the selected model. These runners consume prepared data; they do not automatically download or preprocess the original datasets.

The checkpoint filenames used by the runners are:

| Model | `model_choice` | File under `checkpoints/` |
| --- | --- | --- |
| BENDR | `BENDR` | `bendr.pth` |
| BIOT | `BIOT` | `biot_prest_16_channels.ckpt` |
| LaBraM | `LABRAM` | `labram-base.pth` |
| EEGPT | `EEGPT` | `eegpt_mcae_58chs_4s_large4E.ckpt` |
| CBraMod | `CBRAMOD` | `cbramod.pth` |
| CodeBrain | `CODEBRAIN` | `codebrain.pth` |
| SingLEM | `SINGLEM` | `singlem.pt` |

Use checkpoints matching the architecture and loading logic in the runner. Alternatively, update the corresponding `load_path_*` variable to point to your checkpoint.

## 2. Configure the three evaluation settings

Configuration is edited directly in the `if __name__ == "__main__":` block at the end of each runner; the scripts do not provide command-line options for these settings.

Before running, set:

- `data_path`: your local directory containing the prepared dataset.
- `model_choice`: one of the uppercase identifiers listed above.
- `load_path_*`: the matching pretrained checkpoint.
- `devices`: an available CUDA device, for example `devices=[0]` for a single-GPU machine.
- `batch_size`, `max_epochs`, and `max_lr`: the intended training configuration.
- The subject loop: `range(1, 10)` for all nine BCIC subjects, or `range(1, 16)` for the fifteen SEED-IV / pBCIW subjects. **The current BCIC-2A script is set to `range(7, 10)`; change it for a complete run.**

Choose the setting using the two flags passed to `LitModelCausal`:

| Setting | `freeze_backbone` | `is_load_pretrained` | Description |
| --- | --- | --- | --- |
| Frozen | `True` | `True` | Load pretrained weights, freeze the encoder, and train downstream layers. |
| Fine-tuned | `False` | `True` | Load pretrained weights and train the encoder and downstream layers. |
| Scratch | `False` | `False` | Randomly initialize and train the same architecture. |

For example, select fine-tuning with:

```python
litmodel = LitModelCausal(
    fold=i,
    model_choice=model_choice,
    dataset=dataset,
    load_path=load_path,
    freeze_backbone=False,
    is_load_pretrained=True,
)
```

Run each setting separately for each model and dataset. Keep the intended architecture, data protocol, and training configuration consistent across the comparison. For SingLEM, pretrained loading reads the encoder configuration from the checkpoint, whereas Scratch uses `Config()` defaults; check that these configurations match.

## 3. Run the main evaluation

Run commands from this repository directory, where `models/`, `dataset/`, and `checkpoints/` are located. If you kept the original enclosing folder name, enter it first:

```bash
cd "EEG-Pretraining-Value-main"
```

Create the output directories once:

```bash
python -c "from pathlib import Path; [Path('logs', d).mkdir(parents=True, exist_ok=True) for d in ['bcic2a', 'bcic2b', 'seediv', 'pBCIW']]; Path('checkpoints').mkdir(exist_ok=True)"
```

Execute the runner for the dataset you want to evaluate:

```bash
python run_main_bcic2a.py
python run_main_bcic2b.py
python run_main_seediv.py
python run_main_pBCIW.py
```

These are separate experiments; run the appropriate command after configuring its script. The subject loop holds out one subject at a time from downstream training.

Text results are appended to `logs/<dataset>/`, with setting suffixes `freeze`, `unfreeze`, and `unfreeze_random`. For example:

```text
logs/bcic2a/BENDR_bcic2a_freeze.txt
logs/bcic2a/BENDR_bcic2a_unfreeze.txt
logs/bcic2a/BENDR_bcic2a_unfreeze_random.txt
```

The scripts also create TensorBoard / CSV logs and Lightning checkpoints. Repeated runs append text results; use separate experiment folders or archive earlier logs to avoid mixing runs. Some TensorBoard / CSV directory names do not distinguish all three settings, so update their `name` or `version` when comparing multiple runs.

**Current epoch-selection behavior:** the runners call `trainer.fit(litmodel, train_loader, test_loader)`. The held-out subject is therefore used as Lightning's validation loader, and the reported best epoch is selected by its balanced accuracy. The `Valid` label in the text logs refers to that loader. For a protocol with independent model selection, use the source-subject validation set for selecting the epoch and evaluate the held-out subject separately. The supplied scripts should not be described as using an independent test set for epoch selection.

## 4. Run the analysis experiments

The `analysis/` directory contains the controlled analysis experiments. Configure the same paths, model, GPU, and evaluation flags as in the main runners. Run these scripts **as modules from the repository directory** so imports such as `models` and `analysis.weight_shuffle` resolve correctly.

```bash
python -c "from pathlib import Path; [Path('logs', d).mkdir(parents=True, exist_ok=True) for d in ['bcic2a_shuffle', 'pBCIW_shuffle', 'bcic2b_train_ratio']]"

# Pretrained-weight shuffling
python -m analysis.run_analysis_bcic2a
python -m analysis.run_analysis_pBCIW

# Training-data-ratio experiment
python -m analysis.run_samples_bcic2b
```

- **Weight shuffling:** the current runners test `shuffle_ratio` values of 0.25, 0.50, 0.75, and 1.00 with shuffle seed 2025. A ratio of 0 can be added as the unperturbed baseline. Run Frozen and Fine-tuned settings separately with the same original checkpoint and shuffle seed, then compare against the corresponding Scratch reference. Results are stored in `logs/bcic2a_shuffle/` and `logs/pBCIW_shuffle/`.
- **Perturbation scope:** `weight_shuffle.py` permutes selected entries within eligible encoder Linear / Conv / attention projection tensors. It preserves the selected tensors' stored value distributions and leaves downstream heads untouched. A ratio of 1 means all eligible projection entries, not every encoder parameter or random reinitialization. Reload the original checkpoint before each perturbation; repeated calls on one model compose the shuffles.
- **Training-data ratios:** `run_samples_bcic2b.py` currently uses `train_ratio` values of 0.25 and 0.50. Edit the list to evaluate other fractions, including 1.00 for the full-data reference. Results are stored in `logs/bcic2b_train_ratio/`.

The same epoch-selection behavior described above also applies to the analysis runners.

## 5. Calculate PRR, ACS, and PVS

The runners log conventional performance metrics, including balanced accuracy and Cohen's kappa. Calculate the proposed metrics after collecting the three settings. For the reported dataset-level comparison, first average kappa over the same held-out subjects in each setting, then calculate the ratios:

```python
# kappa_f, kappa_t, kappa_r: mean kappa for Frozen, Fine-tuned, Scratch
if kappa_t <= 0:
    raise ValueError("PRR and ACS require positive fine-tuned kappa.")

prr = kappa_f / kappa_t
acs = kappa_r / kappa_t
pvs = None if prr < 0 else (
    0.0 if acs >= 1 else 2 * prr * (1 - acs) / (prr + 1 - acs)
)
```

PVS is defined here for nonnegative PRR. Ratios of dataset-level means are not the same as means of subject-level ratios. Interpret these scores alongside absolute downstream performance: PRR measures relative readiness, ACS provides a same-architecture Scratch reference, and PVS summarizes joint positive pretraining value.

## Acknowledgments

The model implementations build on the respective EEG foundation model methods. Please acknowledge and cite the original model papers and datasets when using this code.
