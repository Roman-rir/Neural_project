# GNN-BERT Music Context Understanding

Neural Networks course project (CSE425 / EEE474 / CSE715) at BRAC University. This project combines **DistilBERT caption representations** with **GraphSAGE audio-segment graphs** for multi-label music-context prediction and graph-text retrieval.

**Current status - 12 September 2026:** Tasks 1-3 have measured results on MusicCaps, including the completed five-model joint-fusion experiment. Task 4 includes frozen-encoder contrastive retrieval, CLAP comparisons and ten caption queries. Human listening evaluation and acceptance of the documented scope variations remain pending.

## Start here

| Resource | Purpose |
|---|---|
| [Existing report PDF](CSE715_Project_Report.pdf) | Preserved report; its discussion predates the completed joint-fusion results |
| [Assignment brief](CSE425_Project_GNN_BERT_Music_Context.pdf) | Original requirements and 2 October 2026 deadline |
| [Latest joint-fusion results](report/task3_joint_results.md) | Current Task 3 comparison, methods and limitations |
| [End-to-end demo](notebooks/demo_context.ipynb) | Real local audio and caption to joint-fusion predictions |
| [Task 4 extensions](report/task4_extensions.md) | CLAP retrieval, zero-shot caption tags and listening-study status |
| [Submission audit](report/submission_readiness_audit.md) | Requirement checklist; read with the newer joint results and Task 4 extensions |

The existing PDF is unchanged. The task reports and saved result artifacts below record the latest completed work. Earlier audit statements that the report, ten caption queries or zero-shot comparisons are missing have been superseded by the current files.

## Implemented pipeline

```text
MusicCaps audio interval                   MusicCaps caption
          |                                        |
Mono 22.05 kHz; peak normalization         DistilBERT tokenization
          |                                  (max length 128)
Ten 1-second segments                              |
311 acoustic features per node             Contextual text embeddings
          |                                        |
Temporal / similarity / random graph               |
          |                                        |
Two-layer GraphSAGE                                |
Mean + max graph pooling                           |
          +-------------------+--------------------+
                              |
              Concat / gated / cross-attention fusion
                              |
                  30 AudioSet label predictions
```

Task 3 joint training updates GraphSAGE, the active prediction/fusion head and the final DistilBERT block; lower transformer blocks remain frozen. Task 4 instead learns two 64-dimensional contrastive projection heads over frozen, previously supervised text and graph encoders.

The measured work uses segment graphs and AudioSet tags. Chord recognition and DEAM valence/arousal regression are not reported experiments.

## Data and evaluation protocol

[MusicCaps](https://huggingface.co/datasets/google/MusicCaps) provides captions and clip references. Targets use its independently supplied `audioset_positive_labels`, rather than labels extracted from caption keywords.

| Cohort | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Task 1: all metadata captions | 3,864 | 828 | 829 | 5,521 |
| Tasks 2-4: available validated audio/caption pairs | 2,775 | 583 | 606 | 3,964 |

- Split by video ID with seed 42 before label selection. The paired cohort inherits those assignments without re-splitting.
- Select 30 nonconstant AudioSet labels using training counts only, requiring at least 20 training positives per label.
- Fit feature normalization on training data. Select checkpoints, thresholds and models using validation data only.
- Keep the held-out test set separate from model selection. Missing positive annotations are treated as zero for evaluation, but do not establish the absence of an event.
- Macro-F1 averages label F1; Micro-F1 pools label decisions. **mAP is mean average precision, not trapezoidal PR-AUC.** The joint Task 3 results report both separately.

Task 1 uses 829 test clips, while Tasks 2-4 use 606. Do not interpret their headline scores as comparisons on identical test populations. Video-ID separation does not establish artist-disjoint evaluation. The text encoder also saw 1,089 additional training-only captions before the paired experiments.

## Completed results

### Task 1: caption-only classification

Two epochs of full DistilBERT fine-tuning, evaluated on 829 held-out captions with validation-calibrated thresholds:

| Model | Macro-F1 | Micro-F1 | mAP |
|---|---:|---:|---:|
| Label prior | 0.0528 | 0.3530 | 0.0591 |
| Ontology keyword | 0.1568 | 0.2257 | 0.1158 |
| TF-IDF + logistic regression | 0.3393 | **0.6530** | 0.3603 |
| DistilBERT | **0.3840** | 0.5902 | **0.3686** |

Run: `results/task1/audioset_cpu_20260907/`. See the [measured report](report/task1_results.md), [notebook](notebooks/task1_bert_baseline.ipynb) and [reproduction guide](docs/task1/README.md).

The historical [lexical-proxy experiment](docs/task1/legacy_proxy.md) is preserved separately. Its near-perfect scores do not measure the current independent-label task.

### Task 2: audio-only baselines and graph ablations

Five configurations across seeds 42, 43 and 44; all 15 runs use the same 606 test clips. Values are mean +/- sample standard deviation.

| Model | Macro-F1 | Micro-F1 | mAP |
|---|---:|---:|---:|
| Pooled-feature MLP | 0.2610 +/- 0.0097 | 0.5595 +/- 0.0349 | 0.2747 +/- 0.0086 |
| Mel-spectrogram CNN | 0.2215 +/- 0.0104 | 0.3820 +/- 0.0363 | 0.2320 +/- 0.0067 |
| Temporal GraphSAGE | 0.2648 +/- 0.0114 | 0.5729 +/- 0.0207 | 0.2762 +/- 0.0006 |
| Temporal + similarity GraphSAGE | 0.2540 +/- 0.0145 | 0.5651 +/- 0.0157 | 0.2640 +/- 0.0063 |
| Temporal + random GraphSAGE | 0.2475 +/- 0.0140 | 0.5498 +/- 0.0118 | 0.2560 +/- 0.0117 |

Temporal GraphSAGE wins mean validation Macro-F1. Its test gain over pooled features is small; additional similarity edges do not improve the measured result. These standard deviations describe seed variation, not statistical significance.

Run: `results/task2/available_run1/`. See the [measured report](report/task2_results.md), [20-clip graph gallery](results/task2/available_run1/graph_examples/index.html) and [reproduction guide](docs/task2/README.md). The [Task 2 notebook](notebooks/task2_gnn_cnn.ipynb) defaults to a **synthetic pipeline demonstration**; use the real run and report for experimental claims.

### Task 3: joint GNN-BERT fusion

The completed `results/task3/joint_run2/` experiment trains all five modes for three epochs, seed 42, on the paired cohort. The [saved verification](results/task3/joint_run2/verification.json) confirms five checked runs and completed held-out evaluation.

| Model | Validation Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test mAP | Test mean PR-AUC |
|---|---:|---:|---:|---:|---:|
| BERT-only | **0.4602** | 0.3727 | 0.6739 | 0.3922 | 0.3751 |
| GNN-only | 0.2861 | 0.2700 | 0.5398 | 0.2706 | 0.2562 |
| Concatenation | 0.4326 | 0.3877 | **0.6810** | 0.4056 | 0.3884 |
| Gated fusion | 0.4496 | **0.4073** | 0.6563 | **0.4148** | **0.3986** |
| Cross-attention | 0.4104 | 0.3867 | 0.6503 | 0.3943 | 0.3777 |

**Validation selects BERT-only overall and gated fusion among fusion models.** Gated fusion exceeds BERT-only test Macro-F1 by 0.0347, but has lower Micro-F1. This is one joint-training seed; the test result does not change the validation selection or establish significance.

The separate [frozen-encoder experiment](report/task3_results.md) used three head seeds: BERT-only achieved mean test Macro-F1 0.3850 and gated fusion 0.3705. Do not combine those runs with the joint experiment as repeated joint-training seeds.

Completed artifacts include training curves, per-label errors, three fixed validation cases, genre/mood t-SNE, the [joint review notebook](notebooks/task3_joint.ipynb) and [raw-audio demo](notebooks/demo_context.ipynb). Both inference notebooks have saved executions with zero errors. The mood view has only 12 Angry music annotations and cannot establish broad mood separation.

See the [joint results](report/task3_joint_results.md), [comparison CSV](results/task3/joint_run2/comparison.csv) and [Task 3 guide](docs/task3/README.md).

### Task 4: retrieval and zero-shot comparisons

Exact-pair retrieval uses the same 606 held-out query/gallery pairs. Graph values are means over three projection seeds; CLAP uses one fixed pretrained checkpoint without project-specific parameter updates. Recall values below are **percentages**.

| Direction | System | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---|---:|---:|---:|
| Caption to graph | Trained graph projections | 2.59 | 10.62 | 17.77 |
| Graph to caption | Trained graph projections | 2.81 | 11.11 | 18.65 |
| Caption to audio | CLAP zero-shot | 13.70 | 32.67 | 45.38 |
| Audio to caption | CLAP zero-shot | 11.39 | 31.52 | 40.92 |

Contrastive projections improve over untrained and random controls. CLAP performs better, but changes both the backbone and external training data; this is not a controlled graph-structure ablation. Possible CLAP pretraining overlap has not been ruled out.

The separate CLAP caption prompt-similarity tag baseline gives Macro-F1 **0.1090**, Micro-F1 **0.1530** and mAP **0.1025**. Its saved comparator is Task 3's **frozen** supervised gated fusion (0.3705 / 0.6466 / 0.3995), not the later joint model. This text-only zero-shot baseline and audio-plus-text supervised model have different information and training access.

[Ten fixed caption queries](results/task4/listening_study/ten_caption_queries.md) include top-three graph and CLAP matches. Five blinded listener forms are prepared, with 60 judgments per listener, but **no human ratings have been collected**. Exact-ID retrieval is not a measure of listener-rated semantic relevance.

See the [retrieval report](report/task4_results.md), [extension results](report/task4_extensions.md), [review notebook](notebooks/task4_retrieval.ipynb), [run guide](docs/task4/README.md) and [listening instructions](report/task4_listening_guide.md).

## Setup and review

Run commands from the repository root. For a new environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the project's existing environment if it is already configured. The completed runs used CPU execution with six PyTorch threads; they do not require an OpenAI API key.

### Local artifacts required

A source checkout alone does not contain every experiment dependency. `.gitignore` excludes raw audio, processed caches and `*.pt` checkpoints. Restore the matching local artifacts, or follow the task guides to reproduce them:

- Task 1 prepared label CSV, split metadata and trained DistilBERT checkpoint.
- Task 2 source/processed manifests, training normalization, graph samples and temporal GraphSAGE checkpoint.
- Task 3 frozen features/heads and completed `joint_run2` checkpoints/tokenizer.
- The demo's local WAV clip referenced by its manifest; the demo does not download audio.

Preserve matching manifests, labels, splits, normalization and checkpoints. Verifiers use artifact hashes and sample identities; do not mix files from different runs.

### Verify saved experiments

These commands recompute or validate saved evidence without retraining; some refresh verification files. They require the original local artifacts.

```powershell
.\.venv\Scripts\python.exe -m src.task1.evaluate --run-dir results/task1/audioset_cpu_20260907
.\.venv\Scripts\python.exe -m src.task3.experiment --verify-run results/task3/available_run1
.\.venv\Scripts\python.exe -m src.task3.pipeline verify --config configs/task3.yaml
.\.venv\Scripts\python.exe -m src.task4.experiment verify --output-dir results/task4/available_run1
```

Run the software tests separately:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Open `notebooks/demo_context.ipynb` in VS Code or Jupyter with the `.venv` kernel and run its cells in order. It rebuilds features from a real validation audio clip, restores the selected joint fusion model and checks predictions against saved evidence. No training is needed when its dependencies are present.

### Train another joint experiment

`configs/task3.yaml` is the dedicated configuration. It defaults to the completed `joint_run2`, so use an unused output directory consistently for a new run:

```powershell
.\.venv\Scripts\python.exe -m src.task3.pipeline check --config configs/task3.yaml --output-dir results/task3/joint_run3
.\.venv\Scripts\python.exe -m src.task3.pipeline train --config configs/task3.yaml --output-dir results/task3/joint_run3
.\.venv\Scripts\python.exe -m src.task3.pipeline evaluate --config configs/task3.yaml --output-dir results/task3/joint_run3
.\.venv\Scripts\python.exe -m src.task3.pipeline verify --config configs/task3.yaml --output-dir results/task3/joint_run3
.\.venv\Scripts\python.exe -m src.task3.pipeline plot --config configs/task3.yaml --output-dir results/task3/joint_run3
.\.venv\Scripts\python.exe -m src.task3.pipeline analyze --config configs/task3.yaml --output-dir results/task3/joint_run3
```

Choose another unused name if `joint_run3` already exists. Training does not resume partial runs; `joint_run1` remains interrupted. `evaluate` is a separate final test step and refuses repeated test evaluation; use `verify` afterwards. The root `config.yaml` and `config1` are legacy generic-runner settings, not the current joint configuration.

For data acquisition and Tasks 1, 2 and 4 reproduction, use the dedicated guides linked above. New training and preprocessing require fresh destinations to preserve measured results.

## Repository map

```text
configs/task3.yaml       Current joint experiment configuration
src/task1/              Caption preparation, training, controls and evaluation
src/task2/              Audio features, graph construction, CNN/MLP/GNN suite
src/task3/              Frozen and joint fusion, inference and analysis
src/task4/              Contrastive retrieval, CLAP, zero-shot tags and listening
notebooks/              Review notebooks and end-to-end demo
results/task1/          Current independent-label run plus historical artifacts
results/task2/          Real-audio runs and separate synthetic verification
results/task3/          Frozen comparison and joint_run2 results
results/task4/          Retrieval runs, extensions and listening forms
data/                   Local metadata, audio, manifests and processed caches
docs/                   Task-specific setup and reproduction guides
report/                 Measured task reports and submission audits
tests/                  Data, model, evaluation and artifact-integrity checks
```

## Submission status and limitations

The repository documents completed experiments; it does not establish full compliance with every assignment requirement.

| Requirement | Current status |
|---|---|
| Text, audio and fusion experiments | Measured on MusicCaps, with saved comparisons and diagnostics |
| Required Task 2/3 datasets | MusicCaps substitutes for the specified GTZAN/FMA/MagnaTagATune runs; instructor acceptance is not documented in [scope.md](docs/scope.md) |
| Final report | [Existing seven-page PDF](CSE715_Project_Report.pdf) is preserved; its frozen-fusion conclusions predate `joint_run2` |
| At least 20 portable graph files | A 20-clip visual gallery exists; package loadable `.pt`/`.json` samples with feature/edge schema and label mapping, not only PNGs |
| Demo notebook | Executed locally; a submitted copy still needs matching checkpoint, manifest, normalization and audio dependencies |
| Task 4 human evaluation | Five real listener responses remain pending |
| Task 4 encoder updates | Measured training updates projections only; the original algorithm updates encoders. See the [scope deviation](report/task4_scope_deviation.md) |


Other limitations include available-audio selection bias, incomplete labels, coarse one-second segment features, unequal historical encoder-training cohorts and no artist-disjoint evaluation. No emotion-regression result or statistical significance is claimed.

## Data handling

Keep raw dataset audio, credentials and generated caches out of Git. Follow source dataset licensing and attribution requirements; MusicCaps metadata and referenced audio have separate reuse considerations. The [dataset scope and viability notes](docs/scope.md) document the original audit and outstanding acceptance requirement. Include only experimental results backed by saved outputs, and distinguish synthetic checks and lexical proxies from measured real-audio experiments.
