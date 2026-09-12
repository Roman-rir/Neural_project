# Documentation and assignment guide

Updated 12 September 2026 against all nine pages of the [assignment brief](../CSE425_Project_GNN_BERT_Music_Context.pdf). Deadline: **2 October 2026**. The brief lists CSE425 / EEE474 / CSE715.

Start with the [current scope](scope.md) and [requirement audit](../report/submission_readiness_audit.md). The [report index](../report/README.md) distinguishes measured results from historical records. The existing [seven-page PDF](../CSE715_Project_Report.pdf) is preserved unchanged; its conclusions predate the latest joint experiment.

## Task guides

| Assignment section | Implementation and reproduction | Current evidence |
|---|---|---|
| Task 1, p. 3 | [Caption classifier](task1/README.md) | DistilBERT and three controls; 829 test captions |
| Task 2, p. 4 | [Audio graphs and CNN](task2/README.md) | Five configurations, three seeds, 606 test clips; MusicCaps substitution |
| Task 3, p. 4 and Algorithm 3, p. 7 | [Joint and frozen fusion](task3/README.md) | Five-mode joint run, separate frozen ablation, t-SNE and cases |
| Task 4, pp. 5-7 | [Contrastive retrieval](task4/README.md) | Three projection seeds, CLAP, caption tags and ten queries; listening pending |
| Final submission, pp. 8-9 | [Submission audit](../report/submission_readiness_audit.md) | Source, portable graphs, tables/plots, report and executable demo checklist |

## Environment and review

Run commands from the project root (`F:\Neural_project` in this workspace). The [requirements file](../requirements.txt) pins installed direct dependencies; recorded training used Python 3.14.4. Use the configured environment or create a new one:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Examples using `python` assume this environment is active. Alternatively, replace `python` with `.\.venv\Scripts\python.exe`. Acquisition additionally needs FFmpeg/ffprobe and the JavaScript runtime described in the [download guide](task2/downloading_audio.md).

Review existing experiments without retraining:

```powershell
.\.venv\Scripts\python.exe -m src.task1.evaluate --run-dir results/task1/audioset_cpu_20260907
.\.venv\Scripts\python.exe -m src.task3.experiment --verify-run results/task3/available_run1
.\.venv\Scripts\python.exe -m src.task3.pipeline verify --config configs/task3.yaml
.\.venv\Scripts\python.exe -m src.task4.experiment verify --output-dir results/task4/available_run1
```

These commands require matching local data/checkpoints and may refresh verification files. The [Task 2 verification](../results/task2/available_run1/verification.json) records its 15-run evidence. Historical checks do not establish that a newly copied environment has been tested.

## Reproduction and portability

A source checkout excludes raw audio, processed caches and `*.pt` checkpoints. Preserve the prepared CSV, split/processed manifests, training normalization, vocabulary, encoder/head checkpoints and feature caches for each run. Historical configurations may reference the original drive. Use documented relocation support and matching artifacts rather than silently editing provenance records.

The [demo notebook](../notebooks/demo_context.ipynb) loads a real validation WAV, reconstructs features/edges and runs the selected joint fusion checkpoint. It needs the completed `joint_run2` tokenizer/checkpoints and referenced manifests/normalization. Saved outputs demonstrate prior execution; execute the notebook from the delivered copy to establish portability.

New training needs unused output directories, used consistently across training, evaluation and analysis. Do not train into `available_run1` or `joint_run2`. Do not select epochs, seeds, prompts or thresholds using test scores. Current joint training uses `configs/task3.yaml`; root `config.yaml` and `config1` are legacy settings.

## Interpretation

Task 1 uses 829 test captions; paired Tasks 2-4 use 606 clips. Frozen-head seeds and the single joint seed are separate experiments. mAP means average precision; joint results also report trapezoidal mean PR-AUC. Synthetic and lexical-proxy examples are pipeline checks, not general music-understanding evidence.

Task 4 is optional/bonus on p. 7. Claiming the full advanced task still requires its five-listener evaluation and resolution of encoder-update scope. Optional attention visualization, graph-coherence analysis and conditional DEAM regression are not missing core experiments.
