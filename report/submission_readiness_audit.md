# Submission readiness audit against the assignment

**Updated: 12 September 2026.** Source: all nine pages of the [assignment brief](../CSE425_Project_GNN_BERT_Music_Context.pdf). Deadline: **2 October 2026**.

**Verdict: substantial measured work is available, but full assignment compliance and a portable final submission are not yet established.** This reviews local documentation and saved artifacts, not a remote repository, instructor approval, new training or a clean installation of a final archive.

Earlier findings that the report PDF, ten caption queries and zero-shot tag comparison were missing are superseded. The existing [seven-page report](../CSE715_Project_Report.pdf) is present and remains unchanged. Its abstract/discussion/conclusion describe the frozen experiment; the [joint report](task3_joint_results.md) records the latest results.

## Requirement-by-requirement assessment

| Assignment requirement | Current evidence | Status / qualification |
|---|---|---|
| Audio plus text/tag data, pp. 2-3 | MusicCaps captions, aligned audio and independent AudioSet labels | Both modalities supplied; task-specific substitution remains unresolved |
| Resampling, mel/chroma and normalization, pp. 2-3 | 22.05 kHz mono, per-clip peak normalization, 128-bin mel, chroma/MFCC | Implemented; feature normalization fitted on training nodes only |
| Segment graphs and tokenization, p. 3 | Ten one-second nodes, 311 features, temporal/top-2 similarity edges; max text length 128 | Declared choices; top-k edges differ from the displayed cosine-threshold graph |
| Split discipline, p. 3 | Seed-42 video-ID split inherited by paired cohort | No ID overlap; custom split, not artist-disjoint or official FMA/MagnaTagATune split |
| Task 1 fine-tuning, F1 curves and five predictions, p. 3 | [Report](task1_results.md), current `f1_curve.png` and `example_predictions.json` | Independent 30-label targets; permitted lexical proxy retained separately. Attention visualization optional |
| Task 2 GraphSAGE/GAT plus mel-CNN, p. 4 | [Results](task2_results.md): five configurations, three seeds | Implemented GraphSAGE; mean/max pooling extends displayed mean-only readout |
| Task 2 GTZAN/FMA-small, p. 4 | Current results use 3,964 MusicCaps clips | Named-dataset result not supplied; substitution acceptance undocumented |
| Task 3 live fusion and ablations, p. 4; Algorithm 3, p. 7 | `joint_run2`: BERT, GNN, concat, gated and cross-attention | All five trained/evaluated; final BERT block, GNN and active heads update; one joint seed |
| Task 3 FMA-medium/MagnaTagATune, p. 4 | Current results use MusicCaps | Named-dataset result not supplied; substitution acceptance undocumented |
| Task 3 F1 and mean AUC-PR, pp. 4-5 | [Joint comparison](../results/task3/joint_run2/comparison.csv) | F1, mAP and separately computed trapezoidal mean PR-AUC present |
| Genre/mood t-SNE, p. 4 | [Joint plots](../results/task3/joint_run2/semantic_plots/README.md) | Generated on 583 validation clips; 95 genre/style positives and only 12 Angry music annotations |
| Three graph/caption cases, p. 4 | [Frozen graph-path/sensitivity cases](../results/task3/available_run1/case_studies/cases.md); [joint prediction cases](../results/task3/joint_run2/analysis/cases.md) | Three fixed cases each; frozen visualization evidence is not joint-model attribution |
| Task 4 shared embedding and contrastive loss, p. 5 | [Retrieval report](task4_results.md), three projection seeds | Measured over frozen classification-trained backbones |
| Task 4 encoder updates, Algorithm 4, p. 7 | Only projections receive retrieval gradients | Scope deviation; acceptance or completed contrastive encoder fine-tuning not documented |
| Bidirectional Recall@1/5/10, p. 5 | 606-query/gallery exact-pair comparison and controls | Measured; CLAP is an additional external comparator |
| Ten caption queries to top-three clips, p. 5 | [Ten fixed queries](../results/task4/listening_study/ten_caption_queries.md) | Present for graph seed 42 and CLAP |
| Zero-shot caption tags versus Task 3, p. 5 | [Extensions](task4_extensions.md) | Fixed CLAP text prompt baseline versus frozen supervised gated fusion, not the later joint model |
| Five listeners, scale 1-5, p. 6 | [Protocol/forms](task4_listening_guide.md) | Prepared; no returned ratings or human summary found |
| At least two baselines, pp. 6-7 | Prior, keyword, TF-IDF, pooled MLP, CNN, unimodal controls | Present; retain matched-cohort and training-access qualifications |
| Full source repository or ZIP, p. 8 | Current source and guides | Final delivered commit/archive and clean-copy checks not established |
| At least 20 loadable graphs, p. 8 | 3,964 local `.pt` samples; 20-example PNG gallery | No tracked `.pt` files; portable graph bundle needs schema/labels and inclusion in submission |
| Evaluation tables and plots, p. 8 | Reports, CSVs, curves, t-SNE, retrieval examples | Present locally; include matching artifacts in delivered package |
| 6-10 page report in named paper template, pp. 8-9 | Existing seven-page `CSE715_Project_Report.pdf` | Page-count range met; preserved PDF predates joint results. Current content/template compliance is not certified by page count |
| End-to-end demo, p. 8 | [Notebook](../notebooks/demo_context.ipynb) and saved zero-error execution | Executed locally; matching WAV, checkpoint, tokenizer and preprocessing dependencies required |

Task 4 is optional/bonus in the p. 7 rubric. Human ratings remain a requirement of that advanced task. DEAM regression is conditional on available targets; Task 1 attention visualization, graph coherence and PCA+MLP are optional. The brief's Table 3 numbers are illustrative, not required scores.

## Current measured outcomes

| Experiment | Test population | Result and interpretation |
|---|---|---|
| Task 1 DistilBERT | 829 captions | Macro-F1 0.3840, Micro-F1 0.5902, mAP 0.3686; TF-IDF has higher Micro-F1 |
| Task 2 temporal GraphSAGE | 606 clips, three seeds | Mean Macro-F1 0.2648; pooled MLP 0.2610, CNN 0.2215; small graph-over-pooling gain |
| Task 3 frozen | 606 clips, three head seeds | BERT mean Macro-F1 0.3850; gated 0.3705 |
| Task 3 joint | 606 clips, seed 42 | Gated Macro-F1 0.4073 and PR-AUC 0.3986; BERT 0.3727 and 0.3751 |
| Task 4 projections | 606 pairs, three seeds | Text-to-graph R@10 17.77%; reverse 18.65% |
| CLAP zero-shot | Same 606 pairs | Caption-to-audio R@10 45.38%; reverse 40.92%; external pretraining differs |

Validation selects **BERT-only overall and gated among fusion models** in both Task 3 suites. The joint gated test gain does not override selection. Gated Micro-F1 is lower than BERT-only. Joint and frozen seeds must not be pooled; no significance is claimed. Text pretraining used 1,089 more training-only captions than graph pretraining.

## Saved verification reviewed

- [Task 1](../results/task1/audioset_cpu_20260907/verification.json): independent labels, 829 test IDs and calibrated model/control metrics.
- [Task 2](../results/task2/available_run1/verification.json): 15 real-data runs, identical IDs/targets/labels, thresholds, hashes and aggregate statistics.
- [Joint verification](../results/task3/joint_run2/verification.json): valid, five runs, evaluated test, active encoder updates and unchanged frozen layers.
- [Notebook verification](../results/task3/joint_run2/notebook_verification.json): joint review (five code cells) and raw-audio demo (three), zero errors.
- [Retrieval](../results/task4/available_run1/verification.json) and [extensions](../results/task4/extension_verification.json): saved retrieval evidence, aligned CLAP/tag metrics, ten queries and five forms. Forms are not ratings.

These are existing records inspected for this documentation update. No retraining, new test inference or fresh full regression run was performed. Older 77/86-test references are historical snapshots. No current submission ZIP was found at the project root; the Task 1 source snapshot covers only one run.

## Remaining submission actions

1. Document acceptance of MusicCaps for Tasks 2/3, or supply the named-dataset experiments.
2. If claiming the entire advanced task, collect ratings from at least five real people and resolve the frozen-encoder deviation.
3. Package at least 20 loadable graphs with ordered labels, node-feature schema, edge conventions, sample IDs and normalization information. Validate loading from the package, not just images.
4. Package source, requirements/configuration, tables/plots and demo dependencies; account for historical absolute paths. Execute the demo and relevant checks from that delivered copy.
5. Keep the existing report unchanged as requested. Its older conclusions should not imply the joint experiment remains unperformed; consult the current result reports when assessing completed work.

A documentation refresh does not collect ratings, grant approval, create portable graph files or certify an archive.
