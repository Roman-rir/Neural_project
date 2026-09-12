# Current scope and assignment alignment

Updated 12 September 2026 against the [assignment brief](../CSE425_Project_GNN_BERT_Music_Context.pdf). **Measured experiments are complete for the documented MusicCaps scope; full assignment compliance is not established.**

## Dataset decision

MusicCaps supplies aligned ten-second audio references, captions and independent AudioSet positive labels. The text run uses all 5,521 metadata rows; paired experiments use 3,964 validated clips and the same 30 training-selected labels.

| Cohort | Train | Validation | Test |
|---|---:|---:|---:|
| Full caption cohort (Task 1) | 3,864 | 828 | 829 |
| Available paired cohort (Tasks 2-4) | 2,775 | 583 | 606 |

The custom seed-42 video-ID split is inherited by the paired subset. No artist-disjoint or official FMA/MagnaTagATune split is claimed. Availability can bias the retained cohort. Missing positive annotations are operationally zero, not confirmed negatives.

The brief lists MusicCaps and recommends aligned MusicCaps audio for advanced work (pp. 2-3), but **Task 2 names GTZAN/FMA-small and Task 3 names FMA-medium/MagnaTagATune (p. 4)**. Our runs do not establish results on those datasets. Instructor acceptance of substitution is not documented. Training and the original viability pilot are not approval evidence.

## Task boundaries

| Task | Completed implementation | Difference or limitation |
|---|---|---|
| 1 | Fine-tuned DistilBERT, independent labels, three controls, curves and five predictions | 30-label independent AudioSet task; permitted caption-derived proxy retained separately |
| 2 | GraphSAGE, mel-CNN, pooled MLP and temporal/similarity/random ablations; three seeds | MusicCaps instead of named dataset; one-second nodes, top-2 similarity neighbors and mean/max readout are declared choices |
| 3 | Five joint modes with partial BERT/live GNN updates; separate frozen ablation; F1, mAP, trapezoidal PR-AUC, t-SNE and cases | One joint seed; dataset substitution; only 12 Angry music mood annotations. Cross-attention uses a residual graph connection rather than the displayed concatenated readout |
| 4 | Frozen-backbone InfoNCE projections, bidirectional retrieval, ten queries, caption tags and CLAP | Algorithm 4 updates encoders; our retrieval loss updates projections only. Five human responses are absent |

`joint_run2` supplies the live/partial-encoder update aspect of Algorithm 3. It does not make the separate Task 4 retrieval experiment jointly fine-tuned. Validation selects BERT-only overall and gated among fusion models; higher gated test Macro-F1 does not change that selection.

No DEAM regression, chord-recognition experiment, broad mood-discrimination result or statistical significance is claimed. The [submission audit](../report/submission_readiness_audit.md) maps deliverables to PDF pages.

## Leakage and metric policy

Caption text alone enters DistilBERT. Aspect lists, label fields and IDs are excluded from its input. Targets come from `audioset_positive_labels`; vocabulary and normalization use training data only. Checkpoints and thresholds use validation only.

The historical lexical proxy derives targets from caption words. Its near-perfect scores are not comparable to independent-label results. Text pretraining used 1,089 more training-only captions than graph pretraining. Joint fine-tuning itself uses the same paired cohort across modes.

mAP averages per-label average precision; trapezoidal PR-AUC is separately computed in the joint report. Do not rename historical AP fields as trapezoidal area. Exact-ID Recall@K is not listener-rated relevance. Captions, ranks and software-test fixtures cannot supply human ratings.

## Remaining submission decisions

1. Record acceptance of MusicCaps for Tasks 2/3 or supply the named-dataset experiments.
2. Resolve projection-only Task 4 scope and collect five real listener responses if claiming full advanced-task credit.
3. Package at least 20 loadable graphs with schema, labels and preprocessing information; the PNG gallery alone is insufficient.
4. Assemble and verify the source/results/demo package. The existing PDF remains unchanged; current Markdown reports qualify its older frozen-fusion conclusions.

## Historical viability audit (29 August 2026)

The [metadata audit](../results/data_audit.json) found 5,521 unique ten-second windows, no missing required fields and no exact duplicate captions or video IDs. A deterministic metadata-only 200-ID pilot found 187 available clips and 13 unavailable (93.5%); no media was downloaded. A larger probe encountered throttling and was stopped. Those failures were not treated as permanent missing audio.

The pilot supported the original technical go-ahead, conditional on dataset acceptance. Its availability count is superseded by the validated 3,964-clip cohort. The initial audit found aspect phrases in 97.57% of captions, motivating independent targets rather than headline proxy scores.

## Attribution and local data

Attribution and source links are in the [Task 1 guide](task1/README.md#scope-and-attribution). MusicCaps metadata and referenced audio have separate reuse considerations. Keep raw audio and credentials out of Git; preserve permitted local audio needed by the demo. Acquisition tooling is documented in the [audio guide](task2/downloading_audio.md).
