# Reports and submission evidence

Updated 12 September 2026 against the [assignment brief](../CSE425_Project_GNN_BERT_Music_Context.pdf). Pages 8-9 require a 6-10 page NeurIPS/IEEE/ICML report plus source, graph samples, results and a demo.

The existing [CSE715 report](../CSE715_Project_Report.pdf) is seven pages and **preserved unchanged**. It covers frozen-fusion results but predates the completed joint run; the latest findings are below. This index does not certify complete/current coverage in that PDF.

## Measured reports

| Report | Experiment |
|---|---|
| [Task 1](task1_results.md) | Independent 30-label text classification; 829 test IDs; proxy history separately labeled |
| [Task 2](task2_results.md) | Real-audio GraphSAGE/CNN/MLP; 15 runs; 606 test clips |
| [Task 3 joint](task3_joint_results.md) | Latest five-mode live-encoder comparison; F1, mAP and trapezoidal PR-AUC |
| [Task 3 frozen](task3_results.md) | Separate three-seed head comparison over fixed encoders |
| [Task 4](task4_results.md) | Frozen contrastive projections; bidirectional retrieval and controls |
| [Task 4 extensions](task4_extensions.md) | CLAP, zero-shot caption tags and ten top-three query examples |

Joint gated fusion achieves test Macro-F1 **0.4073**, versus **0.3727** for joint BERT-only and **0.2700** for joint GNN-only. Validation selects **BERT-only overall and gated among fusion modes**. Gated Micro-F1 is lower than BERT-only; no significance is claimed. Do not pool joint and frozen seeds.

## Coverage and remaining work

- [Submission audit](submission_readiness_audit.md): PDF pages, evidence and unresolved requirements.
- [Task 3 audit](task3_completion_audit.md): joint/frozen completion with dataset qualifications.
- [Task 4 audit](task4_completion_audit.md): measured retrieval; human evaluation pending.
- [Task 4 scope deviation](task4_scope_deviation.md): projection-only training versus Algorithm 4.
- [Five-listener guide](task4_listening_guide.md): prepared forms are not results.
- [Documentation index](../docs/README.md): setup, reproduction and local-artifact dependencies.

The [Task 2 initial audit](task2_initial_audit.md) is historical; its missing-audio statements do not describe the later real-data run. Older test counts retain their original dates and are not fresh validation of a relocated submission.

Tie every number to its saved run and cohort. Do not use the brief's illustrative scores as results, relabel mAP as trapezoidal PR-AUC, or claim missing ratings or approval as completed work.
