# Three fixed validation cases

The first three validation IDs in processed-manifest order are fixed independently of predictions. Thresholds and the fusion mode were selected on validation. These are in-sample validation diagnostics, not independent estimates of generalization. False positives mean missing positive annotations, which may reflect incomplete labels. No attention weights are used as causal explanations.

## Case 1: -5xOcMJpTUk_70_80

A male guitarist plays the guitar and speaks about technique in this online video tutorial. The male voice is strong and commanding, along with guitar string twang sounds, clearly demonstrating the technique. The audio quality is mediocre.

Annotated labels: Music, Musical instrument, Guitar, Plucked string instrument, Speech, Acoustic guitar.

**bert** (threshold 0.20): true positive: Music, Musical instrument, Guitar, Plucked string instrument, Speech, Acoustic guitar; false positive: none; false negative: none.

**gnn** (threshold 0.10): true positive: Music, Musical instrument, Guitar, Plucked string instrument, Speech, Acoustic guitar; false positive: Effects unit, Electric guitar, Distortion; false negative: none.

**gated** (threshold 0.15): true positive: Music, Musical instrument, Guitar, Plucked string instrument, Speech, Acoustic guitar; false positive: none; false negative: none.

Selected-fusion top probabilities:

| name | probability | target | predicted | status |
| --- | --- | --- | --- | --- |
| Music | 0.9949 | 1 | 1 | true_positive |
| Plucked string instrument | 0.9896 | 1 | 1 | true_positive |
| Guitar | 0.9742 | 1 | 1 | true_positive |
| Musical instrument | 0.9646 | 1 | 1 | true_positive |
| Acoustic guitar | 0.5769 | 1 | 1 | true_positive |
| Speech | 0.5616 | 1 | 1 | true_positive |
| Electric guitar | 0.1430 | 0 | 0 | true_negative |
| Distortion | 0.1249 | 0 | 0 | true_negative |

## Case 2: -8C-gydUbR8_30_40

A children’s choir sings this devotional melody. The song is medium tempo with a steady bass line, drumming rhythm and clapping percussion. The song is black gospel choral music played in front of a live congregation. The audio quality is very poor.

Annotated labels: Music.

**bert** (threshold 0.20): true positive: Music; false positive: none; false negative: none.

**gnn** (threshold 0.10): true positive: Music; false positive: none; false negative: none.

**gated** (threshold 0.15): true positive: Music; false positive: none; false negative: none.

Selected-fusion top probabilities:

| name | probability | target | predicted | status |
| --- | --- | --- | --- | --- |
| Music | 0.9797 | 1 | 1 | true_positive |
| Singing | 0.0422 | 0 | 0 | true_negative |
| Speech | 0.0222 | 0 | 0 | true_negative |
| Musical instrument | 0.0176 | 0 | 0 | true_negative |
| Pop music | 0.0057 | 0 | 0 | true_negative |
| Music of Africa | 0.0051 | 0 | 0 | true_negative |
| Rock music | 0.0050 | 0 | 0 | true_negative |
| Song | 0.0047 | 0 | 0 | true_negative |

## Case 3: -Bu7YaslRW0_30_40

A synth pad is playing a drone sound in the lower mid range. Cymbals are creating atmosphere while a flute/string/brass sound is playing a melody. The whole recording is full of reverb. This song may be playing in a forest documentary.

Annotated labels: Music.

**bert** (threshold 0.20): true positive: Music; false positive: none; false negative: none.

**gnn** (threshold 0.10): true positive: Music; false positive: New-age music; false negative: none.

**gated** (threshold 0.15): true positive: Music; false positive: New-age music; false negative: none.

Selected-fusion top probabilities:

| name | probability | target | predicted | status |
| --- | --- | --- | --- | --- |
| Music | 0.9919 | 1 | 1 | true_positive |
| New-age music | 0.2400 | 0 | 1 | false_positive |
| Traditional music | 0.0708 | 0 | 0 | true_negative |
| Musical instrument | 0.0321 | 0 | 0 | true_negative |
| Electronic music | 0.0165 | 0 | 0 | true_negative |
| Effects unit | 0.0125 | 0 | 0 | true_negative |
| Independent music | 0.0102 | 0 | 0 | true_negative |
| Angry music | 0.0084 | 0 | 0 | true_negative |
