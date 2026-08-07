# Online pipeline runtime audit

All values are milliseconds. Intervals are 95% normal intervals; real-pipeline logs are summarized with equal weight per sequence.

| Module | Runtime | Unit | Evidence |
|---|---:|---|---|
| SAM2-tiny segmentation | 14.5 $\pm$ 0.2 | per frame | 120 synchronized CUDA frames |
| End-to-end perception/tracking | 242.9 $\pm$ 47.9 | per frame | 8 real sequences, 2041 calls |
| Point-cloud fusion | 1649.6 $\pm$ 194.0 | per update | 8 real sequences, 36 calls |
| GP/representation update | 261.3 $\pm$ 24.2 | per planning step | 120 paired scenes |
| Candidate scoring | 0.072 $\pm$ 0.001 | per planning step | 120 paired scenes |
| NBV-to-action mapping | 2.420 $\pm$ 2.684 | per update | 8 real sequences, 36 calls |

The end-to-end perception/tracking row already includes segmentation and is therefore not additive with the standalone segmentation row. Fusion is invoked at planning boundaries. Model loading, visualization, disk export, and the 6 s manipulation primitive are excluded.
