# EchoVein-DIP validation (simulated ground truth)

Setup: 2 seeds x 8 scenarios per configuration, 6 h each, one analysis / 20 min, simulated 2 m x 1.5 m wall, perspective camera 640x480. Positive: ['leak', 'splash']. Negative: ['shadow', 'light_switch', 'person', 'camera_nudge', 'dry_drift', 'condensation'].
These are simulator results: they show the contribution of each module, not real-world accuracy. Repeat with a phone rig.

| Configuration | Detected | Delay h (mean) | Source err cm (median) | IoU leak | Leaks escalated | Neg. scenarios alarmed | False alarms / sim-day | Event class OK |
|---|---|---|---|---|---|---|---|---|
| full (ML pixel + ML events) | 100% | 0.55 | 1.8 | 0.98 | 100% | 0% | 0.00 | 100% |
| full (weighted fusion, rule events) | 75% | 0.70 | 0.6 | 0.74 | 100% | 8% | 0.33 | 100% |
| - illumination library | 100% | 0.55 | 1.7 | 0.98 | 100% | 0% | 0.00 | 100% |
| - illumination field normalisation | 100% | 0.55 | 2.2 | 0.92 | 100% | 8% | 0.33 | 100% |
| - multi-cue fusion (darkening only) | 100% | 0.63 | 1.7 | 0.83 | 100% | 0% | 0.00 | 100% |
| - temporal logic (persist=1, no tracking memory) | 100% | 0.22 | 2.2 | 0.98 | 100% | 17% | 0.67 | 0% |
| - occlusion handling | 100% | 0.55 | 1.8 | 0.98 | 100% | 0% | 0.00 | 100% |
| - camera self-recalibration | 100% | 0.55 | 1.8 | 0.98 | 100% | 8% | 0.33 | 100% |
| - event classifier (no cross-wall/drying logic) | 100% | 0.55 | 1.8 | 0.98 | 100% | 17% | 0.67 | 100% |

## Per-scenario outcome (full ML configuration)

| scenario | seed | alert | escalated | delay h | source err cm | IoU |
|---|---|---|---|---|---|---|
| leak | 40 | yes | yes | 0.5 | 1.6 | 0.99 |
| leak | 41 | yes | yes | 0.77 | 1.9 | 0.97 |
| splash | 40 | yes | no | 0.5 | 0.6 |  |
| splash | 41 | yes | no | 0.43 | 3.4 |  |
| shadow | 40 | no | no |  |  |  |
| shadow | 41 | no | no |  |  |  |
| light_switch | 40 | no | no |  |  |  |
| light_switch | 41 | no | no |  |  |  |
| person | 40 | no | no |  |  |  |
| person | 41 | no | no |  |  |  |
| camera_nudge | 40 | no | no |  |  |  |
| camera_nudge | 41 | no | no |  |  |  |
| dry_drift | 40 | no | no |  |  |  |
| dry_drift | 41 | no | no |  |  |  |
| condensation | 40 | no | no |  |  |  |
| condensation | 41 | no | no |  |  |  |
