"""
1) ROC curve (TPR vs FPR)
What: parameterize threshold from 0→1, plot TPR(t) vs FPR(t), report AUC.
Based on: the raw posterior P(β|D) (or equivalently R) for every event, paired with the ground-truth label.
Tells you: the discriminative power of your likelihood model alone. Prior has no effect on ROC. If ROC is bad, the features or density estimate are bad. If ROC is good, any prior-induced misbehavior is a thresholding choice, not a model failure.

2) Precision-Recall curve
What: P vs R across thresholds; report average precision (AP).
Based on: same scores + labels as ROC.
Tells you: for imbalanced problems PR is more honest than ROC. If your β fraction is, say, 30%, ROC can look good while PR reveals the classifier gets crushed at high recall. Physics audiences tend to prefer efficiency/purity versions of this (same math, different axes).

3) Score distribution per class
What: two overlaid histograms of P(β|D) (or log R) — one for true β events, one for true bg.
Based on: same scores + labels.
Tells you: where the separation happens. A bimodal picture with β peaking near 1 and bg near 0 = good. Heavy overlap near 0.5 = weak features. Spikes at exactly 0 and 1 = you're in the degenerate sparse-bin regime (events whose bin contained only their own class).

Use these libraries at least:
pytest
wandb
matplotlib
...
"""