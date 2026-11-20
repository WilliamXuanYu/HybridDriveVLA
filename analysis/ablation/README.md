# Archived NAVSIM ablations

[results.csv](results.csv) collects the `average` rows from prior PDM evaluations. It includes scene counts, PDM components, the original result filename relative to the experiment workspace, and the evidence used to link each run to a checkpoint. Scores use the 0–1 scale. Rows marked “Legacy config basename” link the historical filename to the curated filename using the logged backbone, stage, and rounded score.

Best-of-N selects the trajectory with the highest simulator PDM score using ground-truth metric caches. These rows measure an oracle upper bound for the sampled trajectories. Compare single-sample rows on the same split for the backbone and training-stage ablation.
