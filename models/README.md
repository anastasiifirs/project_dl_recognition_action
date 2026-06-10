# Model files

This repository includes model-related artifacts:

- `temperature.json`
- `final_metrics.json`
- `training_curves.png`
- `pose_landmarker_full.task`
- `final_checkpoint.pt`

The main R3D-18 checkpoint is larger than the regular GitHub 100 MB file limit, so it is stored through Git LFS:

```text
models/final_checkpoint.pt
```

After cloning the repository, make sure Git LFS is installed and run:

```bash
git lfs pull
```
