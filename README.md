# CMPUT-428-Project

To run (say the synthetic desk easy dataset):

```bash
python src/s1_sfm.py data/synthetic_desk_easy/images datasets/synthetic_desk_easy;
python src/run_pipeline.py datasets/synthetic_desk_easy;
python src/evaluate.py datasets/synthetic_desk_easy --dataset data/synthetic_desk_easy --frames 64;
python src/view_ply.py datasets/synthetic_desk_easy/reconstruction_textured.ply;
```

For Henry's desk:
```bash
python src/run_pipeline.py datasets/workspace_custom;
python src/view_ply.py datasets/workspace_custom/reconstruction_textured.ply;
```
