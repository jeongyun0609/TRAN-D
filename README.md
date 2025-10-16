# TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update

Official implementation of **TRAN-D** (ICCV 2025). This repository bundles the segmentation module (tran-d_seg) and the modified 2D Gaussian Splatting module (tran-d_gs). The preprint is available on [arXiv:2507.11069](https://arxiv.org/abs/2507.11069).

## Repository structure

```
TRAN-D/
├── tran-d_seg/   # Segmentation pipeline (Grounded SAM2 + data preprocessing)
└── tran-d_gs/    # Gaussian Splatting optimization, rendering, and evaluation
```

## TODO
- [X] Code release
- [ ] Dataset release
- [ ] Release of the Grounding DINO checkpoint for transparent objects
- [ ] Docker release

## Environment setup
We tested our code in CUDA 11.8 and python 3.10 / 3.11.

1. We are going to provide Docker Environment for both modules. Segmentation module is based on Grounded SAM2 repo from [here](https://github.com/IDEA-Research/Grounded-Segment-Anything). Gaussian Splatting module is based on 2D Gaussian Splatting from [here](https://github.com/hbb1/2d-gaussian-splatting) and includes our customized gaussian renderer (`tran-d_gs/submodules/diff-surfel-seg-rasterization`).


2. If you want to install to your own environment, please refer to each module's `environment.yml`. While `tran-d_seg` is almost same with Grounded SAM2, `tran-d_gs` requires some additional submodules such as `mpm_engine`.


## Data preparation

We are going to open-source dataset and Grounding DINO checkpoint which is tailored for transparent objects we used for the evaluation.

The segmentation stage expects the dataset layout used in the ICCV submission:

- Synthetic sequences: `<DATA_ROOT>/syn_test_xx/<train or test>_<0 or 1>/data`
- Real sequences: `<DATA_ROOT>/real_test_xx/train_<0 or 1>/camera`


Preprocessing will populate each sequence folder with:

- `transforms_sparse_<train or test>.json` or `transforms_sparse_<train or test>._01.json` - JSON used in Gaussian Splatting module to load the scene. `01` represents the scene after the object removal.
- `color_mask/` – RGB overlays produced from SAM2 masks.
- `binary_mask/` – Foreground/background masks per frame.
- `class/` – Per-pixel object IDs per frame.
- `obj_idx.json` – the ordered list of instances in the scene.

## Quick start

1. **Segmentation** – edit `gpu` and `data_dir` in `tran-d_seg/preprocess.sh`, then run:

   ```bash
   cd tran-d_seg
   bash preprocess.sh
   ```

2. **Gaussian Splatting optimisation** – edit `gpu`, `data_dir`, and `result_dir` in `tran-d_gs/run.sh`, then run:

   ```bash
   cd tran-d_gs
   bash run.sh
   ```


## Citation

If you use this codebase, please cite the ICCV 2025 paper:

```
@inproceedings{kim2025tran,
  title     = {TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update},
  author    = {Kim, Jeongyun and Jeong, Seunghoon and Kim, Giseop and Jeon, Myung-Hwan and Jun, Eunji and Kim, Ayoung},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

## License

Both submodules inherit their respective licenses (see `tran-d_seg/LICENSE` and `tran-d_gs/LICENSE.md`). Consult those files before using the code in commercial settings.