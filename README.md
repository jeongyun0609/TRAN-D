# TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update

Official implementation of **TRAN-D** (ICCV 2025). This repository bundles the segmentation module (tran-d_seg) and the modified 2D Gaussian Splatting module (tran-d_gs). The preprint is available on [arXiv:2507.11069](https://arxiv.org/abs/2507.11069).

## Repository structure

```
TRAN-D/
├── tran-d_seg/   # Grounded SAM2 segmentation + data preprocessing
└── tran-d_gs/    # Gaussian Splatting + physics simulation
```

## TODO
- [X] Code release
- [X] Dataset release
- [X] Release of the Grounding DINO checkpoint for transparent objects
- [X] Docker release

## Environment setup
We tested our code in CUDA 11.8 and python 3.10 / 3.11.

1. We provide Docker Environment for both modules. Segmentation module is based on [Grounded SAM2](https://github.com/IDEA-Research/Grounded-Segment-Anything). Gaussian Splatting module is based on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) and includes our customized gaussian renderer (`tran-d_gs/submodules/diff-surfel-seg-rasterization`).

   ```terminal
   docker pull jungyun0609/grounded_sam_2:1.0
   docker pull jungyun0609/2dgs_depth_trans:1.0
   ```

2. If you want to install to your own environment, please refer to each module's `environment.yml`. While `tran-d_seg` is almost same with Grounded SAM2, `tran-d_gs` requires some additional submodules such as `mpm_engine`.

   ```terminal
   # after installation of 2D Gaussian Splatting
   pip install ./submodules/diff-surfel-seg-rasterization
   pip install taichi==1.7.3
   ```

## Data & checkpoint preparation

You can download the dataset and checkpoint from below:
- Dataset of transparent object sequences: [this link](https://drive.google.com/drive/folders/1Tv6oTm9ggFXTyM8_zQZ_coocwbsFzxmk)
- Grounding DINO checkpoint for transparent object: [this link](https://drive.google.com/file/d/1-Vhh1reoAfJeTlGRmfwkLkiXT2jD0Ark)

For SAM2 checkpoint, we use the same one with the original.

The segmentation module expects the following dataset layout:

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
   # in jungyun0609/grounded_sam_2 docker
   cd tran-d_seg
   bash preprocess.sh
   ```

2. **Gaussian Splatting** – edit `gpu`, `data_dir`, and `result_dir` in `tran-d_gs/run.sh`, then run:

   ```bash
   # in jungyun0609/2dgs_depth_trans:1.0
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