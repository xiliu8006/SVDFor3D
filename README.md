# SVD for 3DGS-Enhancer

**Stable Video Diffusion Training Code for 3DGS-Enhancer**

## Training the SVD for 3DGS-Enhancer

To train the model, run the following command:

```bash
sbatch train_single_nodes.sh
```

## Inference with the SVD for 3DGS-Enhancer

To perform inference, use the following command:

```bash
sbatch multi_nodes_inference.sh
```

### Note

Before using SVD to generate results, ensure you use `split_scene_to_txt.py` to generate a folder containing text files that are written by scene names.

## Acknowledgements

This work builds upon the following codebases:
- [SVD_Xtends](https://github.com/pixeli99/SVD_Xtend)
- [Diffusers](https://github.com/huggingface/diffusers)

We thank the authors of these projects for their excellent contributions!

## Citation

If you use 3DGS-Enhancer in your research, please cite the following paper:

```bibtex
@article{liu20243dgs,
  title={3DGS-Enhancer: Enhancing unbounded 3D Gaussian splatting with view-consistent 2D diffusion priors},
  author={Liu, Xi and Zhou, Chaoyi and Huang, Siyu},
  journal={arXiv preprint arXiv:2410.16266},
  year={2024}
}
