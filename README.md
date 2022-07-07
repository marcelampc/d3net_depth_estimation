# Deep Depth-from-Defocus (Deep-DFD)

## Network Architecture

This code implements the Dense Deep Depth Estimation Network (D3-Net) in PyTorch, from the paper:

[On regression losses for deep depth estimation](https://marcelampc.github.io/material/docs/2018/regression_losses_icip_2018.pdf), [Marcela Carvalho](http://mcarvalho.ml/), [Bertrand Le Saux](https://blesaux.github.io/), Pauline Trouvé-Peloux, Andrés Almansa, Frédéric Champagnat, ICIP 2018.

<figure>
  <img src="/images/d3_net_architecture.png" width="70%" >
  <figcaption>Fig.1 - D3-Net architecture. </figcaption>
</figure>

If you use this work for your projects, please take the time to cite our ICIP paper:

```
@article{Carvalho2018icip,
  title={On regression losses for deep depth estimation},
  author={Marcela Carvalho and Bertrand {Le Saux} and Pauline Trouv\'{e}-Peloux and Andr\'{e}s Almansa and Fr\'{e}d\'{e}ric Champagnat},
  journal={ICIP},
  year={2018},
  publisher={IEEE}
}
```

## Indoor and outdoor DFD dataset

We also publish the dataset for Deep Depth from Defocus estimation created using a DSLR camera and a Xtion sensor (figure 1). This dataset was presented in in:

[Deep Depth from Defocus: how can defocus blur improve 3D estimation using dense neural networks?](https://arxiv.org/pdf/1809.01567.pdf), [Marcela Carvalho](http://mcarvalho.ml/), [Bertrand Le Saux](https://blesaux.github.io/), Pauline Trouvé-Peloux, Andrés Almansa, Frédéric Champagnat, 3DRW ECCV Workshop 2018.

 The [dfd_indoor](/dfd_datasets/dfd_indoor) dataset contains 110 images for training and 29 images for testing. The [dfd_outdoor](/dfd_datasets/dfd_outdoor) dataset contains 34 images for tests; no ground truth was given for this dataset, as the depth sensor only works on indoor scenes.

<figure>
  <img src="/images/dfd_dataset.png" width="100%" >
  <figcaption>Fig.2 - Platform to acquire defocused images and corresponding depth maps. </figcaption>
</figure>

<!-- Add example of the dataset -->

---
### BibTex reference:

## Generate Synthetic Defocused Data
In [generate_blurred_dataset.m](https://github.com/marcelampc/d3net_depth_estimation/matlab/generate_blurred_dataset.m), change lines 14 to 18 to corresponding paths in your computer and run.

If you use this work for your projects, please take the time to cite our ECCV Workshop paper:

```
@article{Carvalho2018eccv3drw,
  title={Deep Depth from Defocus: how can defocus blur improve {3D} estimation using dense neural networks?},
  author={Marcela Carvalho and Bertrand {Le Saux} and Pauline Trouv\'{e}-Peloux and Andr\'{e}s Almansa and Fr\'{e}d\'{e}ric Champagnat},
  journal={3DRW ECCV Workshop},
  year={2018},
  publisher={IEEE}
}
```

---

### Requirements
Requires Python 3.6 with pip and the following libraries:

- Linux
- Python 3.6+
- PyTorch 1.9
- Cuda 10.2
- Visdom


## Depth Estimation
### Installation

Using [Conda](https://www.anaconda.com/products/individual#linux):

```shell
cd pytorch
conda create -n d3net-env python=3.7 -y
conda activate d3net-env
conda config --append channels conda-forge
sh pre_install.sh
```

Create a **checkpoints** and **results** folder, or a symbolic link to your respective folders.

```shell
mkdir checkpoints results

# Or
ln -s path_to_checkpoints checkpoints
ln -s path_to_results results
```

## Train D3-Net on DFD Dataset

Run the script in [train_dfd.sh](pytorch/std_scripts/train_dfd.sh) to train D3-Net on DFD Indoor Dataset. In your terminal, enter your conda environment

```shell
# Start Visdom
visdom -p 8001

# In another terminal, if not already inside,
cd pytorch

# Run train
sh std_scripts/train_dfd.sh
```


### Usage

## Generate Synthetic Defocused Data
In [generate_blurred_dataset.m](https://github.com/marcelampc/d3net_depth_estimation/matlab/generate_blurred_dataset.m), change lines 14 to 18 to corresponding paths in your computer and run.

\[To be added\]

## License
Code (scripts and Jupyter notebooks) are released under the GPLv3 license for non-commercial and research purposes only. For commercial purposes, please contact the authors.


<!-- (/images/DSLR_Xtion.png | width=48 "Platform to acquire defocused images and corresponding depth maps.") -->
