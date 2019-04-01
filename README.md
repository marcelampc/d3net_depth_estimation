# Deep Depth-from-Defocus (Deep-DFD)

In progress... We are still uploading models and improving the code to easy usage.

## Network Architecture

This code implements the Dense Deep Depth Estimation Network (D3-Net) in PyTorch, from the paper:

[On regression losses for deep depth estimation](http://mcarvalho.ml/material/docs/2018/regression_losses_icip_2018.pdf), [Marcela Carvalho](http://mcarvalho.ml/), [Bertrand Le Saux](https://blesaux.github.io/), Pauline Trouvé-Peloux, Andrés Almansa, Frédéric Champagnat, ICIP 2018.

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
=======

## Generate Synthetic Defocused Data
In [generate_blurred_dataset.m](https://github.com/marcelampc/d3net_depth_estimation/matlab/generate_blurred_dataset.m), change lines 14 to 18 to corresponding paths in your computer and run.

If you use this work for your projects, please take the time to cite our ECCV Workshop paper:
>>>>>>> 8238c86d1f6ee810ff821f9e297ad53743b2d75c

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

## Depth Estimation
### Setup
Requires Python 3.6 with pip and the following libraries:
```python
# Pytorch 0.4.0
conda install pytorch torchvision -c pytorch
# Visdom
pip install visdom
# Jupyter Notebook
pip install notebook
```

### Usage

## Generate Synthetic Defocused Data
In [generate_blurred_dataset.m](https://github.com/marcelampc/d3net_depth_estimation/matlab/generate_blurred_dataset.m), change lines 14 to 18 to corresponding paths in your computer and run.
=======
\[To be added\]

## License
Code (scripts and Jupyter notebooks) are released under the GPLv3 license for non-commercial and research purposes only. For commercial purposes, please contact the authors.
>>>>>>> 8238c86d1f6ee810ff821f9e297ad53743b2d75c


<!-- (/images/DSLR_Xtion.png | width=48 "Platform to acquire defocused images and corresponding depth maps.") -->
