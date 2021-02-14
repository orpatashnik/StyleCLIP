# Editing and Generating Images Using StyleGAN And CLIP

In this repo, we provide code for edit and generating images, by optimizing latent codes in the StyleGAN's latent space, guided by CLIP.

### Editing Examples

![](img/me.png)
![](img/ariana.png)
![](img/federer.png)
![](img/styles.png)

### Setup

We rely on the official implementation of [CLIP](https://github.com/openai/CLIP), 
and the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2.
We modified some parts of the StyleGAN implementation, so that the whole implementation is native pytorch.

#### Requirements
- Anaconda
- Pretrained StyleGAN2 generator (can be downloaded from [here](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing))

In addition, run the following commands:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```


### Usage

Given a desired text description, one can both edit a given image, or generate a random image that best fits to the description.
Both operations can be done through the `main.py` script, or the notebook.

#### Editing
To edit an image set `--mode=edit`. Editing can be done on both provided latent vector, and on a random latent vector from StyleGAN's latent space.
We recommend to adjust the `--l2_lambda` according to the desired edit. 
From our experience, different edits require different values of this parameter.

#### Generating Free-style Images
To generate a free-style image set `--mode=free_generation`.

