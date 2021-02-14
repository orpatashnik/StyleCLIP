# Text-Guided Editing of Images (Using CLIP and StyleGAN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/orpatashnik/StyleCLIP/blob/master/playground.ipynb)

This repo contains a code and a few results of my experiments with StyleGAN and CLIP. 
Given a text description, my goal was to edit a given image, or generate a one.
The following diagram describes the way it works:

![](img/arch.png)

In this example, I took an image of Ariana Grande, inverted it using [e4e](https://github.com/omertov/encoder4editing),
 and edited the image so Ariana will look more tanned. 
 To keep the image close to the original, I also used an L2 loss between the optimized latent vector and the original one.

Hope you will enjoy it like I did!

### Editing Examples

Here are some examples, and first of all, some manipulated images of me :)
The description I used to obtain each edited image is written above or below it.

![](img/me.png)

And now a few celebs. The description I used to edit each image is written below it.

![](img/ariana.png)
![](img/federer.png)
![](img/styles.png)

### Setup

The code relies on the official implementation of [CLIP](https://github.com/openai/CLIP), 
and the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2.
Some parts of the StyleGAN implementation were modified, so that the whole implementation is native pytorch.

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

