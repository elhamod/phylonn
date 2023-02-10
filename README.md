# Discovering Novel Biological Traits From Images Using Phylogeny-Guided Neural Networks
##### KDD'23 submission
![teaser](assets/phyloNN_toc.jpeg)

**Abstract** One of the grand challenges in biology is to find features of organisms--or traits--that define groups of organisms, their genetic and developmental underpinnings, and their interactions with environmental selection pressures. Traits can be physiological, morphological, and/or behavioral (e.g., {beak color, stripe pattern, and fin curvature}) and are integrated products of genes and the environment. The analysis of traits is critical for predicting the effects of environmental change or genetic manipulation, and to understand the process of evolution. For example, discovering traits that are heritable across individuals, or across species on the tree of life (also referred to as the phylogeny), can identify features useful for individual recognition or species classification, respectively, and is a starting point for linking traits to underlying genetic factors. Traits with such genetic or phylogenetic signal, termed evolutionary traits, are of great interest to biologists, as the history of genetic ancestry captured by such traits can guide our understanding of how organisms vary and evolve. This understanding enables tasks such as estimating the morphological features of ancestors, how they have responded to environmental changes, or even predicting the potential future course of trait changes. However, the measurement of traits is not straightforward and often relies on subjective and labor-intensive human expertise and definitions. Hence, trait discovery}has remained a highly label-scarce problem, hindering rapid scientific advancement.

## Requirements
Use:
```
pip install -r requirements.txt
```

## Dataset
In this work, we used a curated dataset of Teleost fish images from five ichthyological research collections that participated in  [the Great Lakes Invasives Network Project](https://greatlakesinvasives.org/portal/index.php) (GLIN). After obtaining the raw images from these collections, we handpicked a subset of about $11,000$ images and pre-processed them by resizing and appropriately padding each image to be of a $256\times256$ pixel resolution. Finally, we split the subset into a training set and a validation set of ratios $80\%$ and $20\%$, respectively. Our dataset includes images from 38 species of Teleost fishes with an average number of $40$ images per species.

These images can be found [here](https://drive.google.com/drive/folders/1gkau9TOP6hi76hY8FgP6VpII871SUHRD?usp=share_link).

## Overview of pretrained models
The pretrained model used in our experiments can be found [here](https://drive.google.com/drive/folders/1g-7LL6iPtqaUbQKAgxS8E8I5GvWW8gDZ?usp=share_link). 


## Training a model
In order to train one of our models, use the following command:
```
python main.py --prefix <prefix> --name <name> --postfix <postfix> --base <yaml file> -t True --gpus <comma-separated GPU indices>
```
* **prefix** : The path to where the model will be saved
* **name** : run name
* **postfix**: a postfix for the run's name
* **base**: the `yaml` config file.

The training script is similar and based on that provided by [taming-transformers](https://github.com/CompVis/taming-transformers)

Under `configs` directory, we have prepopulated some of the `yaml` files we have used in our own training.

## Generating images from a trained transformer.
Once a PhyloNN transformer model is trained, images can be generated using the following command:
```
python analysis/generate_with_transformer.py --config <path to analysis yaml file>
```
In the analysis `yaml` file:
* **ckpt_path** : The saved model
* **yaml_path** : The saved model's config
* **outputdatasetdir** : where the images are generated
* **save_individual_images**: whether to save individual images or just collate them in a grid
* **DEVICE**: GPU index to use
* **file_list_path**: path to training dataset
* **size**: image resolution


For vanilla VQGAN, use the same script that was provided by [taming-transformers](https://github.com/CompVis/taming-transformers)

```
python analysis/make_samples.py -r <path to ckpt file> -c <path to model's yaml file> --outdir <directory where to save the images> 
```
* **ckpt file** : The saved model


## Generating histograms 



## Baselines

### Latent Space Factorization (LSF)
LSF is a method for disentangling latent space, described in the paper [Latent Space Factorisation and Manipulation via Matrix Subspace Projection](https://arxiv.org/abs/1907.12385)

### Training a model - LSF

```
python main.py --name <name> --postfix <postfix> --base <yaml file> -t True --gpus <comma-separated GPU indices>
```
* **prefix** : The path to where the model will be saved
* **name** : run name
* **postfix**: a postfix for the run's name
* **base**: the `yaml` config file.

Under `configs` directory, the config file used for training with LSF is provided under the name `lsf.yaml`.

### Image translation - LSF

```
python analysis/translateLSF.py --config <yaml file>
```
* **config**: the `yaml` config file.
Under `analysis/configs` directory, we have prepopulated the `yaml` files we have used in our own experiment for image translation under the name `translateLSF.yaml`.

### TSNE plots - LSF

```
python analysis/tsneLSF.py --config <yaml file>
```
* **config**: the `yaml` config file.
Under `analysis/configs` directory, the config file used for generating this plot is provided under the name `tsneLSF.yaml`.

### Heatmap of cosine distance between latent representations - LSF

```
python analysis/heatmapLSF.py --config <yaml file>
```
* **config**: the `yaml` config file.
Under `analysis/configs` directory, the config file used for generating this plot is provided under the name `heatmapLSF.yaml`.

## Training on custom data

Training on your own dataset can be beneficial to get better tokens and hence better images for your domain.
Those are the steps to follow to make this work:
1. install the repo with `conda env create -f environment.yaml`, `conda activate scripts` and `pip install -e .`
1. put your .jpg files in a folder `your_folder`
2. create 2 text files a `xx_train.txt` and `xx_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/your_folder -name "*.jpg" > train.txt`)
3. adapt `configs/custom_vqgan.yaml` to point to these 2 files
4. run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.

## Data Preparation

### ImageNet
The code will try to download (through [Academic
Torrents](http://academictorrents.com/)) and prepare ImageNet the first time it
is used. However, since ImageNet is quite large, this requires a lot of disk
space and time. If you already have ImageNet on your disk, you can speed things
up by putting the data into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` (which defaults to
`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is one
of `train`/`validation`. It should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```

If you haven't extracted the data, you can also place
`ILSVRC2012_img_train.tar`/`ILSVRC2012_img_val.tar` (or symlinks to them) into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/` /
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`, which will then be
extracted into above structure without downloading it again.  Note that this
will only happen if neither a folder
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` nor a file
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/.ready` exist. Remove them
if you want to force running the dataset preparation again.

You will then need to prepare the depth data using
[MiDaS](https://github.com/intel-isl/MiDaS). Create a symlink
`data/imagenet_depth` pointing to a folder with two subfolders `train` and
`val`, each mirroring the structure of the corresponding ImageNet folder
described above and containing a `png` file for each of ImageNet's `JPEG`
files. The `png` encodes `float32` depth values obtained from MiDaS as RGBA
images. We provide the script `scripts/extract_depth.py` to generate this data.
**Please note** that this script uses [MiDaS via PyTorch
Hub](https://pytorch.org/hub/intelisl_midas_v2/). When we prepared the data,
the hub provided the [MiDaS
v2.0](https://github.com/intel-isl/MiDaS/releases/tag/v2) version, but now it
provides a v2.1 version. We haven't tested our models with depth maps obtained
via v2.1 and if you want to make sure that things work as expected, you must
adjust the script to make sure it explicitly uses
[v2.0](https://github.com/intel-isl/MiDaS/releases/tag/v2)!

### CelebA-HQ
Create a symlink `data/celebahq` pointing to a folder containing the `.npy`
files of CelebA-HQ (instructions to obtain them can be found in the [PGGAN
repository](https://github.com/tkarras/progressive_growing_of_gans)).

### FFHQ
Create a symlink `data/ffhq` pointing to the `images1024x1024` folder obtained
from the [FFHQ repository](https://github.com/NVlabs/ffhq-dataset).

### S-FLCKR
Unfortunately, we are not allowed to distribute the images we collected for the
S-FLCKR dataset and can therefore only give a description how it was produced.
There are many resources on [collecting images from the
web](https://github.com/adrianmrit/flickrdatasets) to get started.
We collected sufficiently large images from [flickr](https://www.flickr.com)
(see `data/flickr_tags.txt` for a full list of tags used to find images)
and various [subreddits](https://www.reddit.com/r/sfwpornnetwork/wiki/network)
(see `data/subreddits.txt` for all subreddits that were used).
Overall, we collected 107625 images, and split them randomly into 96861
training images and 10764 validation images. We then obtained segmentation
masks for each image using [DeepLab v2](https://arxiv.org/abs/1606.00915)
trained on [COCO-Stuff](https://arxiv.org/abs/1612.03716). We used a [PyTorch
reimplementation](https://github.com/kazuto1011/deeplab-pytorch) and include an
example script for this process in `scripts/extract_segmentation.py`.

### COCO
Create a symlink `data/coco` containing the images from the 2017 split in
`train2017` and `val2017`, and their annotations in `annotations`. Files can be
obtained from the [COCO webpage](https://cocodataset.org/). In addition, we use
the [Stuff+thing PNG-style annotations on COCO 2017
trainval](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
annotations from [COCO-Stuff](https://github.com/nightrome/cocostuff), which
should be placed under `data/cocostuffthings`.

### ADE20k
Create a symlink `data/ade20k_root` containing the contents of
[ADEChallengeData2016.zip](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
from the [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/).

## Training models

### FacesHQ

Train a VQGAN with
```
python main.py --base configs/faceshq_vqgan.yaml -t True --gpus 0,
```

Then, adjust the checkpoint path of the config key
`model.params.first_stage_config.params.ckpt_path` in
`configs/faceshq_transformer.yaml` (or download
[2020-11-09T13-33-36_faceshq_vqgan](https://k00.fr/uxy5usa9) and place into `logs`, which
corresponds to the preconfigured checkpoint path), then run
```
python main.py --base configs/faceshq_transformer.yaml -t True --gpus 0,
```

### D-RIN

Train a VQGAN on ImageNet with
```
python main.py --base configs/imagenet_vqgan.yaml -t True --gpus 0,
```

or download a pretrained one from [2020-09-23T17-56-33_imagenet_vqgan](https://k00.fr/u0j2dtac)
and place under `logs`. If you trained your own, adjust the path in the config
key `model.params.first_stage_config.params.ckpt_path` of
`configs/drin_transformer.yaml`.

Train a VQGAN on Depth Maps of ImageNet with
```
python main.py --base configs/imagenetdepth_vqgan.yaml -t True --gpus 0,
```

or download a pretrained one from [2020-11-03T15-34-24_imagenetdepth_vqgan](https://k00.fr/55rlxs6i)
and place under `logs`. If you trained your own, adjust the path in the config
key `model.params.cond_stage_config.params.ckpt_path` of
`configs/drin_transformer.yaml`.

To train the transformer, run
```
python main.py --base configs/drin_transformer.yaml -t True --gpus 0,
```

## More Resources
### Comparing Different First Stage Models
The reconstruction and compression capabilities of different fist stage models can be analyzed in this [colab notebook](https://colab.research.google.com/github/CompVis/phylonn/blob/master/scripts/reconstruction_usage.ipynb). 
In particular, the notebook compares two VQGANs with a downsampling factor of f=16 for each and codebook dimensionality of 1024 and 16384, 
a VQGAN with f=8 and 8192 codebook entries and the discrete autoencoder of OpenAI's [DALL-E](https://github.com/openai/DALL-E) (which has f=8 and 8192 
codebook entries).
![firststages1](assets/first_stage_squirrels.png)
![firststages2](assets/first_stage_mushrooms.png)

### Other
- A [video summary](https://www.youtube.com/watch?v=o7dqGcLDf0A&feature=emb_imp_woyt) by [Two Minute Papers](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg).
- A [video summary](https://www.youtube.com/watch?v=-wDSDtIAyWQ) by [Gradient Dude](https://www.youtube.com/c/GradientDude/about).
- A [weights and biases report summarizing the paper](https://wandb.ai/ayush-thakur/phylonn/reports/-Overview-Taming-Transformers-for-High-Resolution-Image-Synthesis---Vmlldzo0NjEyMTY)
by [ayulockin](https://github.com/ayulockin).
- A [video summary](https://www.youtube.com/watch?v=JfUTd8fjtX8&feature=emb_imp_woyt) by [What's AI](https://www.youtube.com/channel/UCUzGQrN-lyyc0BWTYoJM_Sg).
- Take a look at [ak9250's notebook](https://github.com/ak9250/phylonn/blob/master/tamingtransformerscolab.ipynb) if you want to run the streamlit demos on Colab.

### Text-to-Image Optimization via CLIP
VQGAN has been successfully used as an image generator guided by the [CLIP](https://github.com/openai/CLIP) model, both for pure image generation
from scratch and image-to-image translation. We recommend the following notebooks/videos/resources:

 - [Advadnouns](https://twitter.com/advadnoun/status/1389316507134357506) Patreon and corresponding LatentVision notebooks: https://www.patreon.com/patronizeme
 - The [notebook]( https://colab.research.google.com/drive/1L8oL-vLJXVcRzCFbPwOoMkPKJ8-aYdPN) of [Rivers Have Wings](https://twitter.com/RiversHaveWings).
 - A [video](https://www.youtube.com/watch?v=90QDe6DQXF4&t=12s) explanation by [Dot CSV](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg) (in Spanish, but English subtitles are available)

![txt2img](assets/birddrawnbyachild.png)

Text prompt: *'A bird drawn by a child'*

## Shout-outs
Thanks to everyone who makes their code and models available. In particular,

- The architecture of our VQGAN is inspired by [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
- The very hackable transformer implementation [minGPT](https://github.com/karpathy/minGPT)
- The good ol' [PatchGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [Learned Perceptual Similarity (LPIPS)](https://github.com/richzhang/PerceptualSimilarity)

## BibTeX

```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



