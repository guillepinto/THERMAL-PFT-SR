# Thermal Single Image Super-Resolution

This repository is a fork of the implementation of the paper "Progressive Focused Transformer for Single Image Super-Resolution", CVPR, 2025. Here we adapted the code to:

1. Fine-tune the model for thermal images.
2. Implement a cross-channel loss to supervise the training with RGB images inspired by "ThermalNeRF: Thermal Radiance Fields", ICCP, 2024.

[![Technical report](https://img.shields.io/badge/📜-Link%20to%20paper-blue)](static/paper.pdf)
[![Pretrained Models](https://img.shields.io/badge/Pretrained%20Models-AD1C18.svg?logo=Googledrive)](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing)

By [Guillermo Pinto](https://guillepinto.github.io/), [Andrea Parra](https://github.com/andpgate), Nicolás Ramírez.

> **Abstract:** Thermal Image Super-Resolution (TISR) is crucial to enhance imagery from low-cost infrared sensors, however, its low resolution, weak contrast, and noise present significant challenges for traditional methods. In this work, we adapt the Progressive Focused Transformer (PFT), originally trained in visible-spectrum data, by fine-tuning with thermal-RGB pairs. We introduce a cross-channel loss to transfer RGB textures and a total variation (TV) loss to preserve the characteristic smoothness of thermal images. Using the CIDIS dataset, our method achieves 32,0 dB PSNR and 0,904 SSIM, outperforming bicubic interpolation and SwinFuSR. Ablation studies show that the two losses are complementary, and we observe an emergent colorizing property of thermal images when the cross-channel loss is used in isolation. This results demonstrated that leveraging Transformers designed for the visible spectrum provides an effective and computationally lightweight solution TISR.
> 
> <img width="800" src="figures/pft_m.png"> 
> <br/>
> <img width="800" src="figures/PFT.png"> 



## Contents
1. [Enviroment](#environment)
1. [Inference](#inference)
1. [Training](#training)
1. [Testing](#testing)
1. [Results](#results)
1. [Visual Results](#visual-results)
1. [Visualization of Attention Distributions](#visualization-of-attention-distributions)
1. [Acknowledgements](#acknowledgements)
1. [Citation](#citation)

## Environment
- Python 3.9
- PyTorch 2.5.0

### Installation
```bash
git clone https://github.com/LabShuHangGU/PFT-SR.git

conda create -n PFT python=3.9
conda activate PFT

pip install -r requirements.txt
python setup.py develop

cd ./ops_smm
./make.sh
```

## Inference
Using ```inference.py``` for fast inference on single image or multiple images within the same folder.
```bash
# For classical SR
python inference.py -i inference_image.png -o results/test/ --scale 4 --task classical
python inference.py -i inference_images/ -o results/test/ --scale 4 --task classical

# For lightweight SR
python inference.py -i inference_image.png -o results/test/ --scale 4 --task lightweight
python inference.py -i inference_images/ -o results/test/ --scale 4 --task lightweight
```
The PFT SR model processes the image ```inference_image.png``` or images within the ```inference_images/``` directory. The results will be saved in the ```results/inference/``` directory.


## Training
### Data Preparation
- Download the training dataset DF2K ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) and put them in the folder `./datasets`.
- It's recommanded to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
- Refer to the training configuration files in `./options/train` folder for detailed settings.
- PFT (Classical Image Super-Resolution)
```bash
# batch size = 8 (GPUs) × 4 (per GPU)
# training dataset: DF2K

# ×2 scratch, input size = 64×64, 500k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/001_PFT_SRx2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterationsCUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/002_PFT_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/003_PFT_SRx4_finetune.yml --launcher pytorch
```

- PFT-light (Lightweight Image Super-Resolution)
```bash
# batch size = 4 (GPUs) × 8 (per GPU)
# training dataset: DIV2K

# ×2 scratch, input size = 64×64, 500k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/101_PFT_light_SRx2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/102_PFT_light_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/103_PFT_light_SRx4_finetune.yml --launcher pytorch
```


## Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_4Fy9emAcqdiBwVM6FvbJU50LCtaBoMt/view?usp=sharing)]) and put them in the folder `./datasets`.

### Pretrained Models
- Download the [pretrained models](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing) and put them in the folder `./experiments/pretrained_models`.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.
- PFT (Classical Image Super-Resolution)
- **We have now integrated the patchwise_testing strategy into basicsr/models/pft_model.py. This update allows for successful inference on RTX 4090 GPUs without running into memory issues.**
```bash
python basicsr/test.py -opt options/test/001_PFT_SRx2_scratch.yml
python basicsr/test.py -opt options/test/002_PFT_SRx3_finetune.yml
python basicsr/test.py -opt options/test/003_PFT_SRx4_finetune.yml
```

- PFT-light (Lightweight Image Super-Resolution)
```bash
python basicsr/test.py -opt options/test/101_PFT_light_SRx2_scratch.yml
python basicsr/test.py -opt options/test/102_PFT_light_SRx3_finetune.yml
python basicsr/test.py -opt options/test/103_PFT_light_SRx4_finetune.yml
```


## Results
- Classical Image Super-Resolution

<img width="800" src="figures/classical.png">

- Lightweight Image Super-Resolution

<img width="800" src="figures/lightweight.png">

## Visual Results

<img width="800" src="figures/visual_classical.png">

<img width="800" src="figures/visual_lightweight.png">

## Visualization of Attention Distributions
<img width="800" src="figures/attention_distributions.png">

1. Uncomment the code at this location to enable attention map saving: https://github.com/LabShuHangGU/PFT-SR/blob/master/basicsr/archs/pft_arch.py#L316-L328
2. Perform inference on the image you want to visualize to generate and save the attention maps under the ./results/Attention_map directory:
```
python inference.py -i inference_image.png -o results/test/ --scale 4 --task lightweight
```
3. Modify the corresponding paths and specify the window location you want to visualize in VisualAttention.py (the window is indexed from left to right, top to bottom, assuming the stride equals the window size).
4. Run the following command to visualize the attention map:
```
python VisualAttention.py
```
It should be noted that PFT employs a shift window operation, resulting in different corresponding positions in the attention maps between odd-numbered and even-numbered layers.

## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary.git).

## Citation

```
@article{long2025progressive,
  title={Progressive Focused Transformer for Single Image Super-Resolution},
  author={Long, Wei and Zhou, Xingyu and Zhang, Leheng and Gu, Shuhang},
  journal={arXiv preprint arXiv:2503.20337},
  year={2025}
}
```
