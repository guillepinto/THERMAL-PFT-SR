<div align="center">

# Thermal Single Image Super-Resolution

[![Technical report](https://img.shields.io/badge/📜-Link%20to%20paper-blue)](static/paper.pdf)
[![Pretrained Models](https://img.shields.io/badge/Pretrained%20Models-AD1C18.svg?logo=Googledrive)](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing)

[Guillermo Pinto](https://guillepinto.github.io/), [Andrea Parra](https://github.com/andpgate), Nicolás Ramírez.

> **Abstract:** Thermal Image Super-Resolution (TISR) is crucial to enhance imagery from low-cost infrared sensors, however, its low resolution, weak contrast, and noise present significant challenges for traditional methods. In this work, we adapt the Progressive Focused Transformer (PFT), originally trained in visible-spectrum data, by fine-tuning with thermal-RGB pairs. We introduce a cross-channel loss to transfer RGB textures and a total variation (TV) loss to preserve the characteristic smoothness of thermal images. Using the CIDIS dataset, our method achieves 32,0 dB PSNR and 0,904 SSIM, outperforming bicubic interpolation and SwinFuSR. Ablation studies show that the two losses are complementary, and we observe an emergent colorizing property of thermal images when the cross channel loss is used in isolation. This results demonstrated that leveraging Transformers designed for the visible spectrum provides an effective and computationally lightweight solution TISR.
>
> <img width="800" src="figures/pft_m.png"> 
> <img width="800" src="figures/PFT.png"> 
</div>

This repository is a fork of the implementation of the paper "Progressive Focused Transformer for Single Image Super-Resolution", CVPR, 2025. Here we adapted the code to:

1. Fine-tune the model for thermal images.
2. Implement a cross-channel loss to supervise training with RGB images inspired by "ThermalNeRF: Thermal Radiance Fields", ICCP, 2024.

## Contents
1. [Enviroment](#environment)
1. [Inference](#inference)
1. [Training](#training)
1. [Testing](#testing)
1. [Results](#results)
1. [Visual Results](#visual-results)
1. [Emerging Property](#emerging-property)
1. [Visualization of Attention Distributions](#visualization-of-attention-distributions)
1. [Acknowledgements](#acknowledgements)
1. [Citation](#citation)

## Environment
- Python 3.9
- PyTorch 2.5.0

### Installation
```bash
git clone https://github.com/guillepinto/PFT-SR.git

conda create -n PFT python=3.9
conda activate PFT

pip install -r requirements.txt && python setup.py develop

cd ./ops_smm && ./make.sh && cd ..
```
### Preparing for Inference

Lightweight PFT models were only trained for ×3 super-resolution due to computational constraints. Inference at other scales will automatically fallback to the official PFT models.

1. Download the checkpoints for the models you need.
2. For example, to run ×4 lightweight inference, download the checkpoint named `103_PFT_light_SRx4_finetune`.
3. The original PFT pretrained models are available **[here](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing)**.
   Our fine-tuned lightweight ×3 models are available in the following links: (i) **[fine-tuned (L1)](https://huggingface.co/SemilleroCV/thermal-pft-light-srx3)** and (ii) **[fine-tuned (L1 + cross-channel + TV)](https://huggingface.co/SemilleroCV/thermal-pft-light-srx3-cross-channel-tv)**.
4. Place all downloaded checkpoints in:

   ```
   ./experiments/pretrained_models
   ```

## Inference
Using `inference.py` for fast inference on single image or multiple images within the same folder.
```bash
# For lightweight SR
python inference.py -i inference_image.png -o results/test/ --scale 3 --task lightweight
python inference.py -i inference_images/ -o results/test/ --scale 3 --task lightweight
```
The PFT SR model processes the image ```inference_image.png``` or images within the ```inference_images/``` directory. The results will be saved in the ```results/inference/``` directory.


## Training
### Data Preparation
- Download the training dataset [CIDIS](https://github.com/vision-cidis/CIDIS-dataset) and put it in the folder `./datasets` (or wherever you want, you have to specify the path later).

### Dataset Preprocessing

Below is the full procedure to generate the low-resolution (LR) and ground-truth (GT) pairs used for training. Apply the same process to both train and validation splits.

1. Downscale the original images to the desired LR size using:
   ```bash
   python downscale_data.py --root_dir <input_path> --out_dir <lr_output_path> --width 64 --height 64
   ```
   In our experiments, LR images were downscaled to **64×64**.
2. Generate the corresponding GT images by upscaling the LR dataset by a factor of ×3:
   ```bash
   python downscale_data.py --root_dir <lr_output_path> --out_dir <gt_output_path> --width 256 --height 256
   ```

### Training Commands
Is **important** to update your training configuration YAMLs to point to the correct LR and GT dataset directories. Refer to the training configuration files in `./options/train` folder for detailed settings.

- PFT-light (Lightweight Image Super-Resolution)
```bash
# batch size = 1 (GPUs) × 4 (per GPU)
# training dataset: CIDIS

# ×3 PFT finetune, input size = 64×64, 50k iterations
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=1145  basicsr/train.py -opt options/train/202_Thermal_PFT_light_SRx3_finetune.yml --launcher pytorch

# ×3 PFT finetune with cross-channel and TV, input size = 64×64, 50k iterations
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=1145  basicsr/train.py -opt options/train/202_Thermal_PFT_light_SRx3_finetune_cross_channel_tv.yml --launcher pytorch
```
You can also run the other experiments by just updating the configuration on the `-opt` parameter.

**Note:** All experiments were trained on a single NVIDIA T4 (Lightning AI). Due to this constraint, we only trained lightweight variants of PFT.

## Testing
### Data Preprocessing

Repeat the same preprocessing steps for the **validation/testing** data (refer to [Training](https://github.com/guillepinto/PFT-SR?tab=readme-ov-file#training)):
1. Downscale to 64×64.
2. Upscale that version by ×3 to obtain 256×256 GT.
3. Update LR and GT dataset paths in the testing YAML files.

### Pretrained Models
- Download the pretrained models and put them in the folder `./experiments/pretrained_models`.
  
The original PFT pretrained models are available **[here](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing)**.
   Our fine-tuned lightweight ×3 models are available in the following links: (i) **[fine-tuned (L1)](https://huggingface.co/SemilleroCV/thermal-pft-light-srx3)** and (ii) **[fine-tuned (L1 + cross-channel + TV)](https://huggingface.co/SemilleroCV/thermal-pft-light-srx3-cross-channel-tv)**.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.
- PFT-light (Lightweight Image Super-Resolution)
```bash
python basicsr/test.py -opt options/test/102_PFT_light_SRx3_finetune.yml
python basicsr/test.py -opt options/test/202_Thermal_PFT_light_SRx3_finetune_thermal.yml
python basicsr/test.py -opt options/test/202_Thermal_PFT_light_SRx3_finetune_cross_channel_tv.yml
```

## Results

- Lightweight Image Super-Resolution on CIDIS validation set.

| Method                | Scale | PSNR ↑    | SSIM ↑   |
|-----------------------|:-----:|----------:|---------:|
| INTER_CUBIC           |       | 27.7669   | 0.8332   |
| SwinFuSR (w RGB)      |       | 24.9115   | 0.7144   |
| PFT-*light* (baseline)|  ×3   | 31.8215   | 0.9013   |
| Ours (w RGB)          |       | 31.6725   | 0.8998   |
| **Ours**              |       | **32.0038** | **0.9044** |

## Visual Results

<img width="800" src="figures/qualitative-1-sota.png">

## Emerging Property

When the model is trained using the RGB image as ground truth it tends to colorize the result departing from the thermal image.

<img width="800" src="figures/qualitative-3-emerging-rgb-property.png">

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
This code is built on [PFT-SR](https://github.com/LabShuHangGU/PFT-SR), [BasicSR](https://github.com/XPixelGroup/BasicSR) and [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary.git).

## Citation

```
@article{pft-tisr,
  title={Thermal Single Image Super-Resolution},
  author={Pinto, Guillermo and Parra, Andrea and Ramírez, Nicolás},
  institution={Preprint, Universidad Industrial de Santander},
  year={2025},
  note={Technical Report, available at \url{https://github.com/guillepinto/PFT-SR/static/paper.pdf}},
}
```
