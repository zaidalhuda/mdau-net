# MDAU-Net: Multi-Scale U-Net with Dual Attention Module for Pavement Crack Segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxx.svg)](https://zenodo.org/doi/10.5281/zenodo.xxxxx)

This is the official implementation of the proposed MDAU-Net model in the paper "MDAU-Net: A Multi-Scale U-Net with Dual Attention Module for Pavement Crack Segmentation". The model was trained and tested on two datasets "DeepCrack and Crack500". 

If you want to use the original datasets, please refer to the links below:
1. DeepCrack: (https://github.com/yhlleo/DeepCrack)
2. Crack500: (https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001?select=CRACK500)

![Model Architecture](https://github.com/username/mdau-net/blob/main/Model_Architecture.png?raw=true)


## Requirements

1. python 3.8+
2. pytorch 2.0.0
3. torchvision 0.15.0
4. opencv 4.7.0
5. numpy 1.21.0
6. matplotlib 3.5.0
7. albumentations 1.3.0
8. tqdm 4.64.0

## Quick Start

### 1. Installation
```bash
git clone https://github.com/username/mdau-net.git
cd mdau-net
pip install -r requirements.txt
```

### 2. Download Datasets
Place your datasets in the `Datasets/` folder with the following structure:
```
Datasets/
├── DeepCrack/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
└── Crack500/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

### 3. Training
```bash
# Train on DeepCrack
python Codes/train.py --dataset deepcrack --epochs 100 --batch_size 8

# Train on Crack500
python Codes/train.py --dataset crack500 --epochs 80 --batch_size 8
```

### 4. Testing
```bash
# Test trained model
python Codes/test.py --dataset deepcrack --model Checkpoints/mdau_net_deepcrack.pth

# Inference on single image
python Codes/inference.py --image path/to/image.jpg --model Checkpoints/mdau_net_deepcrack.pth
```

## Performance Results

| Dataset | mIoU | Precision | Recall | F1-Score |
|---------|------|-----------|--------|----------|
| **DeepCrack** | **90.9%** | **91.8%** | **88.8%** | **90.3%** |
| **Crack500** | **80.3%** | **76.8%** | **77.2%** | **77.0%** |

## Model Architecture

MDAU-Net combines three key components:
1. **Multi-Scale U-Net Encoder**: Processes input at multiple scales (0.5x, 1.0x, 1.5x)
2. **Dual Attention Module (DAM)**: Combines Global Average Pooling and Position Attention
3. **Hybrid Loss Function**: Boundary Loss + Weighted Cross-Entropy + Dice Loss


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10481232,
  author={Al-Huda, Zaid and Peng, Bo and Al-antari, Mugahed A. and Algburi, Riyadh Nazar Ali and Saleh, Radhwan A. A. and Moghalles, Khaled},
  booktitle={2023 18th International Conference on Intelligent Systems and Knowledge Engineering (ISKE)}, 
  title={MDAU-Net: A Multi-Scale U-Net with Dual Attention Module for Pavement Crack Segmentation}, 
  year={2023},
  pages={170-177},
  doi={10.1109/ISKE60036.2023.10481232}}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please contact:
- Zaid Al-Huda: [zaid@stir.cdu.edu.cn](mailto:zaid@stir.cdu.edu.cn)
