# SMuRF: Swin Transformer-based MultiModal and Multi-Region Data Fusion Framework

## Overview

SMuRF is a deep learning framework designed for predicting outcomes in Oral Squamous Cell Carcinoma (OPSCC) using multimodal data fusion. It integrates radiology (CT scans) and pathology images via Swin Transformers to perform multitask learning, including grade classification and survival prediction. The framework supports multi-region analysis (tumor and lymph nodes) and leverages advanced transformer architectures for robust feature extraction and fusion.

This project is based on the research paper: [SMuRF: Swin Transformer-based MultiModal and Multi-Region Data Fusion Framework to Predict OPSCC Outcomes](https://example.com/paper-link) (placeholder link).

## Features

- **Multimodal Fusion**: Combines radiology and pathology modalities.
- **Multi-Region Analysis**: Handles tumor and lymph node regions separately.
- **Multitask Learning**: Simultaneous grade classification and survival prediction.
- **Swin Transformer Backbone**: Utilizes hierarchical vision transformers for spatial feature extraction.
- **Customizable**: Supports various fusion types, loss functions, and hyperparameters.

## Installation

### Prerequisites

- Python >= 3.8
- uv (Python package manager and virtual environment tool)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SMuRF_MultiModal
   ```

2. Create and activate a virtual environment using uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

   This will install all required packages as specified in `pyproject.toml`, using the configured cernet pip mirror for faster downloads in China.

## Usage

### Data Preparation

The framework expects data in specific formats:
- **Radiology**: CT scans in NIfTI format (.nii.gz).
- **Pathology**: Embeddings from CLAM (Cancer Lymphocyte Assessment in MSI) or raw images.
- **Metadata**: CSV files with patient information, grades, survival times, and censoring status.

Place your data in a directory structure like:
```
data/
├── radiology/
│   ├── patient1/
│   │   └── CT_img.nii.gz
│   └── ...
├── pathology/
│   ├── clam_output/
│   │   ├── format1/
│   │   │   ├── embeddings/
│   │   │   │   └── patient1.npy
│   │   │   └── ...
│   └── ...
└── data_table.csv
```

Update the `--dataroot` argument to point to your data directory.

### Training

Run the main training script with uv:
```bash
uv run python main.py --dataroot /path/to/data --exp_name my_experiment --feature_type raptomic --fusion_type fused_attention --task multitask --n_epochs 100 --batch_size 16
```

Key arguments:
- `--dataroot`: Path to data directory.
- `--feature_type`: Modalities to use (e.g., `raptomic` for radiology + pathology).
- `--fusion_type`: Fusion method (e.g., `fused_attention`).
- `--task`: Task type (`multitask`, `grade`, or `survival`).
- `--gpu_ids`: GPU IDs (e.g., `0` for single GPU).
- `--cv`: Enable cross-validation.

For full options, run:
```bash
uv run python main.py --help
```

### Evaluation

After training, models are saved in the `checkpoints/` directory. Use the same script for evaluation by setting appropriate flags (e.g., disable training).

### Visualization

Use `grad_cam_final_bolin.py` for Grad-CAM visualizations to interpret model predictions.

## Project Structure

- `main.py`: Main training and evaluation script.
- `datasets.py`: Dataset classes for loading and preprocessing data.
- `models.py`: Model definitions, including Swin Transformers and fusion layers.
- `losses.py`: Loss functions (Cox, MultiTask, MMO).
- `utils.py`: Utility functions for preprocessing, metrics, and optimization.
- `swintransformer.py`: Swin Transformer implementation.
- `parameters.py`: Argument parsing.
- `preprocessing.py`: Data preprocessing scripts.
- `building_code.ipynb`: Jupyter notebook for code development.
- `docs/`: Documentation files.
  - `@AGENTS.md`: Analysis of project components and architecture.
  - `README.md`: This file.

## Dependencies

All dependencies are listed in `pyproject.toml`. Key packages include:
- PyTorch ecosystem (torch, torchvision, torchaudio)
- Data processing (numpy, pandas, medpy, opencv-python)
- Machine learning (scikit-learn, lifelines)
- Visualization (matplotlib)
- Others (einops, h5py, tqdm, etc.)

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and add tests.
4. Submit a pull request.

Please follow the code style guidelines and ensure all tests pass.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this code in your research, please cite:
```
@article{smurf2023,
  title={SMuRF: Swin Transformer-based MultiModal and Multi-Region Data Fusion Framework to Predict OPSCC Outcomes},
  author={Authors},
  journal={Journal},
  year={2023}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.