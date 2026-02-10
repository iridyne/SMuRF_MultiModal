# SMuRF: Swin Transformer-based MultiModal and Multi-Region Data Fusion Framework

## Overview

SMuRF is a deep learning framework for predicting outcomes in Oral Squamous Cell Carcinoma (OPSCC) using multimodal data fusion. It integrates radiology (CT scans) and pathology images via Swin Transformers for multitask learning, including grade classification and survival prediction.

Key features:
- Multimodal fusion (radiology + pathology)
- Multi-region analysis (tumor and lymph nodes)
- Multitask: Grade classification + Survival prediction
- Transformer-based architecture

## Quick Start

### Installation
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Training
```bash
uv run python main.py --dataroot /path/to/data --feature_type raptomic --task multitask
```

For full details, see [docs/README.md](docs/README.md).

## Project Structure

- `main.py`: Main training script
- `datasets.py`: Data loading and preprocessing
- `models.py`: Model definitions
- `docs/`: Detailed documentation
  - [README.md](docs/README.md): Comprehensive guide
  - [@AGENTS.md](docs/@AGENTS.md): Component analysis

## Citation

If you use this code, please cite the original paper.

## License

MIT License