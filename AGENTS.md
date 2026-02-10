<file_path>
SMuRF_MultiModal/docs/@AGENTS.md
</file_path>

<edit_description>
Create @AGENTS.md document analyzing the project's components and architecture
</edit_description>

# @AGENTS.md: SMuRF MultiModal Framework Analysis

## Project Overview
SMuRF (Swin Transformer-based MultiModal and Multi-Region Data Fusion Framework) is designed for predicting outcomes in Oral Squamous Cell Carcinoma (OPSCC) using multimodal data fusion. The framework integrates radiology and pathology modalities via Swin Transformers for multitask learning (grade classification and survival prediction).

## Core Components (Agents/Modules)

### 1. Data Handling Agents
- **RadPathDataset**: Loads and preprocesses radiology (CT scans) and pathology images. Handles tumor/lymph node regions, applies augmentations, and extracts embeddings.
- **PathDataset**: Specialized for pathology-only data, loading CLAM embeddings.
- **RadDataset**: Radiology-focused dataset without pathology.
- **HandCraftedFeaturesDataset**: Processes engineered features for baseline comparisons.
- **Custom Collates**: `custom_collate` and `custom_collate_pathology` for batching multimodal data with padding.

### 2. Model Agents
- **SwinTransformer**: Core transformer architecture for feature extraction from 3D/2D images. Supports hierarchical feature maps.
- **MultiTaskModel**: Multilayer perceptron for fusion and prediction, handling grade and hazard outputs.
- **FusionModelBi**: Bipartite fusion model combining modalities.
- **Model, RModel, PModel**: Specialized models for different modalities (radiology, pathology).

### 3. Loss and Training Agents
- **MultiTaskLoss**: Combines BCE loss for grading and Cox loss for survival.
- **CoxLoss**: Implements Cox proportional hazards model for survival analysis.
- **MMOLoss**: Matrix nuclear norm regularization for multimodal embeddings.

### 4. Utility Agents
- **Utils**: Data preprocessing (windowing, cropping), optimizer/scheduler definition, metrics computation (CI, AUC).
- **Parameters**: Argument parsing for training configurations (e.g., fusion type, modalities, hyperparameters).

### 5. Training Orchestrator
- **Main.py**: Central script managing training loops, cross-validation, model saving, and evaluation.

## Architecture Flow
1. Data loading and preprocessing (datasets.py, utils.py).
2. Feature extraction via Swin Transformers (swintransformer.py).
3. Multimodal fusion and prediction (models.py).
4. Loss computation and optimization (losses.py, main.py).
5. Evaluation and metrics (utils.py).

## Key Features
- Multimodal fusion: Radiology + Pathology.
- Multi-region analysis: Tumor and lymph nodes.
- Multitask: Grade classification + Survival prediction.
- Transformer-based: Leverages Swin Transformer for spatial hierarchies.

This analysis highlights the modular "agents" enabling robust multimodal prediction in OPSCC.

## Related documentation
- [Feature extractor guide](docs/feature_extractor.md): Implementation notes and usage examples for the Swin-based radiology (`SwinTransformerRadiologyModel`) and pathology (`SwinTransformerPathologyModel`) feature extractors — includes input/output shapes, forward-flow explanation, and a recommended `FeatureExtractor` wrapper for reuse.