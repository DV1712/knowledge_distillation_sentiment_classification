# Knowledge Distillation for Sentiment Classificaiton using Transformer models

A PyTorch-based project for extracting features from pre-trained transformer models (BERT, RoBERTa, DistilBERT etc.) and training neural classifiers for multiclass sentiment analysis. Includes support for knowledge distillation and hyperparameter grid search optimization.

## Overview

This project implements a complete pipeline for:
- **Feature Extraction**: Extract contextual features from multiple pre-trained transformer models
- **Neural Classification**: Train custom neural network classifiers on extracted features
- **Hyperparameter Tuning**: Grid search across configurations for optimal performance
- **Knowledge Distillation**: Optional student-teacher distillation for model compression
- **Comprehensive Evaluation**: Detailed metrics tracking and result analysis

## Key Components

### Notebooks

- **`pipeline.ipynb`**: Main multi-teacher feature extraction and neural classification pipeline for all models (BERT, RoBERTa, DistilBERT)
- **`single_teacher_pipeline.ipynb`**: Single teacher model pipeline with feature extraction and classification
- **`grid_search_pipeline.ipynb`** (old): Grid search over configurations and hyperparameters

### Utilities

- **`analyze_max_length.py`**: Analyzes dataset token lengths and recommends optimal `max_length` values for tokenization
- **`distillation_config.json`**: Configuration for knowledge distillation setup
- **`grid_search_config.json`**: Master configuration file for models, feature extraction, training, and distillation parameters

### Dataset

- **`dataset/multiclass_sentiments.jsonl`**: Input dataset containing multiclass sentiment annotations

### Results & Metrics

Results stored in CSV files:
- `feature_distillation_results.csv`, `feature_distillation_detailed_results.csv`: Distillation results
- `single_teacher_results.csv`, `single_teacher_detailed_results.csv`: Single teacher model results
- `grid_search_best_results.csv`, `grid_search_all_configs.csv`: Grid search optimization results

Metrics stored in directories:
- `distillation_metrics/`: Training history for each distilled model
- `training_metrics/`: Training history for standard models
- `pipeline_metrics/`: Additional pipeline evaluation metrics

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (recommended for GPU acceleration)

### Setup

1. Clone or download the project
2. Install dependencies:
```bash
pip install torch transformers sklearn pandas numpy matplotlib seaborn
```

3. (Optional) Analyze dataset token lengths:
```bash
python analyze_max_length.py --dataset dataset/multiclass_sentiments.jsonl
```

## Usage

### Configuration

Edit `grid_search_config.json` to customize:

**Models:**
```json
"models": [
  {"name": "BERT", "model_id": "bert-base-uncased"},
  {"name": "RoBERTa", "model_id": "roberta-base"},
  {"name": "DistilBERT", "model_id": "distilbert-base-uncased"}
]
```

**Feature Extraction:**
```json
"feature_extraction": {
  "batch_size": 64,
  "max_length": 256
}
```

**Model-Specific Training:**
```json
"model_config": {
  "BERT": {
    "epochs_options": [200],
    "batch_size_options": [128],
    "hidden_dims_options": [[512, 256, 128]],
    "dropout_options": [0.1],
    "learning_rate": 0.0003
  }
}
```

**Knowledge Distillation (Optional):**
```json
"distillation": {
  "enabled": true,
  "alpha": 0.5,
  "temperature": 3.0,
  "student_config": {
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    "learning_rate": 0.001,
    "epochs": 50,
    "patience": 10
  }
}
```

### Running the Pipeline

1. **Main Pipeline** - Extract features and train classifiers:
   - Open and run `pipeline.ipynb` in Jupyter
   - Extracts features from all configured models
   - Trains neural classifiers on extracted features
   - Saves results to CSV and metrics to directories

2. **Single Teacher** - Focus on one model:
   - Open and run `single_teacher_pipeline.ipynb`
   - Useful for debugging and detailed analysis of a specific model

3. **Grid Search** - Optimize hyperparameters:
   - Open and run `grid_search_pipeline.ipynb`
   - Searches across configuration space
   - Identifies best performing combinations

## Pipeline Flow

```
1. Load Configuration
   ↓
2. Load Dataset & Split (Train/Val/Test)
   ↓
3. Load Pre-trained Models
   ↓
4. Extract Features from Hidden Layers
   ↓
5. Train Neural Classifiers on Features
   ↓
6. Evaluate on Test Set
   ↓
7. (Optional) Knowledge Distillation
   ↓
8. Save Results & Metrics
```

## Output Files

After running the pipeline, you'll get:

**Results CSVs:**
- Model performance metrics (accuracy, precision, recall, F1)
- Cross-validation scores
- Training configurations

**Metrics CSVs:**
- Training/validation loss per epoch
- Model-specific learning curves

## Key Features

### Multi-Teacher Architecture
- Features extracted from three different transformer models
- Separate neural classifiers trained per extracted feature set
- Comparative analysis across model architectures

### Knowledge Distillation
- Optional student model compression
- Configurable distillation loss weight (alpha) and temperature
- Tracks teacher-student knowledge transfer

### Grid Search
- Comprehensive hyperparameter search
- Model-specific configuration options
- Cross-validation with configurable splits (default: 5-fold)

### Detailed Metrics
- Per-epoch training history
- Validation monitoring with early stopping
- Test accuracy, precision, recall, F1-score
- Gradient clipping and optimization tracking

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for reproducibility |
| `batch_size` | int | 64 | Batch size for feature extraction |
| `max_length` | int | 256 | Maximum token sequence length |
| `epochs` | int | 200 | Training epochs for classifier |
| `patience` | int | 10 | Early stopping patience |
| `learning_rate` | float | 0.0003 | Optimizer learning rate |
| `dropout` | float | 0.1 | Dropout rate in classifier |
| `gradient_clip_norm` | float | 1.0 | Gradient clipping value |
| `temperature` | float | 3.0 | Distillation temperature |
| `alpha` | float | 0.5 | Distillation loss weight |

## Model Architectures

### Extracted Models
- **BERT**: 768-dim embeddings, 12 hidden layers
- **RoBERTa**: 768-dim embeddings, 12 hidden layers
- **DistilBERT**: 768-dim embeddings, 6 hidden layers (distilled)

### Classification Network
Configurable MLP classifier on extracted features:
```
Input (feature_dim) → Hidden Layers → Output (num_classes)
```
Default: `input → 512 → 256 → 128 → num_classes`

## Performance Considerations

- **GPU Memory**: ~4-6GB for concurrent feature extraction on large batches
- **CPU-only**: Feasible but slower (~2-3x); recommend GPU for production
- **Dataset Size**: Tested on datasets with 5K-50K samples
- **Distillation**: Adds 20-30% computation time but reduces inference latency

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in feature extraction
- Reduce `hidden_dims` in classifier
- Process models sequentially instead of parallel

### Poor Model Performance
- Verify dataset is properly loaded with correct labels
- Check `max_length` covers 95%+ of sequences using `analyze_max_length.py`
- Increase `epochs` or reduce `patience` for longer training
- Adjust `learning_rate` and `dropout` based on validation metrics

### Missing Features in Distillation
- Ensure teacher and student hidden dimensions are compatible
- Check that teacher model is properly loaded before distillation
- Review distillation loss values in training history

## Directory Structure

```
knowledge_distillation_sentiment_classification/
├── pipeline.ipynb
├── single_teacher_pipeline.ipynb
├── analyze_max_length.py
├── grid_search_config.json
├── distillation_config.json
├── dataset/
├── distillation_metrics/
```

## Publication & Citation

This project is part of a CAPSTONE study. If you use this code in research, please cite accordingly.

## License

[Add appropriate license information]

## Contact

For questions or issues, please refer to the project documentation or contact the development team.

---

**Last Updated**: April 2026
