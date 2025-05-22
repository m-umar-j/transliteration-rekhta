# transliteration-rekhta
This project focuses on developing and training a transformer-based NMT model for transliterating Urdu poetry into the Devanagari script.
## Project Overview
This project implements a neural machine translation (NMT) approach to transliterate Urdu poetry, into Devanagari script. The model preserves the poetic nuances and diacritical marks essential for maintaining the original meter and pronunciation.
## Dataset
Dataset provided by Rekhta Foundation
- Size: 30,000 ghazals (~400,000 lines of Urdu poetry
- Format: JSONL format where each instance contains Urdu, Devnagri, and Roman form of a line.

## Technical Implementation
### Preprocessing
The preprocessing pipeline includes:

- Custom tokenizers for both Urdu and Devanagari scripts

- NFC normalization to preserve diacritics (matras) that were lost with default NFD normalization

- Text cleaning and formatting for TensorFlow compatibility

### Model Architecture
[![image.png](https://i.postimg.cc/PfgVXG53/image.png)](https://postimg.cc/ThcJ04mg)
- Framework: TensorFlow

- Architecture: Transformer-based model

- Depth: 3 layers (configurable for future retraining)

- Custom learning rate scheduler for optimization
### Training
- Epochs: 10 (convergence observed around 5 epochs)

Performance:

- Training accuracy: ~97%

- Validation accuracy: ~96%

- Hardware: Trained on Kaggle Notebooks with 2×T4 GPUs and 28GB RAM
## Installation and Setup
Install tensorflow and tensorflow_text
```python
pip install tensorflow tensorflow_text
```
Import installed packages
```python
import tensorflow as tf
import tensorflow_datasets as tfds
```
## Usage 
### Load the Model
```

```
### Performing Transliteration
```python

```

## Challenges and Solutions
### Diacritic Preservation:
- Issue: Default NFD normalization in TensorFlow caused loss of important diacritics

- Solution: Implemented NFC normalization to preserve these critical markers
### Computational Resources:

- Issue: Memory leaks in TensorFlow pipeline on Google Colab

- Solution: Migrated to Kaggle Notebooks with higher GPU and RAM allocation
## Future Improvements
- Increase transformer layers for potentially higher accuracy

- Expand the dataset to include more diverse Urdu poetry styles

- Optimize for deployment on lower-resource environments

## Acknowledgements
- rekhta.org
- https://www.tensorflow.org/text/guide/subwords_tokenizer
- https://www.tensorflow.org/text/tutorials/transformer
