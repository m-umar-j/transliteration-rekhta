# transliteration-rekhta
This project focuses on developing and training a transformer-based NMT model for transliterating Urdu poetry into the Devanagari script.
## Project Overview
This project implements a neural machine translation (NMT) approach to transliterate Urdu poetry into Devanagari script. The model preserves the poetic nuances and diacritical marks essential for maintaining the original meter and pronunciation.
## Dataset
Dataset provided by Rekhta Foundation
- Size: 30,000 ghazals (~400,000 lines of Urdu poetry
- Format: JSONL format where each instance contains Urdu, Devnagri, and Roman form of a line.
[![image.png](https://i.postimg.cc/tg3RQZ5C/image.png)](https://postimg.cc/wtBYDBZS)
## Technical Implementation
### Preprocessing
The preprocessing pipeline includes:

- Custom tokenizers for both Urdu and Devanagari scripts

- NFC normalization to preserve diacritics (matras) that were lost with default NFD normalization

- Text cleaning and formatting for TensorFlow compatibility

### Model Architecture
A sample transformer architecture
[![A sample transformer architecture for Portuguese to English translation](https://i.postimg.cc/PfgVXG53/image.png)](https://postimg.cc/ThcJ04mg)
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
import tensorflow_text
```
## Usage 
Download the zipped folder of the model and extract all the contents.
### Load the Model
The translator directory should contain all the required model files
```python
model=tf.saved_model.load('translator')
```
### Performing Transliteration
```python
response = model('ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے').numpy().decode('utf-8')
print(response)
```
Output
```
हज़ारों ख़्वाहिशें ऐसी कि हर ख़्वाहिश पे दम निकले
```
To perform transliteration in a loop, modify and run the above code and save the response in an appropriate output file (text/csv, etc)
For example, consider a CSV file
```python
df_test=pd.read_csv(path-to-input-csv-file)
```
```python
translations=[]
for i in range(len(df_test)):
    input_sentence = df_test.iloc[i]['nastaaliq']
    # Add batch dimension and convert to tensor
    result = translator(tf.expand_dims(input_sentence, 0))
    # Convert tensor to string
    translated_text = result[0].numpy().decode('utf-8')
    # Update DataFrame
    translations.append(translated_text)
```
```python
df_test['predicted_devanagari_sentence'] = translations
df_test.to_csv('results.csv')
```
## Challenges and Solutions
### Diacritic Preservation:
- Issue: Default NFD normalization in TensorFlow caused loss of important diacritics

- Solution: Implemented NFC normalization to preserve these critical markers
### Computational Resources:

- Issue: Memory leaks in TensorFlow pipeline on Google Colab, which led to an increase in RAM consumption linearly during training.

- Solution: Migrated to Kaggle Notebooks with higher GPU and RAM allocation
## Future Improvements
- Increase transformer layers for potentially higher accuracy

- Expand the dataset to include more diverse Urdu poetry styles

- Optimize for deployment on lower-resource environments

## Acknowledgements
- https://www.rekhta.org
- https://www.tensorflow.org/text/guide/subwords_tokenizer
- https://www.tensorflow.org/text/tutorials/transformer
