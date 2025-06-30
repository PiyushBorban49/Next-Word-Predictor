# LSTM Text Generation for News Articles

A deep learning project implementing Long Short-Term Memory (LSTM) networks for text generation using TensorFlow/Keras. This model is trained on news articles to generate coherent text sequences.

## Table of Contents
- [Overview](#overview)
- [LSTM Theory and Background](#lstm-theory-and-background)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Research Papers](#research-papers)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)

## Overview

This project implements an LSTM-based text generation model that learns to predict the next word in a sequence given the previous 20 words. The model is trained on news articles and can generate new text by predicting one word at a time in an autoregressive manner.

## LSTM Theory and Background

### What are LSTMs?

Long Short-Term Memory (LSTM) networks are a special kind of Recurrent Neural Network (RNN) designed to overcome the vanishing gradient problem that traditional RNNs face when processing long sequences. LSTMs are particularly effective for sequential data like text, speech, and time series.

### Key Components of LSTM

1. **Cell State (C_t)**: The long-term memory that flows through the network
2. **Hidden State (h_t)**: The short-term memory/output at each timestep
3. **Three Gates**:
   - **Forget Gate**: Decides what information to discard from cell state
   - **Input Gate**: Decides what new information to store in cell state
   - **Output Gate**: Controls what parts of cell state to output

### Mathematical Formulation

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t * tanh(C_t)  # Hidden state
```

Where σ is the sigmoid function, and W, b are learned parameters.

### Why LSTMs for Text Generation?

- **Long-term Dependencies**: Can remember context from much earlier in the sequence
- **Selective Memory**: Gates allow the model to selectively remember or forget information
- **Gradient Flow**: Solves vanishing gradient problem through the cell state pathway
- **Sequential Processing**: Natural fit for text where word order matters

## Dataset

The model uses the "Fake or Real News" dataset containing news articles. For this implementation:
- **Total samples**: Limited to 1000 articles for demonstration
- **Preprocessing**: Text cleaning, lowercasing, URL/HTML removal
- **Sequence Length**: 20 words input, 1 word output
- **Vocabulary Size**: 5000 most frequent words

## Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 20, 100)          vocab_size * 100
_________________________________________________________________
lstm (LSTM)                  (None, 150)               150,600
_________________________________________________________________
dense (Dense)                (None, vocab_size)        vocab_size * 151
=================================================================
```

### Layer Details:

1. **Embedding Layer**: 
   - Converts word indices to dense vectors of size 100
   - Trainable word representations
   
2. **LSTM Layer**: 
   - 150 hidden units
   - Processes sequences and captures temporal dependencies
   
3. **Dense Layer**: 
   - Softmax activation for probability distribution over vocabulary
   - Outputs next word probabilities

## Implementation Details

### Text Preprocessing
- Convert to lowercase
- Remove URLs, HTML tags, punctuation, and digits
- Tokenization using Keras Tokenizer
- Sequence padding/truncation

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: Default (32)
- **Epochs**: 5
- **Train/Test Split**: 80/20

### Text Generation Strategy
- **Autoregressive Generation**: Generate one word at a time
- **Context Window**: Use last 20 words as input
- **Prediction**: Select word with highest probability (greedy decoding)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lstm-text-generation.git
cd lstm-text-generation

# Install required packages
pip install tensorflow pandas numpy matplotlib scikit-learn

# Download the dataset
# Place 'fake_or_real_news.csv' in the project directory
```

## Usage

```python
# Run the complete pipeline
python lstm_text_generation.py

# For custom text generation
seed_text = "the government announced"
generated = generate_text(seed_text, 15, model, tokenizer)
print(generated)
```

### Code Fix Note
There's a syntax error in the original `generate_text` function. Replace:
```python
for * in range(next*words):
```
with:
```python
for _ in range(next_words):
```

## Results

The model generates coherent short text sequences. Training results show:
- Loss decreases over epochs
- Validation loss tracks training loss
- Generated text maintains grammatical structure
- Vocabulary usage reflects training data distribution

Sample output:
```
Input: "the news report"
Generated: "the news report said the government has been working on the new policy"
```

## Research Papers

### Foundational Papers

1. **Hochreiter, S., & Schmidhuber, J. (1997)**
   - *Long Short-Term Memory*
   - Neural Computation, 9(8), 1735-1780
   - [Paper Link](https://www.bioinf.jku.at/publications/older/2604.pdf)
   - **Contribution**: Original LSTM architecture, solving vanishing gradient problem

2. **Gers, F. A., Schmidhuber, J., & Cummins, F. (1999)**
   - *Learning to forget: Continual prediction with LSTM*
   - Neural Computation, 12(10), 2451-2471
   - **Contribution**: Introduction of forget gate, improving LSTM performance

3. **Graves, A. (2013)**
   - *Generating sequences with recurrent neural networks*
   - arXiv preprint arXiv:1308.0850
   - [Paper Link](https://arxiv.org/abs/1308.0850)
   - **Contribution**: LSTM for text generation, character-level modeling

### Modern Developments

4. **Sutskever, I., Vinyals, O., & Le, Q. V. (2014)**
   - *Sequence to sequence learning with neural networks*
   - Advances in Neural Information Processing Systems, 27
   - [Paper Link](https://arxiv.org/abs/1409.3215)
   - **Contribution**: Encoder-decoder architecture with LSTMs

5. **Karpathy, A., Johnson, J., & Fei-Fei, L. (2015)**
   - *Visualizing and understanding recurrent networks*
   - arXiv preprint arXiv:1506.02078
   - [Paper Link](https://arxiv.org/abs/1506.02078)
   - **Contribution**: Analysis of what RNNs/LSTMs learn

6. **Merity, S., Keskar, N. S., & Socher, R. (2017)**
   - *Regularizing and optimizing LSTM language models*
   - arXiv preprint arXiv:1708.02182
   - [Paper Link](https://arxiv.org/abs/1708.02182)
   - **Contribution**: Advanced regularization techniques for LSTM language models

## Code Explanation

### Data Preprocessing Pipeline
```python
def preprocess_text(text):
    text = text.lower()  # Normalize case
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    return ' '.join(text.split())  # Normalize whitespace
```

### Sequence Generation
```python
# Create input-output pairs
for sequence in sequences:
    for i in range(20, len(sequence)):
        input_seq = sequence[i - 20:i + 1]  # 20 words input + 1 target
        input_sequences.append(input_seq)
```

### Model Architecture
```python
model = Sequential([
    Embedding(vocab_size, 100, input_length=20),  # Word embeddings
    LSTM(150),  # Recurrent layer with 150 units
    Dense(vocab_size, activation='softmax')  # Output layer
])
```

### Text Generation Process
1. **Input Preparation**: Convert seed text to token sequence
2. **Padding**: Ensure input length matches training format
3. **Prediction**: Get probability distribution over vocabulary
4. **Selection**: Choose most likely next word
5. **Update**: Add predicted word to context and repeat

## Model Limitations

- **Limited Context**: Only considers last 20 words
- **Greedy Decoding**: May produce repetitive text
- **Vocabulary Constraints**: Limited to 5000 most frequent words
- **Training Data**: Small dataset may limit generalization

## Future Improvements

1. **Attention Mechanisms**: Add attention to focus on relevant context
2. **Beam Search**: Implement beam search for better text generation
3. **Larger Datasets**: Train on more diverse and larger corpora
4. **Transformer Architecture**: Compare with modern transformer models
5. **Fine-tuning**: Domain-specific fine-tuning for different text types

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original LSTM paper by Hochreiter & Schmidhuber
- TensorFlow/Keras team for the deep learning framework
- News dataset contributors
- Research community for advancing sequence modeling
