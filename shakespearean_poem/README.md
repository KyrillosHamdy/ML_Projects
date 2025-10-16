# Shakespearean Text Generator 🎭

A character-level LSTM neural network that generates text in the style of William Shakespeare. This project uses TensorFlow/Keras to train a recurrent neural network on Shakespeare's works and generate new text that mimics his distinctive writing style.

## 📖 Overview

This project implements a character-level text generation model using Long Short-Term Memory (LSTM) networks. The model learns patterns in Shakespeare's writing at the character level and can generate new text that follows similar linguistic patterns, vocabulary, and style.

## ✨ Features

- **Character-level text generation**: Learns patterns at the character level for fine-grained text control
- **LSTM architecture**: Uses 128 LSTM units for effective sequence modeling
- **Temperature sampling**: Implements temperature-based sampling for controllable creativity
- **Seed text priming**: Allows starting generation with custom seed phrases
- **Interactive generation**: User-friendly interface for generating text with custom parameters

## 🏗️ Model Architecture

```
Sequential Model:
├── LSTM Layer (128 units)
│   ├── Input Shape: (sequence_length, vocab_size)
│   ├── Return Sequences: False
│   └── Activation: tanh/sigmoid (default)
└── Dense Layer (vocab_size units)
    └── Activation: softmax
```

**Key Parameters:**
- Sequence Length: 40 characters
- Step Size: 3 characters
- Vocabulary Size: ~36 unique characters
- Batch Size: 256
- Training Epochs: 10
- Optimizer: RMSprop (learning_rate=0.01)

## 📊 Dataset

- **Source**: Shakespeare's complete works (~1.1M characters)
- **Subset Used**: 300,000 characters (positions 200,000-500,000)
- **Preprocessing**: 
  - Lowercased for consistency
  - Character-level tokenization
  - One-hot encoding for neural network input

## 🚀 Getting Started

### Prerequisites

```python
tensorflow>=2.0.0
numpy>=1.19.0
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd shakespearean-text-generator
```

2. Install required packages:
```bash
pip install tensorflow numpy
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Shakespearean_Poem.ipynb
```

### Usage

1. **Load and preprocess data**:
   - Downloads Shakespeare dataset automatically
   - Preprocesses text (lowercasing, tokenization)
   - Creates training sequences with one-hot encoding

2. **Train the model**:
   ```python
   model.fit(x, y, batch_size=256, epochs=10)
   ```

3. **Generate text**:
   ```python
   seed = "When shall we three meet again"
   generated_text = generate_character(desired_length=100, seed=seed)
   print(generated_text)
   ```

## 🎛️ Key Functions

### `sample(preds, temperature)`
Implements temperature-based sampling for character selection:
- **Low temperature (0.5-0.7)**: More conservative, predictable text
- **High temperature (1.0-1.5)**: More creative, diverse text

### `seed_preprocessing(seed)`
Converts input seed text to one-hot encoded format matching the model's requirements.

### `generate_character(desired_length, seed)`
Main text generation function that:
- Processes the seed text
- Primes the model's hidden state
- Generates characters iteratively
- Returns complete generated text

## 📈 Training Results

The model shows steady improvement in loss over 10 epochs:
- Epoch 1: Loss ~2.61
- Epoch 10: Loss ~1.29
- Training time: ~2-4 seconds per epoch on GPU

## 🎯 Example Output

**Input Seed**: "When shall we three meet again"
**Generated Text**: 
```
When shall we three meet again
and make with me wild for thy grace.

hastings:
thy hearts of but it all thee;
how that i with the
```

## 🔬 How It Works

1. **Character Encoding**: Each character is converted to a one-hot vector
2. **Sequence Creation**: Text is split into overlapping sequences of 40 characters
3. **LSTM Processing**: The model learns to predict the next character given a sequence
4. **Generation**: Starting with a seed, the model predicts subsequent characters iteratively
5. **Temperature Sampling**: Randomness is controlled via temperature parameter

## 🎨 Applications

This character-level approach is particularly useful for:
- **Creative writing assistance**: Generate Shakespeare-style poetry and prose
- **Educational tools**: Demonstrate neural text generation concepts
- **Style transfer**: Learn and mimic specific writing styles
- **Data augmentation**: Create synthetic text data for training other models

## ⚡ Performance Considerations

- **Memory Usage**: One-hot encoding requires significant memory for large vocabularies
- **Training Time**: Character-level models need more iterations than word-level models
- **Generation Speed**: Iterative character prediction can be slow for long texts

## 🚧 Limitations

- **Coherence**: Character-level generation may lack long-term semantic coherence
- **Context Window**: Limited to 40-character context window
- **Vocabulary**: Restricted to characters present in training data
- **Modern Prompts**: May not handle non-Shakespearean input seeds well

## 🔮 Future Improvements

- [ ] Implement beam search for better text quality
- [ ] Add attention mechanisms for longer context
- [ ] Experiment with different architectures (GRU, Transformer)
- [ ] Fine-tune on specific Shakespeare plays
- [ ] Add text quality metrics and evaluation
- [ ] Implement model checkpointing and resuming

## 📚 References

- [TensorFlow Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 👨‍💻 Author

Created as an educational project demonstrating character-level text generation with LSTM networks.

---

*"All the world's a stage, and all the men and women merely players"* - William Shakespeare