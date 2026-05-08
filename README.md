# NLP Sentiment Analysis

A C++ desktop application for sentiment analysis using neural networks. Built with LibTorch, Qt6, and Eigen3.

## Features

- **Text Sentiment Classification**: Analyze text and classify as positive, negative, or neutral
- **Interactive GUI**: Qt-based interface with probability visualization
- **Custom Tokenizer**: Vocabulary-based tokenization (supports 2500 tokens)
- **Neural Network Model**: 3-layer linear network with LayerNorm
- **Training Pipeline**: Configurable training with adjustable epochs and learning rate

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | C++17 |
| Build System | CMake 3.20+ |
| ML Framework | LibTorch (PyTorch C++) |
| GUI Framework | Qt6 (Widgets, Charts, Concurrent) |
| Linear Algebra | Eigen3 |

## Project Structure

```
NLP_sentiment_analysis/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── pretrained_weights/     # Pre-trained model assets
│   ├── tokens.txt          # Vocabulary (2500 tokens)
│   ├── embeding_matrix.bin # Binary embedding matrix (125-dim)
│   └── linear_model.pt     # Trained PyTorch model
├── src/
│   ├── gui/                # Qt GUI components
│   │   ├── main.cpp        # Application entry point
│   │   ├── mainwindow.cpp  # Main window UI
│   │   ├── trainingsettingsdialog.cpp
│   │   └── probabilitychart/
│   └── core/               # Core NLP components
│       ├── structure.hpp   # Global structures & constants
│       ├── neural_network_functions/
│       ├── model/          # Sentiment model (train + pretrained)
│       ├── tokenizer/      # Custom tokenizer (train + pretrained)
│       └── embedding/      # Word embeddings (train + pretrained)
└── build/                   # Build output directory
```

## Architecture

### Model Architecture

The sentiment classifier is a 3-layer linear neural network:

```
Input (125-dim embeddings)
    → Layer 1 (Linear + LayerNorm)
    → Layer 2 (Linear + LayerNorm)
    → Layer 3 (Linear)
    → Output (3 classes: positive, negative, neutral)
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_TOKENS_AMOUNT` | 2500 | Maximum vocabulary size |
| `AMOUNT_ROWS` | 125 | Embedding dimension |
| `MAX_EPOCH_AMOUNT` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.0003 | Default learning rate |

## Build Instructions

### Prerequisites

Install dependencies (macOS with Homebrew):

```bash
# Qt6
brew install qt6

# Eigen3
brew install eigen

# LibTorch (download pre-built from PyTorch website)
# Extract to: /Users/vladbuyel/Documents/Libraries/C++/libtorch
```

### Build

```bash
cd NLP_sentiment_analysis
mkdir -p build && cd build
cmake ..
make
```

### Run

```bash
./SentimentAnalysis
```

## Usage

1. Launch the application
2. Enter text in the input field
3. Click "Analyze" to get sentiment prediction
4. View probability distribution in the chart

### Training

1. Open Training Settings dialog
2. Configure epochs and learning rate
3. Start training (runs in background thread)

## For Developers

### Adding New Features

1. **Model changes**: Edit `src/core/model/pretrained/model.cpp` and `model.hpp`
2. **Tokenizer modifications**: Edit `src/core/tokenizer/`
3. **GUI updates**: Qt UI files in `src/gui/`

### Key Files

- `src/core/structure.hpp` - Global constants and shared state
- `src/core/model/pretrained/model.cpp` - Model inference
- `src/core/embedding/pretrained/embedding.cpp` - Embedding lookup
- `src/core/tokenizer/pretrained/tokenizer.cpp` - Text tokenization

### Extending the Model

To change model architecture:

1. Modify `LinearModel` class in `model.hpp`
2. Update `forward()` in `model.cpp`
3. Retrain and export via LibTorch
4. Replace `linear_model.pt` in `pretrained_weights/`

## Acknowledgments

- [PyTorch](https://pytorch.org/) - ML framework
- [Qt](https://www.qt.io/) - GUI framework
- [Eigen](https://eigen.tuxfamily.org/) - Linear algebra library