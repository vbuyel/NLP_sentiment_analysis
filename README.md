# NLP Sentiment Analysis

A C++ desktop application for sentiment analysis using neural networks. Built with LibTorch, Qt6, and Eigen3.

## Features

- **Text Sentiment Classification**: Analyze text and classify as positive, negative, or neutral
- **Interactive GUI**: Qt-based interface with probability visualization
- **Custom Tokenizer**: BPE-based tokenization with vocabulary up to 2500 tokens
- **Neural Network Model**: 3-layer feed-forward network (125→12000→125→3) with LayerNorm and ReLU
- **Training Pipeline**: Configurable training with adjustable epochs and learning rate, runs in background thread

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
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── pretrained_weights/         # Pre-trained model assets
│   ├── tokens.txt              # Vocabulary (2500 tokens)
│   ├── embeding_matrix.bin     # Binary embedding matrix (125-dim)
│   └── linear_model.pt         # Trained PyTorch model (LibTorch)
├── src/
│   ├── gui/                    # Qt GUI components
│   │   ├── main.cpp            # Application entry point
│   │   ├── mainwindow.cpp      # Main window UI logic
│   │   ├── mainwindow.hpp      # Main window header
│   │   ├── trainingsettingsdialog.cpp
│   │   ├── trainingsettingsdialog.hpp
│   │   └── probabilitychart/   # Bar chart visualization
│   │       ├── probabilitychart.cpp
│   │       └── probabilitychart.hpp
│   └── core/                   # Core NLP components
│       ├── main_types.hpp      # Shared type aliases (KEY, VALUE)
│       ├── structure.hpp       # Global constants & extern declarations
│       ├── structure.cpp       # Global variable definitions
│       ├── neural_network_functions/
│       │   ├── neural_network.hpp   # BackProp struct
│       │   ├── neural_network.cpp   # FC layer, ReLU, cross-entropy
│       │   └── math_functions.cpp   # Softmax, batch_norm, stats
│       ├── model/
│       │   ├── pretrained/     # Inference with LibTorch
│       │   │   ├── model.hpp
│       │   │   └── model.cpp
│       │   └── train/          # Training loop (Adam optimizer)
│       │       └── model_train_main.cpp
│       ├── tokenizer/
│       │   ├── tokenizer_globals.hpp
│       │   ├── pretrained/     # Inference tokenizer
│       │   │   ├── tokenizer.hpp
│       │   │   └── tokenizer.cpp
│       │   └── train/          # BPE training
│       │       ├── tokenize_train.hpp
│       │       ├── tokenize_train.cpp
│       │       └── tokenizer_main.cpp
│       └── embedding/
│           ├── pretrained/     # Embedding lookup
│           │   ├── embedding.hpp
│           │   └── embedding.cpp
│           └── train/          # Embedding matrix training
│               ├── embedding_structure.hpp
│               ├── embedding_main.cpp
│               └── matrix_train/
│                   ├── main_train.cpp
│                   ├── main_train.hpp
│                   ├── train_to_predict.cpp
│                   └── train_to_predict.hpp
└── build/                       # Build output directory
```

## Architecture

### Model Architecture

The sentiment classifier is a 3-layer feed-forward neural network built with LibTorch:

```
Input text
    → Tokenizer (BPE)
    → Embedding lookup (tokens → 125-dim vectors)
    → Mean pooling across all token vectors
    → Layer 1 (Linear: 125 → 12000 + LayerNorm + ReLU)
    → Layer 2 (Linear: 12000 → 125   + LayerNorm + ReLU)
    → Layer 3 (Linear: 125 → 3)
    → Softmax
    → Output: [Negative, Neutral, Positive] probabilities
```

**Note:** The first hidden layer expands to **12000 dimensions** — significantly wider than the embedding dimension — giving the model high capacity for learning complex sentiment patterns.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_TOKENS_AMOUNT` | 2500 | Maximum vocabulary size |
| `AMOUNT_ROWS` | 125 | Embedding vector dimension |
| `FIRST_HIDEN_LAYER` | 12000 | Width of first hidden layer (125 → 12000) |
| `MAX_EPOCH_AMOUNT` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.0003 | Default learning rate (Adam optimizer) |

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
cmake -B build -S .
cmake --build build
```

### Run

```bash
cd build
./SentimentAnalysis
```

> The binary expects `pretrained_weights/` at `../` relative to the working directory.

## Usage

1. Launch the application
2. Enter text in the input field
3. Click "Analyze" to get sentiment prediction
4. View probability distribution in the chart

### Training

1. Click **"Choose dataset for training Tokenizer and Embedding"** — select a `.txt` dataset
2. Click **"Choose dataset for training Model"** — select a semicolon-separated dataset (text;label)
3. Click **"Training Settings"** — configure:
   - **Max tokens** per input sequence (default: 128)
   - **Train tokenizer & embedding** checkbox
   - **Train model** checkbox
4. Click **"Start Training"** — runs tokenizer/embedding training first, then model fine-tuning (background thread via `QtConcurrent`)

## For Developers

### Adding New Features

1. **Model changes**: Edit `src/core/model/pretrained/model.cpp` and `model.hpp`
2. **Tokenizer modifications**: Edit `src/core/tokenizer/`
3. **GUI updates**: Qt UI files in `src/gui/`

### Key Files

- `src/core/structure.hpp` - Global constants and extern declarations
- `src/core/structure.cpp` - Global variable definitions
- `src/core/model/pretrained/model.cpp` - Model inference (`predict()`)
- `src/core/model/train/model_train_main.cpp` - Model training loop (`fit()`, `load()`)
- `src/core/embedding/pretrained/embedding.cpp` - Embedding lookup
- `src/core/tokenizer/pretrained/tokenizer.cpp` - BPE text tokenization
- `src/core/neural_network_functions/neural_network.cpp` - FC layer, ReLU, cross-entropy
- `src/core/neural_network_functions/math_functions.cpp` - Softmax, batch norm, stats

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