# Sequence-to-Sequence Learning with Transformers for Arithmetic

This project implements an Encoder-Decoder Transformer model from scratch (using PyTorch's `nn.MultiheadAttention` but not `nn.TransformerEncoder/Decoder`) to perform basic arithmetic (addition and subtraction) on character sequences.

## Structure

*   `config.py`: Contains all hyperparameters, vocabulary definitions, file paths, and configuration settings.
*   `data_utils.py`: Handles synthetic data generation, vocabulary management, tokenization, PyTorch Dataset creation, and DataLoader setup.
*   `model.py`: Defines the Transformer architecture components: Positional Encoding, EncoderLayer, DecoderLayer, Encoder, Decoder, and the main Seq2SeqTransformer model.
*   `train.py`: Orchestrates the entire process: data generation (if needed), model building, training loop, evaluation, inference (greedy decoding), metric calculation (Exact Match, Char Accuracy, Perplexity), model saving, and basic analysis (test set performance, generalization test, sample error printing).
*   `report.pdf`: Contains the detailed analysis, results, discussion, ablation studies, etc.
*   `models/`: Directory where the trained model weights (`transformer_arithmetic.pt`) will be saved.
*   `data/`: Directory where the generated datasets (`train.json`, `val.json`, `test.json`, `gen_test.json`) will be saved.
*   `README.md`: This file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vidhirathore/Transformers-for-Arithmetic.git
    cd Transformers-for-Arithmetic
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch tqdm numpy
    # Ensure you install the correct PyTorch version for your system (CPU/GPU)
    # See: https://pytorch.org/get-started/locally/
    ```

## Running the Code

1.  **Generate Data, Train, and Evaluate:**
    Execute the main training script. It will automatically handle data generation if the `.json` files are not found in the `data/` directory.
    ```bash
    python train.py
    ```
    *   This script performs:
        *   Data generation (if necessary).
        *   Model initialization.
        *   Training loop over specified epochs, saving the best model based on validation loss to `models/transformer_arithmetic.pt`.
        *   Evaluation on the test set using the best saved model.
        *   Evaluation on the generalization test set (longer inputs).
        *   Prints final metrics (Loss, Perplexity, Exact Match Accuracy, Character-Level Accuracy).
        *   Prints a few examples of incorrect predictions from the test set for basic error analysis.

2.  ## Analysis (Part 4)

- The quantitative results and generalization performance were printed by `train.py`.
- Initial error analysis was conducted by examining the printed errors.
- Further analysis was carried out as follows:

  - **Deeper Error Analysis:** Errors were categorized and correlated with input features such as sequence length and the presence of carries or borrows. For this purpose, `train.py` was modified and an additional script/notebook (`analysis.ipynb`) was created to load the model and datasets.

  - **Ablation/Sensitivity Study:** Various hyperparameters and architectural choices in `config.py` and `model.py` were altered. `train.py` was re-run—often with fewer epochs or reduced data for faster experimentation—and the outcomes were compared against the baseline. The results of these experiments were documented in the report.

    
```
## File Structure Overview
├── config.py                        # Configuration and hyperparameters
├── data_utils.py                    # Data generation and loading utilities
├── model.py                         # Transformer model definition
├── train.py                         # Main script for training and evaluation
├── README.md                        # This file
├── report.pdf                       # (Student needs to create this report)
├── data/                            # Directory for generated datasets (created by script)
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── gen_test.json
└── models/                          # Directory for saved model weights (created by script)
    └── transformer_arithmetic.pt
```

The best trained model weights are saved as `models/transformer_arithmetic.pt`.
