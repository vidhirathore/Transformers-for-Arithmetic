# config.py
import torch

NUM_TRAIN_SAMPLES = 50000
NUM_VAL_SAMPLES = 5000
NUM_TEST_SAMPLES = 10000
MAX_DIGITS_TRAIN = 4
MAX_DIGITS_TEST = MAX_DIGITS_TRAIN
MAX_DIGITS_GENERALIZATION = MAX_DIGITS_TRAIN + 2
OPERATIONS = ['+', '-']

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
BASE_VOCAB = list("0123456789+-")
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
VOCAB = SPECIAL_TOKENS + BASE_VOCAB

stoi = {char: i for i, char in enumerate(VOCAB)}
itos = {i: char for i, char in enumerate(VOCAB)}
PAD_IDX = stoi[PAD_TOKEN]
SOS_IDX = stoi[SOS_TOKEN]
EOS_IDX = stoi[EOS_TOKEN]
VOCAB_SIZE = len(VOCAB)

# --- Model Configuration ---
D_MODEL = 128          # Embedding dimension
N_HEADS = 8            # Number of attention heads
N_LAYERS = 3           # Number of encoder/decoder layers
D_FF = 512             # Dimension of the feed-forward network
DROPOUT = 0.5          # Dropout rate
MAX_SEQ_LEN = 2 * MAX_DIGITS_GENERALIZATION + 3 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
NUM_EPOCHS = 20        # Adjust as needed based on convergence
CLIP_GRAD = 1.0        # Gradient clipping value

DATA_DIR = "data"
MODEL_SAVE_PATH = "models/transformer_arithmetic.pt"
TRAIN_DATA_PATH = f"{DATA_DIR}/train.json"
VAL_DATA_PATH = f"{DATA_DIR}/val.json"
TEST_DATA_PATH = f"{DATA_DIR}/test.json"
GEN_TEST_DATA_PATH = f"{DATA_DIR}/gen_test.json" 