# data_utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import json
import os
import config 
# --- Data Generation ---

def generate_arithmetic_problem(max_digits, operations):
    """Generates a single arithmetic problem string and its solution string."""
    num_digits1 = random.randint(1, max_digits)
    num_digits2 = random.randint(1, max_digits)
    num1 = random.randint(0, 10**num_digits1 - 1)
    num2 = random.randint(0, 10**num_digits2 - 1)
    op = random.choice(operations)

    input_str = f"{num1}{op}{num2}"

    try:
        result = eval(input_str) 
    except Exception as e:
        print(f"Error evaluating '{input_str}': {e}")
        return None, None 

    target_str = str(result)
    return input_str, target_str

def generate_dataset(num_samples, max_digits, operations, filename):
    """Generates a dataset and saves it to a JSON file."""
    dataset = []
    generated_count = 0
    while generated_count < num_samples:
        input_str, target_str = generate_arithmetic_problem(max_digits, operations)
        if input_str is not None:
            dataset.append({"input": input_str, "target": target_str})
            generated_count += 1
        if generated_count % 1000 == 0 and generated_count > 0:
             print(f"Generated {generated_count}/{num_samples} for {os.path.basename(filename)}")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(dataset, f)
    print(f"Dataset saved to {filename}")
    return dataset

# --- Tokenization and Vocabulary ---

def tokenize_sequence(sequence, stoi_map, add_sos=False, add_eos=False):
    """Converts a sequence string to a list of token indices."""
    tokens = list(sequence)
    indices = []
    if add_sos:
        indices.append(config.SOS_IDX)
    indices.extend([stoi_map.get(token, config.PAD_IDX) for token in tokens]) 
    if add_eos:
        indices.append(config.EOS_IDX)
    return torch.tensor(indices, dtype=torch.long)

# --- PyTorch Dataset ---

class ArithmeticDataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to the JSON file with data.
        """
        try:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            print("Please run the data generation part of train.py first.")
            self.data = [] 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_seq = item['input']
        target_seq = item['target']

        
        input_tokens = tokenize_sequence(input_seq, config.stoi)
        target_tokens = tokenize_sequence(target_seq, config.stoi, add_sos=True, add_eos=True)

        return input_tokens, target_tokens

# --- DataLoader Collator ---

def collate_fn(batch):
    """
    Pads sequences in a batch and returns tensors.
    Args:
        batch: A list of tuples (input_tokens, target_tokens).
    Returns:
        A tuple of tensors (src_batch, tgt_batch).
    """
    src_list, tgt_list = zip(*batch)

    src_padded = pad_sequence(src_list, batch_first=True, padding_value=config.PAD_IDX)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=config.PAD_IDX)

    return src_padded, tgt_padded

# --- Function to get DataLoaders ---

def get_dataloaders(batch_size):
    """Creates train, validation, and test DataLoaders."""
    train_dataset = ArithmeticDataset(config.TRAIN_DATA_PATH)
    val_dataset = ArithmeticDataset(config.VAL_DATA_PATH)
    test_dataset = ArithmeticDataset(config.TEST_DATA_PATH)
    gen_test_dataset = ArithmeticDataset(config.GEN_TEST_DATA_PATH) 

    if not train_dataset.data or not val_dataset.data or not test_dataset.data or not gen_test_dataset.data:
         print("One or more datasets are empty. Exiting DataLoader creation.")
         return None, None, None, None 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    gen_test_loader = DataLoader(gen_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, gen_test_loader

if __name__ == "__main__":
    print("Generating datasets...")
    generate_dataset(config.NUM_TRAIN_SAMPLES, config.MAX_DIGITS_TRAIN, config.OPERATIONS, config.TRAIN_DATA_PATH)
    generate_dataset(config.NUM_VAL_SAMPLES, config.MAX_DIGITS_TRAIN, config.OPERATIONS, config.VAL_DATA_PATH)
    generate_dataset(config.NUM_TEST_SAMPLES, config.MAX_DIGITS_TEST, config.OPERATIONS, config.TEST_DATA_PATH)
    generate_dataset(config.NUM_TEST_SAMPLES // 2, config.MAX_DIGITS_GENERALIZATION, config.OPERATIONS, config.GEN_TEST_DATA_PATH) # Smaller gen test set
    print("Data generation complete.")

    print("\nTesting DataLoader...")
    train_loader, _, _, _ = get_dataloaders(batch_size=4)
    if train_loader:
        src_batch, tgt_batch = next(iter(train_loader))
        print("Source Batch Shape:", src_batch.shape)
        print("Target Batch Shape:", tgt_batch.shape)
        print("Sample Source Sequence:", src_batch[0])
        print("Sample Target Sequence:", tgt_batch[0])
        print("Vocab Size:", config.VOCAB_SIZE)
        print("PAD Index:", config.PAD_IDX)
    else:
        print("DataLoader creation failed.")