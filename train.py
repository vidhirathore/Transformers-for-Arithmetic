# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
import os
import numpy as np
from tqdm import tqdm 
import config
import data_utils
import model

# --- Helper Functions ---

def calculate_loss(output, target, criterion, pad_idx):
    """Calculates loss, ignoring padding and SOS token."""
    
    output_flat = output.reshape(-1, output.shape[-1]) 
    target_flat = target[:, 1:].reshape(-1)                 

    loss = criterion(output_flat, target_flat)
    return loss

def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    """Performs one training epoch."""
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        decoder_input = tgt[:, :-1]

        output = model(src, decoder_input) 
        loss = calculate_loss(output, tgt, criterion, config.PAD_IDX)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):
            src, tgt = src.to(device), tgt.to(device)

            decoder_input = tgt[:, :-1]
            output = model(src, decoder_input)

            loss = calculate_loss(output, tgt, criterion, config.PAD_IDX)
            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def greedy_decode(model, src, max_len, sos_idx, eos_idx, device):
    """Generates output sequence using greedy decoding."""
    model.eval()
    src = src.to(device)
    batch_size = src.shape[0]

    with torch.no_grad():
        src_key_padding_mask = model._create_padding_mask(src, config.PAD_IDX)
        memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)
        memory_key_padding_mask = src_key_padding_mask 
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        for i in range(max_len - 1):
            tgt_key_padding_mask = model._create_padding_mask(ys, config.PAD_IDX) 
            tgt_mask = model._create_look_ahead_mask(ys.size(1), device)

            out = model.decoder(ys, memory,
                                tgt_mask=tgt_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask)
            
            prob = model.fc_out(out[:, -1]) # Shape: [batch_size, vocab_size]
            _, next_word = torch.max(prob, dim=1) # Shape: [batch_size]
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1) 

            # Early stop check (optional but good): Check if all sequences ended
            # if (next_word == eos_idx).all():
            #     break

    return ys[:, 1:]


def calculate_metrics(model, dataloader, device, max_len=config.MAX_SEQ_LEN):
    """Calculates Exact Match Accuracy and Character-Level Accuracy."""
    model.eval()
    total_samples = 0
    exact_matches = 0
    total_chars = 0
    correct_chars = 0

    all_preds = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Calculating Metrics", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            generated_ids = greedy_decode(model, src, max_len, config.SOS_IDX, config.EOS_IDX, device)
            
            target_ids_no_special = [
                [idx for idx in t if idx not in (config.SOS_IDX, config.EOS_IDX, config.PAD_IDX)]
                for t in tgt.cpu().tolist()
            ]
            target_strings = ["".join([config.itos[i] for i in ids]) for ids in target_ids_no_special]

            predicted_strings = []
            for gen_seq in generated_ids.cpu().tolist():
                pred_str = ""
                for idx in gen_seq:
                    if idx == config.EOS_IDX:
                        break
                    if idx != config.PAD_IDX:
                        pred_str += config.itos.get(idx, '?')
                predicted_strings.append(pred_str)

            input_ids_no_special = [
                [idx for idx in s if idx != config.PAD_IDX]
                for s in src.cpu().tolist()
            ]
            input_strings = ["".join([config.itos[i] for i in ids]) for ids in input_ids_no_special]

            all_preds.extend(predicted_strings)
            all_targets.extend(target_strings)
            all_inputs.extend(input_strings)

            for pred_str, target_str in zip(predicted_strings, target_strings):
                total_samples += 1
                if pred_str == target_str:
                    exact_matches += 1

                len_pred = len(pred_str)
                len_target = len(target_str)
                for i in range(min(len_pred, len_target)):
                    if pred_str[i] == target_str[i]:
                        correct_chars += 1
                total_chars += len_target 

    exact_match_acc = exact_matches / total_samples if total_samples > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0

    return exact_match_acc, char_acc, all_preds, all_targets, all_inputs

# --- Main Execution ---

if __name__ == "__main__":
    print("Using device:", config.DEVICE)

    # --- 1. Data Handling ---
    print("\n--- Part 1: Data Handling ---")
    if not all(os.path.exists(p) for p in [config.TRAIN_DATA_PATH, config.VAL_DATA_PATH, config.TEST_DATA_PATH, config.GEN_TEST_DATA_PATH]):
        print("Data files not found. Generating datasets...")
        data_utils.generate_dataset(config.NUM_TRAIN_SAMPLES, config.MAX_DIGITS_TRAIN, config.OPERATIONS, config.TRAIN_DATA_PATH)
        data_utils.generate_dataset(config.NUM_VAL_SAMPLES, config.MAX_DIGITS_TRAIN, config.OPERATIONS, config.VAL_DATA_PATH)
        data_utils.generate_dataset(config.NUM_TEST_SAMPLES, config.MAX_DIGITS_TEST, config.OPERATIONS, config.TEST_DATA_PATH)
        data_utils.generate_dataset(config.NUM_TEST_SAMPLES // 2, config.MAX_DIGITS_GENERALIZATION, config.OPERATIONS, config.GEN_TEST_DATA_PATH)
        print("Data generation complete.")
    else:
        print("Data files found.")

    print("Loading dataloaders...")
    train_loader, val_loader, test_loader, gen_test_loader = data_utils.get_dataloaders(config.BATCH_SIZE)
    if train_loader is None:
         print("Failed to create dataloaders. Exiting.")
         exit()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}, Gen Test batches: {len(gen_test_loader)}")

    # --- 2. Model Implementation ---
    print("\n--- Part 2: Model Implementation ---")
    transformer_model = model.build_model()
    print(f"Model created with {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad):,} trainable parameters.")
    print("Hyperparameters:")
    print(f"  d_model: {config.D_MODEL}, n_heads: {config.N_HEADS}, n_layers: {config.N_LAYERS}")
    print(f"  d_ff: {config.D_FF}, dropout: {config.DROPOUT}, max_seq_len: {config.MAX_SEQ_LEN}")

    # --- 3. Training ---
    print("\n--- Part 3: Training ---")
    optimizer = optim.Adam(transformer_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5 

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(transformer_model, train_loader, optimizer, criterion, config.CLIP_GRAD, config.DEVICE)
        val_loss, val_perplexity = evaluate(transformer_model, val_loader, criterion, config.DEVICE)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} | Val. Perplexity: {val_perplexity:.3f}')

        # --- Model Saving ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(transformer_model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"\tBest validation loss improved. Saved model to {config.MODEL_SAVE_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"\tValidation loss did not improve for {epochs_no_improve} epoch(s).")

        # Early stopping (optional)
        # if epochs_no_improve >= patience:
        #     print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
        #     break

    # --- 4. Analysis and Report ---
    print("\n--- Part 4: Analysis and Report ---")
    print("Loading best saved model...")
    try:
        transformer_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}. Evaluating with the last epoch's model.")
    except Exception as e:
         print(f"Error loading model state_dict: {e}. Evaluating with the last epoch's model.")


    print("\nEvaluating on Test Set...")
    test_loss, test_perplexity = evaluate(transformer_model, test_loader, criterion, config.DEVICE)
    test_exact_match, test_char_acc, _, _, _ = calculate_metrics(transformer_model, test_loader, config.DEVICE)

    print(f'\nTest Set Performance:')
    print(f'\tLoss: {test_loss:.3f}')
    print(f'\tPerplexity: {test_perplexity:.3f}')
    print(f'\tExact Match Accuracy: {test_exact_match:.4f}')
    print(f'\tCharacter-Level Accuracy: {test_char_acc:.4f}')

    # --- Generalization Test ---
    print("\nEvaluating on Generalization Test Set (Longer Inputs)...")
    gen_test_loss, gen_test_perplexity = evaluate(transformer_model, gen_test_loader, criterion, config.DEVICE)
    gen_test_exact_match, gen_test_char_acc, _, _, _ = calculate_metrics(transformer_model, gen_test_loader, config.DEVICE)

    print(f'\nGeneralization Test Set Performance:')
    print(f'\tLoss: {gen_test_loss:.3f}')
    print(f'\tPerplexity: {gen_test_perplexity:.3f}')
    print(f'\tExact Match Accuracy: {gen_test_exact_match:.4f}')
    print(f'\tCharacter-Level Accuracy: {gen_test_char_acc:.4f}')

    # --- Error Analysis (Example: Print some incorrect predictions) ---
    print("\nPerforming Error Analysis on Test Set (showing first 10 errors)...")
    _, _, test_preds, test_targets, test_inputs = calculate_metrics(transformer_model, test_loader, config.DEVICE)
    errors_found = 0
    print("Input -> Predicted | Target")
    print("---------------------------")
    for input_str, pred_str, target_str in zip(test_inputs, test_preds, test_targets):
        if pred_str != target_str and errors_found < 10:
            print(f"{input_str} -> {pred_str} | {target_str}")
            errors_found += 1
    if errors_found == 0:
        print("No errors found in the first few samples examined.")

    print("\nFurther analysis (Ablation studies, detailed error categorization) should be performed and included in the report.pdf")