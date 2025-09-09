## Dataset class
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from datasets import load_dataset

class MultiTokenPredictionDataset(Dataset):
    def __init__(self, ds, tokenizer, config):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.num_mask = config.num_masks

        # Use pad_token_id if available, otherwise define one
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.pad_token_id = self.tokenizer.pad_token_id

        self.mask_tokens = [f"<mask_{i}>" for i in range(self.num_mask)]
        self.mask_token_ids = self.tokenizer.convert_tokens_to_ids(self.mask_tokens)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        assistant_token_ids = self.tokenizer.encode('<|assistant|>\n', add_special_tokens=False)

        for attempt in range(10):
            try:
                current_idx = (idx + attempt) % len(self.ds)
                item = self.ds[current_idx]
                messages = item['messages']
                tokenized_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False
                )
                position = self._find_subsequence(tokenized_input, assistant_token_ids)
                user_prompt_label_mask_len = len(tokenized_input[:position]) * (self.num_mask +1)

                input_id, position_id, labels, mtp_mask = self._create_masked_input(tokenized_input)
                seq_len = input_id.shape[0]
                actual_length = len(tokenized_input)
                labels[:user_prompt_label_mask_len] = -100
                valid_labels = (labels != -100).sum().item()
                if valid_labels == 0:
                    print(f"Skipping example {current_idx}: no valid labels")
                    continue
                return {
                    'input_ids': input_id,
                    'mtp_position_ids': position_id,
                    'mtp_mask': mtp_mask,
                    'labels': labels,
                }
            except:
                continue
        # Fallback item if all attempts fail
        fallback_text = "<|user|>\nHello<|assistant|>\nHi<|endoftext|>"
        tokenized_input = self.tokenizer.encode(fallback_text, add_special_tokens=False)
        input_id, position_id, labels, mtp_mask = self._create_masked_input(tokenized_input)

        return {
            'input_ids': input_id,
            'mtp_position_ids': position_id,
            'mtp_mask': mtp_mask,
            'labels': labels,
        }

    def _find_subsequence(self, sequence, target):
        for i in range(len(sequence) - len(target) + 1):
            if sequence[i:i + len(target)] == target:
                return i
        return -1

    def _create_attention_bias_matrix(self, mtp_mask, seq_len, actual_length):

        # Initialize bias matrix - start with all positions blocked (-inf)
        bias_matrix = torch.full((seq_len, seq_len), -1e4, dtype=torch.float16)

        # Convert mask to tensor for easier indexing
        if isinstance(mtp_mask, torch.Tensor):
            mtp_mask = mtp_mask[:actual_length].clone().detach()
        else:
            mtp_mask = torch.tensor(mtp_mask[:actual_length], dtype=torch.bool)

        ntp_mask = ~mtp_mask  # NTP mask is opposite of MTP mask

        # Identify MTP blocks (consecutive sequences of MTP tokens)
        mtp_blocks = []
        if mtp_mask.any():
            mtp_positions = torch.where(mtp_mask)[0]

            # Group consecutive MTP positions into blocks
            current_block = [mtp_positions[0].item()]
            for i in range(1, len(mtp_positions)):
                if mtp_positions[i] == mtp_positions[i-1] + 1:
                    current_block.append(mtp_positions[i].item())
                else:
                    mtp_blocks.append(current_block)
                    current_block = [mtp_positions[i].item()]
            mtp_blocks.append(current_block)

        # Create block diagonal structure
        for i in range(min(actual_length, len(mtp_mask))):

            if ntp_mask[i]:  # Current token is NTP
                # NTP tokens attend only to previous NTP tokens (causal attention)
                for j in range(i + 1):  # j <= i (causal)
                    if j < len(ntp_mask) and ntp_mask[j]:
                        bias_matrix[i, j] = 0.0

            else:  # Current token is MTP
                # Find which MTP block this token belongs to
                current_block = None
                for block in mtp_blocks:
                    if i in block:
                        current_block = block
                        break

                if current_block is not None:
                    # MTP token attends to:
                    # 1. All previous NTP tokens
                    for j in range(min(actual_length, len(ntp_mask))):
                        if j < i and ntp_mask[j]:  # Previous NTP tokens only
                            bias_matrix[i, j] = 0.0

                    # 2. All tokens in the same MTP block (but only previous ones + self - causal)
                    for j in current_block:
                        if j < actual_length and j < len(ntp_mask) and j <= i:  # Causal: j <= i
                            bias_matrix[i, j] = 0.0

        return bias_matrix

    def _create_masked_input(self, sequence):
        input_id = []
        mtp_position_id = []
        labels = []
        mtp_mask = []

        ## Find the user prompt up till assistant prompt. Make labels -100 for user

        for i in range(len(sequence)):
            # Add original token
            input_id.append(sequence[i])

            # Add mask tokens
            input_id += self.mask_token_ids

            # Position IDs: original token gets i, masks get i+1..i+num_mask
            mtp_position_id.extend([i] + [i+1+j for j in range(self.num_mask)])

            # Labels: original token predicts next token, masks predict future tokens
            labels.append(sequence[i+1] if i+1 < len(sequence) else -100)
            for j in range(self.num_mask):
                future_idx = i + 1 + j + 1
                if future_idx < len(sequence):
                    labels.append(sequence[future_idx])
                else:
                    labels.append(-100)

            # MTP mask: original token = False, mask tokens = True
            mtp_mask.extend([0] + [1] * self.num_mask)

        # Truncate
        input_id = input_id[:self.max_length]
        mtp_position_id = mtp_position_id[:self.max_length]
        labels = labels[:self.max_length]
        mtp_mask = mtp_mask[:self.max_length]

        # Pad to max_length
        pad_len = self.max_length - len(input_id)
        if pad_len > 0:
            input_id += [self.pad_token_id] * pad_len
            labels += [-100] * pad_len
            mtp_mask += [False] * pad_len

            # Position ids: monotonic 0..max_length-1
            mtp_position_id += list(range(len(mtp_position_id), self.max_length))

        input_id = torch.tensor(input_id, dtype=torch.long)
        mtp_position_id = torch.tensor(mtp_position_id, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        mtp_mask = torch.tensor(mtp_mask, dtype=torch.float)

        return input_id, mtp_position_id, labels, mtp_mask