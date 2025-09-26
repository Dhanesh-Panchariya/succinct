import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Succinct.logging import logger
from Succinct.entity import DataTransformationConfig

class SummDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=1024, max_output_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        inputs = self.tokenizer(
            example["dialogue"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.tokenizer(
            example["summary"],
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": outputs.input_ids.squeeze(0)
        }

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokeniser_name)

    def load_data(self):
        """Load JSON dataset splits from the data path"""
        data_path = self.config.data_path
        datasets = {}
        for split in ["train", "val", "test"]:
            file_path = os.path.join(data_path, f"{split}.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    datasets[split] = json.load(f)
            else:
                logger.warning(f"{split}.json not found in {data_path}. Skipping this split.")
        return datasets

    def convert(self):
        """Tokenize and convert datasets to PyTorch format, then save as .pt"""
        datasets_dict = self.load_data()
        datasets_pt = {
            split: SummDataset(data, self.tokenizer)
            for split, data in datasets_dict.items()
        }

        save_path = os.path.join(self.config.root_dir, "samsum_dataset.pt")
        torch.save(datasets_pt, save_path)
        logger.info(f"Processed dataset saved to {save_path}")

        return datasets_pt