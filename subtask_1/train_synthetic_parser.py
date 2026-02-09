#!/usr/bin/env python3
"""
Train a Synthetic Syllogism Parser (T5-small)
=============================================
This script:
1. Generates synthetic training data using nonsense words (0% content bias).
2. Fine-tunes T5-small to map NL Syllogisms -> "Mood: XYZ, Figure: N"
3. Saves the model for the symbolic engine.

Usage:
    python train_synthetic_parser.py --samples 50000 --epochs 3
"""

import os
import random
import torch
import argparse
import logging
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. SYNTHETIC DATA GENERATOR
# ============================================================================

NONSENSE_NOUNS = [
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"
]

TEMPLATES = {
    "A": [
        "All {S} are {P}", "Every {S} is a {P}", "Each {S} is a {P}", 
        "Any {S} is a {P}", "Everything that is a {S} is a {P}", 
        "All things which are {S} are {P}", "The entire class of {S} is composed of {P}",
        "Every single {S} is a {P}", "Anything that can be called a {S} is, without exception, a {P}",
        "{S} are, by definition, {P}", "Invariably, {S} are {P}", "It is universally true that {S} are {P}",
        "Any entity identified as {S} is necessarily a {P}", "All {S} without exception are {P}"
    ],
    "E": [
        "No {S} is a {P}", "No {S} are {P}", "None of the {S} are {P}", 
        "Nothing that is a {S} is a {P}", "There are no {S} that are {P}",
        "Not a single {S} is a {P}", "{S} are never {P}",
        "No entity that is a {S} can be a {P}", "It is false that some {S} are {P}",
        "Under no circumstances is a {S} a {P}", "Not a single creature that is a {S} is a {P}",
        "It must be true that no {S} is a {P}"
    ],
    "I": [
        "Some {S} are {P}", "Some {S} is a {P}", "At least one {S} is a {P}", 
        "A portion of {S} are {P}", "There exist {S} that are {P}",
        "A number of {S} are {P}", "Certain {S} are {P}",
        "A certain number of {S} are considered {P}", "There are {S} that can be seen as {P}",
        "It is the case that some {S} are {P}", "Typically, some {S} are {P}",
        "We can find some {S} that are {P}"
    ],
    "O": [
        "Some {S} are not {P}", "Not all {S} are {P}", "At least one {S} is not a {P}",
        "Some {S} is not a {P}", "There are {S} that are not {P}",
        "A portion of {S} are not {P}", "Not every {S} is a {P}",
        "Not everything that is a {S} is a {P}", "It is not true that all {S} are {P}",
        "Some {S} fail to be {P}", "There exist {S} which are not {P}",
        "A subset of {S} are not {P}"
    ]
}

CONNECTORS = [
    ". ", ". ", ". ", "; ", ", and ", ". Also, ", ". Furthermore, ", ". Moreover, ",
    ". Additionally, ", ". It is also known that ", ". We also know that ", 
    ". Further, ", ". In addition, "
]

CONCLUSIONS = [
    "Therefore,", "Thus,", "Hence,", "So,", "Consequently,", "It follows that",
    "This implies that", "We can conclude that", "As a result, it is clear that",
    "As such, it is necessarily true that", "It logically follows that",
    "One must conclude that", "Ideally, one concludes that", "This proves that",
    "For this reason,"
]

def generate_entry() -> Dict:
    """Generate a single training example with random mood/figure/words"""
    # Select distinct terms
    terms = random.sample(NONSENSE_NOUNS, 3)
    major, minor, middle = terms[0], terms[1], terms[2] # P, S, M
    
    # Select form
    moods = random.choices(["A", "E", "I", "O"], k=3) # P1, P2, Conc
    figure = random.randint(1, 4)
    
    # Organize terms based on figure
    # Fig 1: M-P, S-M
    # Fig 2: P-M, S-M
    # Fig 3: M-P, M-S
    # Fig 4: P-M, M-S
    
    if figure == 1:
        p1_s, p1_p = middle, major
        p2_s, p2_p = minor, middle
    elif figure == 2:
        p1_s, p1_p = major, middle
        p2_s, p2_p = minor, middle
    elif figure == 3:
        p1_s, p1_p = middle, major
        p2_s, p2_p = middle, minor
    else: # 4
        p1_s, p1_p = major, middle
        p2_s, p2_p = middle, minor
        
    # Build text
    t1 = random.choice(TEMPLATES[moods[0]]).format(S=p1_s, P=p1_p)
    t2 = random.choice(TEMPLATES[moods[1]]).format(S=p2_s, P=p2_p)
    
    # Canonicalize S and P for conclusion generation? 
    # Actually conclusion is always S-P (Minor-Major) in standard form.
    # But wait, SemEval task sometimes has valid syllogisms with different orders.
    # However, strictly speaking, a standard syllogism conclusion is ALWAYS S-P.
    # Let's generate standard S-P conclusions. The output Mood/Figure code assumes S-P conclusion.
    
    conc_text = random.choice(TEMPLATES[moods[2]]).format(S=minor, P=major)
    
    # Combine
    syllogism = f"{t1}{random.choice(CONNECTORS)}{t2} {random.choice(CONCLUSIONS)} {conc_text}"
    
    # Target label
    target = f"Mood: {''.join(moods)}, Figure: {figure}"
    
    return {
        "input_text": syllogism,
        "target_text": target
    }

class SyllogismDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        input_enc = self.tokenizer(
            item["input_text"], 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            item["target_text"], 
            max_length=32, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_enc.input_ids.squeeze(),
            "attention_mask": input_enc.attention_mask.squeeze(),
            "labels": target_enc.input_ids.squeeze()
        }

# ============================================================================
# 2. TRAINING LOOP
# ============================================================================

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200000, help="Number of synthetic samples to generate")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start training from (for resuming)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--output_dir", default="t5_syllogism_parser", help="Directory to save model checkpoints")
    parser.add_argument("--model", type=str, default="google/flan-t5-large", help="HuggingFace model name")
    
    # Check if running in Jupyter/Colab
    if 'ipykernel' in sys.modules:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Generate Data
    logger.info(f"Generating {args.samples} synthetic samples...")
    data = [generate_entry() for _ in range(args.samples)]
    
    # Split
    train_data, val_data = train_test_split(data, test_size=0.05)
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Model & Tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Datasets
    train_ds = SyllogismDataset(train_data, tokenizer)
    val_ds = SyllogismDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=total_steps
    )
    
    # Loop
    logger.info("Starting training...")
    
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # T5 handles shifting labels internally
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                val_loss += outputs.loss.item()
        
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save checkpoint
        save_path = f"{args.output_dir}/checkpoint-{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    # Final Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training Done! Model saved to {args.output_dir}")

if __name__ == "__main__":
    train()
