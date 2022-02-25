#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Vladislav Lialin and Namrata Shivagunde 
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script is based on
# https://github.com/huggingface/transformers/blob/426b96230a71f3c6e4decabae131c6e4f8bf4f5c/examples/pytorch/language-modeling/run_clm_no_trainer.py
# It was simplified for the purposes of this assignment.
#
# Usage example: python cli/train.py --tokenizer_path output_dir --output_dir output_dir --device cuda --batch_size 16
#
import os
import argparse
import logging
import math
import random

import datasets
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformer_lm.modeling_transformer import TransformerLM
import wandb


# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    # Data arguments
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        type=str,
        default=None,
        help="A path to tokenizer.json file",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )

    # Model arguments
    parser.add_argument(
        "--num_layers",
        default=6,
        type=int,
        help="Number of hidden layers in the Transformer encoder",
    )
    parser.add_argument(
        "--hidden_size",
        default=512,
        type=int,
        help="Hidden size of the Transformer encoder",
    )
    parser.add_argument(
        "--num_heads",
        default=8,
        type=int,
        help="Number of attention heads in the Transformer encoder",
    )
    parser.add_argument(
        "--fcn_hidden",
        default=512,
        type=int,
        help="Hidden size of the FCN",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout rate of the Transformer encoder",
    )

    # Training arguments
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) on which the code should run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=2_000,
        help="Perform an evaluation every this number of steps.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )

    # Misc
    parser.add_argument(
        "--wandb_project", 
        default="nlp_module_4_assignment", 
        help="wandb project name to log metrics to"
    )

    args = parser.parse_args()

    return args


def evaluate(model, eval_dataloader, device):
    # turn on evlauation mode: no dropout
    model.eval()
    n_correct = 0
    n_examples = 0
    total_eval_loss = torch.tensor(0.0, device=device)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            # Task 4.5: Evaluation step
            # 1. Compute the loss, just like in the training step
            # 2. Do not compute gradients, instead add the loss to the total_eval_loss
            # 3. Compute the number of correct predictions, and add it to n_correct, convert it to a python int
            # You can do it like this: (torch.argmax(logits, dim=-1) == labels).sum().item()
            # 4. Increment n_examples by the batch size, which is the same as len(labels), for example
            # Our implementation is 5 lines
            # YOUR CODE STARTS HERE
            input_ids, labels = batch['input_ids'].to(device=device), batch['labels'].to(device=device)
            logits = model(input_ids)
            total_eval_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1).long())
            n_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
            n_examples += input_ids.size(0)

            # YOUR CODE ENDS HERE

    eval_loss = (total_eval_loss / len(eval_dataloader)).item()
    accuracy = n_correct / n_examples
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        logger.warning("Perplexity is infinity. Loss is probably NaN, please check your code for bugs.")
        perplexity = float("inf")
    
    # turn off evaluation mode
    model.train()

    return {
        "eval/loss": eval_loss,
        "eval/perplexity": perplexity,
        "eval/accuracy": accuracy,
    }


def main():
    args = parse_args()

    # Use Datsets to load wikitext corpus. This is a very relatively dataset, less than 1Gb.
    # wikitext-103-v1 is a particular version of this dataset, you can look for other versions here: https://huggingface.co/datasets/wikitext
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    raw_datasets = load_dataset("wikitext", "wikitext-103-v1")

    # remove all of texts less than 2 characters
    raw_datasets = raw_datasets.filter(lambda x: len(x["text"]) > 1)

    # Initialize wandb as soon as possible to log all stdout to the cloud
    wandb.init(project=args.wandb_project, config=args)

    # Set random seed to make results reproducible
    if args.seed is not None:
        set_seed(args.seed)

    # Task 4.1: Load tokenizer from args.tokenizer_path using transformers.PreTrainedTokenizerFast.from_pretrained
    # https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
    # Our implementation is one line.
    # YOUR CODE STARTS HERE
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    # YOUR CODE ENDS HERE

    # Task 4.2: Create TransformerEncoder object
    # Provide all of the TransformerLM initialization arguments from args.
    # Specify vocab_size using .vocab_size attribute of the tokenizer
    # Move model to the device we use for training
    # YOUR CODE STARTS HERE
    model = TransformerLM(args.num_layers, args.hidden_size, args.num_heads, args.fcn_hidden, tokenizer.vocab_size,
                          args.max_seq_length, args.dropout_rate)
    if args.device == 'cuda':
        model.cuda()
    # YOUR CODE ENDS HERE

    wandb.watch(model)
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text"

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Tokenizing the dataset",
    )

    # Inline question 1: How does .map function work? What does `batched` means and why are we using multiple workers?
    # Documentation: https://huggingface.co/docs/datasets/about_map_batch.html
    # YOUR ANSRWER HERE: It updates the table according to tokenize_function. batched=True means that this mapping will be handled with smaller batches and this process can be parallalized with multiple workers.

    def group_texts(examples):
        """
        Args:
            examples: a dict with key "input_ids" containing a list of lists of input_ids of tokenized texts
        """
        # Concatenate all texts.
        concatenated_examples = []
        for example in examples["input_ids"]:
            concatenated_examples.extend(example)

        total_length = len(concatenated_examples)
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= args.max_seq_length:
            total_length = (total_length // args.max_seq_length) * args.max_seq_length

        # Split by chunks of max_len.
        chunks = []
        for i in range(0, total_length, args.max_seq_length):
            chunks.append(concatenated_examples[i : i + args.max_seq_length])
        
        input_ids, labels = [] , []
        for chunk in chunks:
            input_ids.append(chunk[:-1])
            labels.append(chunk[1:])

        result = {"input_ids": input_ids, "labels": labels}
        return result

    # Inline question 2: What does group_texts do? What is the purpose of it?
    # YOUR ANSWER HERE: Grouped data into chunks for preparing dataset for training. It is structured for next word prediction tasks.

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # import ipdb; ipdb.set_trace()
    column_names = tokenized_datasets["train"].column_names
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,  # required when changing the size of the dataset in the map function
        desc=f"Grouping texts in chunks of {args.max_seq_length}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Create DataLoader objects
    # They will make batches from your dataset examples and shuffle them if you want.
    # collate_fn is a function that merges several examples into a batch
    def collate_fn(examples):
        return {
            "input_ids": torch.tensor([e["input_ids"] for e in examples], dtype=torch.long),
            "labels": torch.tensor([e["labels"] for e in examples], dtype=torch.long),
        }

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.batch_size
    )

    # Task 4.3: Create Optimizer
    # Use AdamW optimizer and provide learning_rate and weight_decay from args.
    # YOUR CODE STARTS HERE
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # YOUR CODE ENDS HERE

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))
    global_step = 0

    # Training loop
    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # Task 4.4: Training step
            # batch is a dictionary with keys "input_ids" and "labels"
            # 1. Move input_ids and labels to the device you use for training
            # 2. Produce logits with your model
            # 3. Use F.cross_entropy to compute the loss.
            # Notice that you might need to reshape the tensors to do that.
            # 4. Compute the loss gradients with .backward()
            # 5. Update the parameters
            # 6. Update the learning rate using the scheduler
            # 7. Zero out the gradients so that they don't accumulate between steps
            # YOUR CODE STARTS HERE
            input_ids, labels = batch['input_ids'].to(device=args.device), batch['labels'].to(device=args.device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1).long())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # YOUR CODE ENDS HERE

            progress_bar.update(1)
            global_step += 1

            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if global_step >= args.max_train_steps:
                break

            if global_step % args.eval_every_steps == 0:
                metrics = evaluate(model, eval_dataloader, args.device)
                wandb.log(metrics, step=global_step)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                model.save_pretrained(args.output_dir)
    
    logger.info("Final evaluation")
    metrics = evaluate(model, eval_dataloader, args.device)
    wandb.log(metrics, step=global_step)

    logger.info("Saving final model checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir)

    logger.info("Uploading tokenizer, model and config to wandb")
    wandb.save(os.path.join(args.output_dir, "*"))

    logger.info(f"Script finished succesfully, model saved in {args.output_dir}")


if __name__ == "__main__":
    main()
