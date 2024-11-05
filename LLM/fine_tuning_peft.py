from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)
from peft import get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
import time
from functools import wraps
import datetime
import psutil
import os

id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start monitoring
        start_time = time.time()
        start_mem = get_memory_usage()
        print(f"\nStarting {func.__name__}...")
        print(f"Initial memory: {start_mem:.2f} GB")
        
        # Execute function
        result = func(*args, **kwargs)
        
        # End monitoring
        end_time = time.time()
        end_mem = get_memory_usage()
        execution_time = end_time - start_time
        memory_increase = end_mem - start_mem
        
        # Print performance metrics
        print(f"\nPerformance metrics for {func.__name__}:")
        print(f"Execution time: {str(datetime.timedelta(seconds=execution_time))}")
        print(f"Final memory: {end_mem:.2f} GB")
        print(f"Memory increase: {memory_increase:.2f} GB")
        
        return result
    return wrapper

# Set device
device = torch.device("cpu")
print(f"Using device: {device}")
print(f"Initial system memory usage: {get_memory_usage():.2f} GB")
print(f"CPU Threads available: {torch.get_num_threads()}")

@performance_monitor
def load_base_model():
    model_checkpoint = 'distilbert-base-uncased'
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer, model_checkpoint

@performance_monitor
def load_and_process_dataset(tokenizer):
    dataset = load_dataset("shawhin/imdb-truncated")
    
    def tokenize_function(examples):
        text = examples["text"]
        tokenizer.truncation_side = "left"
        return tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

@performance_monitor
def create_peft_model(model):
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=['q_lin']
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

@performance_monitor
def train_model(model, tokenized_dataset, tokenizer, model_checkpoint):
    # Training configuration
    training_args = TrainingArguments(
        output_dir=model_checkpoint + "-lora-text-classification",
        learning_rate=1e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=True
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda p: {
            "accuracy": accuracy.compute(
                predictions=np.argmax(p[0], axis=1),
                references=p[1]
            )
        }
    )
    
    # Train
    return trainer.train()

@performance_monitor
def evaluate_model(model, tokenizer, device):
    text_list = [
        "It was good.", 
        "Not a fan, don't recommed.", 
        "Better than the first one.", 
        "This is not worth watching even once.", 
        "This one is a pass."
    ]
    
    print(f"\nModel predictions:")
    print("-" * 20)
    for text in text_list:
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predictions = torch.max(logits, 1).indices
        print(f"{text} - {id2label[predictions.tolist()[0]]}")

# Load accuracy metric
accuracy = evaluate.load("accuracy")

# Main execution
if __name__ == "__main__":
    print("\nStarting sentiment analysis training pipeline...")
    print("=" * 50)
    
    # Load base model and tokenizer
    model, tokenizer, model_checkpoint = load_base_model()
    
    # Load and process dataset
    tokenized_dataset = load_and_process_dataset(tokenizer)
    
    # Evaluate untrained model
    print("\nEvaluating untrained model:")
    evaluate_model(model, tokenizer, device)
    
    # Create PEFT model
    model = create_peft_model(model)
    
    # Train model
    training_results = train_model(model, tokenized_dataset, tokenizer, model_checkpoint)
    
    # Evaluate trained model
    print("\nEvaluating trained model:")
    evaluate_model(model, tokenizer, device)
    
    # Print final memory usage
    print(f"\nFinal system memory usage: {get_memory_usage():.2f} GB")