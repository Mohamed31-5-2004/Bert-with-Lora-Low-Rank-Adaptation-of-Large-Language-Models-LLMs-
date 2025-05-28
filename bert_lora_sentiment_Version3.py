import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

# 1. Load dataset (example: yelp_review_full)
dataset = load_dataset('yelp_review_full')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding='max_length',
        max_length=128
    )

encoded = dataset.map(preprocess, batched=False)
encoded = encoded.rename_column('label', 'labels')
encoded = encoded.remove_columns(['text'])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir='./results_sentiment',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs_sentiment',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    tokenizer=tokenizer,
)

trainer.train()
results = trainer.evaluate()
print(results)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return f'Sentiment: {pred}', probs.detach().numpy()

# Example usage
text = "I loved this product, it was amazing!"
result, probabilities = predict_sentiment(text)
print(f'Result: {result}, Probabilities: {probabilities}')