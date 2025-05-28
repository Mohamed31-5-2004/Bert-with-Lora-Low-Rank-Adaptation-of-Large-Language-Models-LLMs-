import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

dataset = load_dataset("glue", "stsb")

def binarize(example):
    example["label"] = int(example["label"] >= 4.0)
    return example

dataset = dataset.map(binarize)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

encoded = dataset.map(preprocess, batched=False)
encoded = encoded.remove_columns(['sentence1', 'sentence2', 'idx'])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./results_sts_binary",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs_sts_binary",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
)

trainer.train()
results = trainer.evaluate()
print("Validation results:", results)

def predict_similarity(sent1, sent2):
    inputs = tokenizer(sent1, sent2, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return "Similar" if pred == 1 else "Not similar", probs.detach().numpy()

s1 = "A man is playing a guitar."
s2 = "A person plays an instrument."
result, probabilities = predict_similarity(s1, s2)
print(f"Prediction: {result}, Probabilities: {probabilities}")