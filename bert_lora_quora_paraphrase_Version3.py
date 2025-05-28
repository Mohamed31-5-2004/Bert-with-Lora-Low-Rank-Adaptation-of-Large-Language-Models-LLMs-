import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

dataset = load_dataset('quora')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess(example):
    return tokenizer(
        example['questions']['text'][0],
        example['questions']['text'][1],
        truncation=True,
        padding='max_length',
        max_length=128
    )

encoded = dataset.map(preprocess, batched=False)
encoded = encoded.rename_column('is_duplicate', 'labels')
encoded = encoded.remove_columns(['questions', 'id'])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir='./results_paraphrase',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs_paraphrase',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded['train'],
    eval_dataset=encoded['validation'],
    tokenizer=tokenizer,
)

trainer.train()
results = trainer.evaluate()
print("Validation results:", results)

def predict_paraphrase(q1, q2):
    inputs = tokenizer(q1, q2, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return "Paraphrase" if pred == 1 else "Not paraphrase", probs.detach().numpy()

q1 = "How can I learn to play guitar?"
q2 = "What's the best way to become a guitarist?"
result, probabilities = predict_paraphrase(q1, q2)
print(f"Prediction: {result}, Probabilities: {probabilities}")