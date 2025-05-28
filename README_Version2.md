# BERT with LoRA: Tasks Implementation

This repository contains Python scripts for three core NLP tasks using BERT fine-tuned with LoRA (Low-Rank Adaptation):

- **Sentiment Analysis (SST-5, 5-way)**  
- **Paraphrase Detection (Quora, binary)**  
- **Semantic Textual Similarity (STS-B, binary)**  

All tasks use HuggingFace's `transformers`, `datasets`, and the `peft` library for Parameter-Efficient Fine-Tuning (LoRA).

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Sentiment Analysis (SST-5)](#1-sentiment-analysis-sst-5)
  - [2. Paraphrase Detection (Quora)](#2-paraphrase-detection-quora)
  - [3. Semantic Textual Similarity (STS-B)](#3-semantic-textual-similarity-sts-b)
- [Jupyter Notebook](#jupyter-notebook)
- [Citation](#citation)

---

## Requirements

```bash
pip install torch transformers datasets peft
```

- Python ≥ 3.8 recommended

## Usage

### 1. Sentiment Analysis (SST-5)

```bash
python bert_lora_sst_sentiment.py
```
- Dataset: [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5)
- 5-class sentiment (0 = very negative ... 4 = very positive)

### 2. Paraphrase Detection (Quora)

```bash
python bert_lora_quora_paraphrase.py
```
- Dataset: [quora](https://huggingface.co/datasets/quora)
- Binary classification (0 = not paraphrase, 1 = paraphrase)

### 3. Semantic Textual Similarity (STS-B)

```bash
python bert_lora_sts_binary.py
```
- Dataset: [GLUE STS-B](https://huggingface.co/datasets/glue/viewer/stsb)
- The similarity score is binarized (0 = not similar, 1 = similar, threshold ≥ 4.0)

---

## Jupyter Notebook

For interactive exploration and running all three tasks, see:

- [bert_lora_tasks_notebook.ipynb](./bert_lora_tasks_notebook.ipynb)

---

## Citation

If you use these scripts, please cite the original [LoRA paper](https://arxiv.org/abs/2106.09685) and [HuggingFace Transformers](https://github.com/huggingface/transformers).

---

**Author:** [Mohamed31-5-2004](https://github.com/Mohamed31-5-2004)