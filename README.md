# Fact-Checking Outputs from ChatGPT

This repository contains the implementation for **Assignment 4: Fact-checking Outputs from ChatGPT**. The goal of this project is to analyze and evaluate the factual accuracy of large language model (LLM) outputs, using techniques like word overlap and textual entailment models.

## Overview

The assignment focuses on developing and comparing fact-checking techniques to assess ChatGPT-generated claims. It involves processing claims, verifying them against Wikipedia passages, and classifying them as "Supported" (S) or "Not Supported" (NS).

### Objectives:
1. **Word Overlap Method:** Implement a baseline approach using text similarity metrics like cosine similarity and Jaccard similarity.
2. **Textual Entailment Model:** Leverage a pre-trained DeBERTa-v3 model fine-tuned on entailment tasks to determine whether claims are supported by evidence.
3. **Error Analysis:** Analyze false positives and negatives to improve model performance.

---

## Features

- **Dataset:** Human-labeled claims and evidence from Wikipedia, focusing on factual claims from ChatGPT biographies.
- **Methods:**
  - **Word Overlap:** Calculates text overlap to classify factual accuracy.
  - **Textual Entailment:** Utilizes pre-trained NLP models for logical entailment.
- **Error Analysis:** Provides insights into model misclassifications.

---

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:<username>/Assignment-4-Factuality-and-ChatGPT.git
   cd Assignment-4-Factuality-and-ChatGPT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Python 3.8+ and PyTorch installed.

---

## Usage

### Run Baseline Models
Execute the baseline models to evaluate random predictions and "always entail" methods:
```bash
python factchecking_main.py --mode random
python factchecking_main.py --mode always_entail
```

### Word Overlap Model
Run the word overlap model:
```bash
python factchecking_main.py --mode word_overlap
```

### Textual Entailment Model
Run the entailment model (requires GPU for optimal performance):
```bash
python factchecking_main.py --mode entailment --cuda
```

---

## Error Analysis

Analyze model misclassifications by reviewing false positives and false negatives. Results can be used to fine-tune thresholds and improve accuracy.

---

## References

- [FActScore](https://github.com/shmsw25/FActScore) - Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.
- He, P., Liu, X., Gao, J., & Chen, W. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
- Thorne, J., et al. (2018). FEVER: A large-scale dataset for fact extraction and verification.

---

## Contributors

This repository is maintained by **Steven R. Murphey II** as part of academic coursework on natural language processing and machine learning.

