# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

class FactExample:
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            logits = outputs.logits
            num_classes = logits.size(-1)

            # Handle cases where logits are size 2 or 3 (binary or multi-class classification)
            if num_classes == 3:
                entailment_score = torch.softmax(logits, dim=-1)[0][2].item()  # Score for "entailment"
            elif num_classes == 2:
                entailment_score = torch.sigmoid(logits[0][1]).item()  # Binary entailment score
            else:
                raise ValueError(f"Unexpected number of classes: {num_classes}")
            
            return entailment_score

        del inputs, outputs, logits
        gc.collect()


class FactChecker:
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return np.random.choice(["S", "NS"])


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        # Define stop words using nltk's set, but avoid tokenization that depends on 'punkt'
        stop_words = set(stopwords.words('english'))
        
        # Tokenize fact and passages by splitting on whitespace
        fact_tokens = set(fact.lower().split()) - stop_words
        best_score = 0

        for passage in passages:
            passage_tokens = set(passage['text'].lower().split()) - stop_words
            score = len(fact_tokens & passage_tokens) / len(fact_tokens | passage_tokens)
            best_score = max(best_score, score)
        
        # Classify as "Supported" if overlap is above threshold
        return "S" if best_score > 0.5 else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model: EntailmentModel):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        best_score = -1
        for passage in passages:
            entailment_score = self.ent_model.check_entailment(fact, passage['text'])
            best_score = max(best_score, entailment_score)
        
        return "S" if best_score > 0.7 else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations