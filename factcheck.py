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
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self):
        # Load the pre-trained DeBERTa model for entailment tasks
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            entailment_score = torch.softmax(logits, dim=-1)[0][2].item()  # Entailment class score
            return entailment_score

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()


class FactChecker:
    """
    Fact checker base type
    """
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
        stop_words = set(stopwords.words('english'))
        fact_tokens = set(word_tokenize(fact.lower())) - stop_words
        best_score = 0

        for passage in passages:
            passage_tokens = set(word_tokenize(passage['text'].lower())) - stop_words
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
        
        # Classify as "Supported" if entailment score is above threshold
        return "S" if best_score > 0.7 else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
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