# factcheck.py

import torch
from typing import List, Dict
import numpy as np
import gc
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter, defaultdict
import re, math

import nltk
from nltk.tokenize import sent_tokenize

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
    def __init__(self, model, tokenizer, max_length=256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def check_entailment_batch(self, premises: list, hypothesis: str):
        with torch.no_grad():
            # Tokenize premises and hypothesis in batches
            inputs = self.tokenizer(
                premises, [hypothesis] * len(premises),
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            # Get model predictions
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Calculate entailment probabilities
            entailment_scores = torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()  # Entailment scores
        # Cleanup
        del inputs, outputs, logits
        gc.collect()
        
        return entailment_scores  # Return only entailment scores for efficiency


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.27):
        self.threshold = threshold

    def preprocess(self, text: str):
        # Basic tokenization, removing very common words
        stop_words = {"is", "the", "a", "and", "of"}
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [word for word in tokens if word not in stop_words]

    def vectorize(self, tokens: List[str]) -> Dict[str, int]:
        # Count occurrences of each word
        return Counter(tokens)

    def cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        # Compute cosine similarity manually
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([val**2 for val in vec1.values()])
        sum2 = sum([val**2 for val in vec2.values()])
        denominator = np.sqrt(sum1) * np.sqrt(sum2)

        return numerator / denominator if denominator != 0 else 0.0

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = self.preprocess(fact)
        fact_vector = self.vectorize(fact_tokens)
        
        max_similarity = 0.0
        for passage in passages:
            passage_tokens = self.preprocess(passage['text'])
            passage_vector = self.vectorize(passage_tokens)
            similarity = self.cosine_similarity(fact_vector, passage_vector)
            max_similarity = max(max_similarity, similarity)

        return "S" if max_similarity >= self.threshold else "NS"


class EntailmentFactChecker:
    def __init__(self, ent_model, batch_size=4, early_exit_threshold=0.75, final_threshold=0.55):
        self.ent_model = ent_model
        self.batch_size = batch_size
        self.early_exit_threshold = early_exit_threshold
        self.final_threshold = final_threshold

    def predict(self, fact: str, passages: list) -> str:
        max_entailment_score = 0.0  # Track the highest entailment score
        
        for passage in passages:
            sentences = sent_tokenize(passage['text'])
            top_scores = []  # Track top entailment scores for averaging
            
            # Process sentences in batches
            for i in range(0, len(sentences), self.batch_size):
                batch = [sentence for sentence in sentences[i:i + self.batch_size] if sentence.strip()]
                entailment_scores = self.ent_model.check_entailment_batch(batch, fact)
                
                # Check for early exit on high-confidence entailment
                for score in entailment_scores:
                    if score > self.early_exit_threshold:
                        return "S"
                    top_scores.append(score)
                
                # Keep only top 3 scores for final averaging to reduce noise
                top_scores = sorted(top_scores, reverse=True)[:3]
            
            # Update max_entailment_score using a weighted average of top scores
            if top_scores:
                max_entailment_score = max(max_entailment_score, sum(top_scores) / len(top_scores))

        # Final decision with adjusted threshold
        return "S" if max_entailment_score > self.final_threshold else "NS"

# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
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
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

