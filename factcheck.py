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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            logits = outputs.logits

        entailment_score = torch.softmax(logits, dim=-1)[0][0].item()

        del inputs, outputs, logits
        gc.collect()

        return entailment_score


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


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, overlap_threshold=0.2):
        # Use the passed entailment model and tokenizer
        self.model = ent_model.model
        self.tokenizer = ent_model.tokenizer
        # Initialize the word overlap checker with the specified threshold
        self.overlap_checker = WordRecallThresholdFactChecker(threshold=overlap_threshold)

        # Move model to mixed precision on GPU if available
        if torch.cuda.is_available():
            self.model = self.model.half().to("cuda")

    def predict(self, fact: str, passages: List[dict]) -> str:
        max_entailment_score = 0.0

        for passage in passages:
            sentences = nltk.sent_tokenize(passage['text'])

            for sentence in sentences:
                # Check if sentence has sufficient overlap with the fact
                overlap_score = self.overlap_checker.cosine_similarity(
                    self.overlap_checker.vectorize(self.overlap_checker.preprocess(fact)),
                    self.overlap_checker.vectorize(self.overlap_checker.preprocess(sentence))
                )
                if overlap_score < self.overlap_checker.threshold:
                    continue  # Skip sentences with low overlap

                # Perform entailment scoring on sentences that pass the overlap check
                inputs = self.tokenizer(fact, sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    entailment_prob = torch.softmax(logits, dim=-1)[0][0].item()  # Probability of entailment
                    contradiction_prob = torch.softmax(logits, dim=-1)[0][2].item()  # Probability of contradiction

                # Update max_entailment_score only if the sentence is more confidently entailing
                if entailment_prob > 0.75 and contradiction_prob < 0.2:
                    max_entailment_score = max(max_entailment_score, entailment_prob)

        # Decision based on the maximum entailment score
        return "S" if max_entailment_score >= 0.75 else "NS"

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

