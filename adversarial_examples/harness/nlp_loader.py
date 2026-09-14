import spacy
from sentence_transformers import SentenceTransformer

_nlp_spacy = None
_sentence_model = None

def get_spacy_model():
    """Lazily loads and returns the spaCy en_core_web_sm model."""
    global _nlp_spacy
    if _nlp_spacy is None:
        _nlp_spacy = spacy.load("en_core_web_sm")
    return _nlp_spacy

def get_sentence_transformer():
    """Lazily loads and returns the SentenceTransformer model."""
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model
