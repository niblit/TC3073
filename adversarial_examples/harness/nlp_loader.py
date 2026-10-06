import os
import spacy
from sentence_transformers import SentenceTransformer

_nlp_spacy = None
_sentence_model = None
_use_model = None
_counter_fitted_vectors = None

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

def get_use_model():
    """
    Lazily loads the Universal Sentence Encoder (USE) from TensorFlow Hub.
    Returns a callable that takes a list of strings and returns a numpy array of embeddings.
    
    Uses the USE v4 model: https://tfhub.dev/google/universal-sentence-encoder/4
    """
    global _use_model
    if _use_model is None:
        import tensorflow_hub as hub
        _use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return _use_model

def get_counter_fitted_vectors(vectors_path=None):
    """
    Lazily loads counter-fitted word vectors (Mrkšić et al., 2016).
    
    Expects word2vec-format .txt file. Default path can be overridden via:
      - The `vectors_path` argument
      - The COUNTER_FITTED_VECTORS_PATH environment variable
      - Falls back to 'data/counter-fitted-vectors.txt' relative to project root
    
    Returns a gensim KeyedVectors object.
    """
    global _counter_fitted_vectors
    if _counter_fitted_vectors is None:
        from gensim.models import KeyedVectors
        
        if vectors_path is None:
            vectors_path = os.environ.get(
                "COUNTER_FITTED_VECTORS_PATH",
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "counter-fitted-vectors.txt")
            )
        
        _counter_fitted_vectors = KeyedVectors.load_word2vec_format(
            vectors_path, binary=False, no_header=True
        )
    return _counter_fitted_vectors
