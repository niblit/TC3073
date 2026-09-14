from abc import ABC, abstractmethod
from harness.budget import PerturbationBudget

class Transformation(ABC):
    """
    The root interface for all intent-preserving transformations.
    """
    def __init__(self, name: str):
        self.name = name

    def transform(self, text: str, budget: PerturbationBudget) -> str:
        """
        Public API to execute the transformation. 
        Subclasses implement _apply to house the actual logic.
        Validates intent preservation before returning.
        """
        # Logging could be injected here for reproducibility
        transformed_text = self._apply(text, budget)
        if not self.verify_intent(text, transformed_text):
            return text
        return transformed_text

    def verify_intent(self, original_text: str, transformed_text: str, threshold: float = 0.70) -> bool:
        """
        Verifies that the transformed text preserves the semantic intent and invariants.
        Uses sentence-transformers for semantic similarity and spaCy for hard invariants.
        """
        from harness.nlp_loader import get_spacy_model, get_sentence_transformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if original_text == transformed_text:
            return True
            
        # 1. Semantic Similarity Check
        st_model = get_sentence_transformer()
        embeddings = st_model.encode([original_text, transformed_text])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        if sim < threshold:
            return False
            
        # 2. Hard Invariants Check
        nlp = get_spacy_model()
        doc_orig = nlp(original_text)
        doc_trans = nlp(transformed_text)
        
        def get_invariants(doc):
            entities = {ent.text.lower() for ent in doc.ents if ent.label_ not in ('DATE', 'TIME')}
            numbers = {tok.text.lower() for tok in doc if tok.like_num}
            urls_emails = {tok.text.lower() for tok in doc if tok.like_url or tok.like_email}
            # Find the root verb
            root_verbs = {tok.lemma_.lower() for tok in doc if tok.dep_ == 'ROOT' and tok.pos_ == 'VERB'}
            return entities, numbers, urls_emails, root_verbs
            
        orig_ents, orig_nums, orig_urls, orig_roots = get_invariants(doc_orig)
        trans_ents, trans_nums, trans_urls, trans_roots = get_invariants(doc_trans)
        
        if not orig_ents.issubset(trans_ents):
            return False
        if not orig_nums.issubset(trans_nums):
            return False
        if not orig_urls.issubset(trans_urls):
            return False
        # Relaxing the root verb constraint to allow verbs to be substituted with synonyms
        # if orig_roots and not orig_roots.issubset(trans_roots):
        #     return False
            
        return True

    @abstractmethod
    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        """
        Internal transformation logic. Must be overridden by subclasses.
        Must explicitly call budget.consume(cost) before returning.
        """
        pass

# --- The Taxonomy Tiers ---

class ContentTransformation(Transformation):
    """
    Transformations targeting the semantic payload (e.g., paraphrase, synonyms).
    """
    pass

class StructureTransformation(Transformation):
    """
    Transformations targeting the syntax, layout, or DOM (e.g., HTML tag injection).
    """
    pass

class MetadataTransformation(Transformation):
    """
    Transformations targeting the delivery mechanism (e.g., Sender headers, timestamps).
    """
    pass
