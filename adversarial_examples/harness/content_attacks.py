import string
import nltk
from nltk.corpus import wordnet, stopwords
from harness.taxonomy import ContentTransformation
from harness.budget import PerturbationBudget, BudgetExceededError

for resource in ['wordnet', 'averaged_perceptron_tagger_eng', 'stopwords', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class SynonymSubstitution(ContentTransformation):
    """
    Substitutes specific trigger words with intent-preserving synonyms.
    Consumes 1 budget unit per replacement.
    """
    def __init__(self, name: str = "synonym_substitution"):
        super().__init__(name)
        # Deterministic mapping to preserve intent (HotFlip rule).
        self.synonyms = {
            "update": "change",
            "immediately": "urgently",
            "verify": "confirm",
            "account": "profile"
        }

    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        words = text.split()
        mutated_words = []
        
        for word in words:
            # Basic sanitization to check against dictionary
            clean_word = word.lower().strip(".,!?")
            
            if clean_word in self.synonyms and budget.remaining >= 1:
                budget.consume(1)
                mutated_words.append(self.synonyms[clean_word])
            else:
                mutated_words.append(word)
                
        return " ".join(mutated_words)


class ParaphraseTransformation(ContentTransformation):
    """
    Paraphrases an entire sentence or clause.
    Has a fixed higher budget cost due to structural changes.
    """
    def __init__(self, name: str = "formal_paraphrase", cost: int = 3):
        super().__init__(name)
        self.cost = cost
        
    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        # Pre-check budget before attempting a heavy structural change
        if budget.remaining < self.cost:
            raise BudgetExceededError(
                f"Paraphrase requires {self.cost} budget, but only {budget.remaining} left."
            )
            
        budget.consume(self.cost)
        
        # Prototype deterministic paraphrase. 
        # Future iterations can inject a local LLM or seq2seq model here.
        if "Click the link to verify your email" in text:
            return text.replace(
                "Click the link to verify your email", 
                "Please confirm your email address by following the provided link"
            )
            
        return text


class DynamicSynonymSubstitution(ContentTransformation):
    """
    Substitutes words dynamically, matching Part-of-Speech and evaluating
    contextual semantic similarity (inspired by TextFooler).
    """
    def __init__(self, name: str = "dynamic_synonym_substitution"):
        super().__init__(name)
        self.stop_words = set(stopwords.words('english'))
        self.skip_chars = set(string.punctuation)

    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        from harness.nlp_loader import get_spacy_model, get_sentence_transformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        nlp = get_spacy_model()
        st_model = get_sentence_transformer()
        
        doc = nlp(text)
        mutated_tokens = []
        words = [tok.text for tok in doc]
        
        for i, token in enumerate(doc):
            if (budget.remaining <= 0 or 
                token.text.lower() in self.stop_words or 
                any(char in self.skip_chars for char in token.text)):
                mutated_tokens.append(token.text)
                continue
                
            # Get WordNet POS based on spaCy POS
            wn_pos = None
            if token.pos_ == 'ADJ':
                wn_pos = wordnet.ADJ
            elif token.pos_ == 'VERB':
                wn_pos = wordnet.VERB
            elif token.pos_ == 'NOUN':
                wn_pos = wordnet.NOUN
            elif token.pos_ == 'ADV':
                wn_pos = wordnet.ADV
                
            if not wn_pos:
                mutated_tokens.append(token.text)
                continue
                
            synonyms = set()
            for syn in wordnet.synsets(token.text, pos=wn_pos):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    if lemma_name.lower() != token.text.lower():
                        synonyms.add(lemma_name)
                        
            best_synonym = None
            best_sim = -1.0
            
            if synonyms:
                orig_embedding = st_model.encode([text])
                
                for syn_candidate in synonyms:
                    # Check if POS of the replacement matches original
                    syn_doc = nlp(syn_candidate)
                    if len(syn_doc) > 0 and syn_doc[0].pos_ != token.pos_:
                        continue
                        
                    # Create candidate sentence
                    candidate_words = words.copy()
                    candidate_words[i] = syn_candidate
                    candidate_text = " ".join(candidate_words).replace(" '", "'").replace(" ,", ",").replace(" .", ".")
                    
                    candidate_embedding = st_model.encode([candidate_text])
                    sim = cosine_similarity(orig_embedding, candidate_embedding)[0][0]
                    
                    if sim > best_sim:
                        # Also check verify_intent (it will be checked on final result anyway, but good to filter here)
                        best_sim = sim
                        best_synonym = syn_candidate
                        
            if best_synonym is not None and best_sim > 0.5: # basic threshold for selection
                mutated_tokens.append(best_synonym)
                budget.consume(1)
            else:
                mutated_tokens.append(token.text)
                
        result = " ".join(mutated_tokens).replace(" '", "'").replace(" ,", ",").replace(" .", ".")
        return result