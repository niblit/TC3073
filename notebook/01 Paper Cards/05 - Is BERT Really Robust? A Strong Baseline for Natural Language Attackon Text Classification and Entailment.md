---
title: "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment"
authors: ["Di Jin", "Zhijing Jin", "Joey Tianyi Zhou", "Peter Szolovits"]
year: 2020
tags: ["adversarial-ml", "nlp", "evasion-attack", "textfooler", "week-5"]
project: "Adversarial Robustness Harness"
---

# Paper Card: Is BERT Really Robust?

**1. Full citation and link.**
> Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2020). Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 34, No. 05, pp. 8018-8025). https://arxiv.org/abs/1907.11932

---

**2. What problem is studied?**
- **The Core Problem:** The vulnerability of state-of-the-art NLP models (such as BERT, WordCNN, and WordLSTM) to black-box adversarial text attacks. Specifically, the authors investigate whether generating imperceptible, utility-preserving alterations (adversarial examples) can successfully fool these models into making wrong predictions.

---

**3. What evidence or data is used?**
- **Text Classification Datasets:** AG's News, Fake News Detection, MR (Movie Reviews), IMDB, and Yelp Polarity.
- **Textual Entailment Datasets:** SNLI and MultiNLI. 
- **Evaluation Subset:** 1,000 randomly selected examples from the test set for each task.

---

**4. What method is used?**
- **TEXTFOOLER:** A black-box attack framework comprising two main steps:
  1.  **Word Importance Ranking:** The system calculates an importance score for each word by querying the target model with sentences where individual words are deleted, tracking how much the prediction confidence drops. 
  2.  **Word Transformer:** The system filters out stop words and iteratively replaces the highest-ranked words with synonyms. Replacement candidates must match the original part-of-speech (POS) and maintain a high semantic similarity (measured via the Universal Sentence Encoder) to the original text. 

---

**5. What are the two most important findings?**
- **High Success Rate with Low Perturbation:** TEXTFOOLER successfully reduced the accuracy of the highly robust BERT model to below 15% (in most text classification tasks) and below 5% (for entailment) while perturbing fewer than 20% of the words. 
- **Necessity of Constraints:** Ablation studies showed that without the initial word importance ranking, the attack becomes largely ineffective. Furthermore, removing the semantic similarity constraint allows for easier attacks but severely degrades the meaning preservation of the text, making the attack obvious to humans.

---

**6. What is one important limitation?**
- **Semantic and Task-Sensitive Shifts:** The generated adversarial examples are susceptible to word sense ambiguity (e.g., using "testify" instead of "show" as a synonym) and task-sensitive content shifts, where changing a single word might inadvertently change the actual ground-truth meaning of the sentence (e.g., changing "boy" to "girl" might legitimately change an entailment label to neutral). 

---

**7. What will we use, change, test, or avoid because of this paper?**
> **Application to the Adversarial Robustness Harness Project:** 
> - **Use:** We will utilize TEXTFOOLER's two-step architecture (importance ranking followed by constrained synonym substitution) as a core blueprint for implementing our Week 4 "content-level transformations" module. We will also adopt their use of POS checking and cosine similarity limits to enforce our "intent-preserving" criteria[cite: 1, 2].
> - **Change/Test:** Instead of simply attacking until the model breaks, we will embed this methodology into our *budgeted search engine* to systematically plot degradation curves against our own detector zoo. We will test exactly at what perturbation budget (e.g., 5%, 10%, 15% modification limits) different models (like a TF-IDF logistic-regression model vs. a fine-tuned transformer) fail. 
> - **Avoid:** We will strictly avoid unconstrained synonym replacements (ablating the semantic similarity checks), as this violates our core threat model requirement that attacks must remain realistic, intent-preserving, and non-gibberish to human victims[cite: 1, 2].