---
title: "TextAttack A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP"
authors: ["John X. Morris", "Eli Lifland", "Jin Yong Yoo", "Jake Grigsby", "Di Jin", "Yanjun Qi"]
year: 2020
tags: [ "AdversarialML", "NLP", "EvasionHarness", "ModelRobustness", PerturbationBudget ]
project: "Adversarial Robustness Harness"
---
**1. Full citation and link.**
> Morris, J. X., Lifland, E., Yoo, J. Y., Grigsby, J., Jin, D., & Qi, Y. (2020). TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP. *arXiv preprint arXiv:2005.05909v4*. https://arxiv.org/abs/2005.05909

---

**2. What problem is studied?**
- **The fragmentation of NLP adversarial research:** Prior to this paper, adversarial attacks against NLP models were implemented in disparate, isolated code repositories. This lack of standardization made it exceedingly difficult to fairly benchmark attacks against one another, reproduce claimed results, or easily apply these attacks to improve model robustness through adversarial training.

---

**3. What evidence or data is used?**
- **Broad Model & Dataset Testing:** The framework is evaluated using 82+ pre-trained models (including LSTMs, CNNs, BERT, and other transformers) across multiple datasets, including all GLUE benchmark tasks, IMDB, AG News, and Rotten Tomatoes.
- **Benchmarking Literature:** The authors provide comparative evidence by re-implementing 16 existing attacks (e.g., TextFooler, DeepWordBug, BAE) and comparing TextAttack's success rates and perturbed word percentages against the original metrics reported in those papers.

---

**4. What method is used?**
- **Modular Architecture:** The authors built an extensible Python framework that frames NLP attacks as combinatorial search problems. They decomposed every attack into exactly four interchangeable components: a **Goal Function**, a set of **Constraints**, a **Transformation**, and a **Search Method**. 
- **Under-the-Hood Optimization:** They implemented an `AttackedText` object to preserve capitalization and tokenization during word swaps, and utilized caching/memoization to drastically speed up combinatorial searches.

---

**5. What are the two most important findings?**
- **Component Reusability Unifies the Field:** By decomposing attacks into four modular components, researchers can accurately recreate 16 highly distinct attacks from the literature (from genetic algorithms to greedy word swaps) within a single ecosystem, proving that most novel attacks are just slight reconfigurations of existing components.
- **Seamless Defensive Integration:** Providing a standardized attack framework allows for immediate improvements in model defense; the authors demonstrated that adversarial training using TextAttack significantly boosts robustness (e.g., an LSTM trained on the SST-2 dataset saw its accuracy under a TextFooler attack improve from 23.46% to 44.74% after 75 epochs).

---

**6. What is one important limitation?**
- **Dataset-Agnostic Resource Sacrifices:** Because the framework is designed to work universally across datasets, it occasionally deviates from the highly specialized, bespoke implementations of the original papers. For example, TextAttack uses standard GloVe embeddings instead of counter-fitted ones for certain baselines, and concatenates HowNet synonym sets, leading to slight variances in exact reproducibility compared to the originally reported attack metrics.

---

**7. What will we use, change, test, or avoid because of this paper?**
> **Use:** As outlined, our goal is to build a controlled evasion and robustness engine. We will directly adopt TextAttack's 4-component architectural paradigm (Goal, Constraints, Transformation, Search) as the software blueprint for our own extensible harness to evaluate our self-trained detector zoo. We will also utilize their `AttackedText` caching mechanism to manage string modifications efficiently.
> **Change & Test:** To align with our specific research objective of measuring *realistic* degradation curves under an *explicit perturbation budget*, we will strictly tune the **Constraints** module (e.g., maximum edit distance, max percentage of words perturbed). Instead of letting the search method run until the model breaks with gibberish, we will enforce strict semantic validity constraints to ensure the evasion attacks remain intent-preserving problem-space attacks, revealing whether our detector zoo learned malice or just fragile shortcuts.
