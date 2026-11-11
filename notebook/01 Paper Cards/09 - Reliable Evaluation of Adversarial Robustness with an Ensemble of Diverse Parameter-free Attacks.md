---
title: "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks"
authors: [Francesco Croce, Matthias Hein]
year: 2020
tags: [adversarial-ml, robustness-evaluation, evasion-attacks, autoattack, pgd, gradient-masking]
project: "Adversarial Robustness Harness"
---

# Paper Card: Reliable Evaluation of Adversarial Robustness

**1. Full citation and link.**
> Croce, F., & Hein, M. (2020). Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks. *Proceedings of the 37th International Conference on Machine Learning (ICML)*. arXiv:2003.01690v2. [Link](https://arxiv.org/abs/2003.01690)

---

**2. What problem is studied?**
- **The illusion of adversarial robustness:** Many machine learning defenses appear highly robust against evasion attacks during initial testing but are easily broken later. This overestimation of robustness stems from poor evaluation protocols—specifically, suboptimal hyperparameter tuning (e.g., fixed step sizes) in Projected Gradient Descent (PGD) attacks and defenses that inadvertently cause gradient masking/obfuscation.

---

**3. What evidence or data is used?**
- **Large-scale model evaluation:** The authors evaluated over 50 robust classifiers proposed in 35 top-tier papers (from venues like ICML, NeurIPS, ICLR, CVPR).
- **Standard vision datasets:** The evaluation spanned MNIST, CIFAR-10, CIFAR-100, and ImageNet, encompassing both deterministic and randomized defense mechanisms under $l_\infty$ and $l_2$ perturbation threat models.

---

**4. What method is used?**
- **AutoAttack Ensemble:** The authors created a parameter-free, user-independent ensemble of four complementary attacks to accurately estimate true worst-case robustness:
  1. $APGD_{CE}$: A novel "Auto-PGD" attack using Cross-Entropy loss that dynamically adjusts its step size based on a defined iteration budget and momentum, eliminating the fixed step-size flaw.
  2. $APGD_{DLR}^T$: Auto-PGD using a novel, targeted Difference of Logits Ratio (DLR) loss, which is mathematically invariant to logit scaling and shifting.
  3. $FAB^T$: A targeted Fast Adaptive Boundary attack minimizing perturbation norm.
  4. **Square Attack:** A score-based, black-box random search attack to catch defenses that completely shatter gradients.

---

**5. What are the two most important findings?**
- **Widespread failure of state-of-the-art defenses:** When subjected to the parameter-free AutoAttack, all except one of the evaluated deterministic models exhibited lower robust accuracy than originally reported. In 13 cases, the robust accuracy dropped by more than 10%, exposing many "state-of-the-art" defenses as completely broken.
- **DLR loss defeats gradient masking:** Standard Cross-Entropy (CE) loss gradients can artificially vanish when logits are rescaled, tricking PGD into failing. The newly proposed DLR loss maintains scaling and shift invariance, effectively neutralizing defenses that rely on this specific type of gradient obfuscation.

---

**6. What is one important limitation?**
- **Discontinuous classifiers still challenge gradient-based methods:** While APGD and DLR loss effectively handle scaling and shifting, they can still overestimate the robustness of models with discontinuous loss landscapes (like the k-winners-take-all defense). In these edge cases, the ensemble must rely heavily on the computationally expensive, gradient-free Square Attack to find adversarial examples.

---

**7. What will we use, change, test, or avoid because of this paper?**
> **For the Adversarial Robustness Harness & Detector Zoo:**
> - **Change:** We must eliminate hardcoded/fixed step sizes ($\eta$) in our engine's gradient-based transformations. We will implement APGD's budget-aware momentum scheduling to govern the attack trajectory under our perturbation limits.
> - **Use:** We will incorporate the scale-invariant DLR objective function into our evasion engine. Relying strictly on Cross-Entropy loss will cause us to falsely certify detectors that merely mask gradients.
> - **Test:** Our detector zoo must be evaluated against a *diverse ensemble* (white-box + score-based black box) rather than a single parameterized attack. This ensures intent-preserving transformations succeed against the underlying model logic, not just its gradient topology.
> - **Avoid:** We will strictly avoid allowing manual hyperparameter tuning of the evasion attacks during detector benchmarking. If an attack requires fine-tuning to break a specific detector, the evaluation protocol is not reproducible.
