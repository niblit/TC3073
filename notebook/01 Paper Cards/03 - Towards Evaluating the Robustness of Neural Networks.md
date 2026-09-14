---
title: "Towards Evaluating the Robustness of Neural Networks"
authors: [David Wagner, Nicholas Carlini]
year: 2017
tags: [adversarial-machine-learning, evasion, robustness, week-3, optimization]
project: Adversarial Robustness Harness
---

# Paper Card: Towards Evaluating the Robustness of Neural Networks

**1. Full citation and link.**
> Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. *2017 IEEE Symposium on Security and Privacy (SP)*. [1608.04644](https://arxiv.org/abs/1608.04644)[cite: 2]

**2. What problem is studied?**
The paper studies the problem of evaluating the true adversarial robustness of deep neural networks, specifically demonstrating that a recently proposed defense mechanism called "defensive distillation" does not actually increase robustness. The authors investigate whether the perceived security of defensively distilled networks is due to genuine resilience or simply the failure of existing attack algorithms to find adversarial examples.

**3. What evidence or data is used?**
The authors evaluate their attacks on three standard image classification datasets: MNIST (digit recognition), CIFAR-10 (small image recognition), and ImageNet (large image recognition with 1,000 classes). They utilize standard convolutional architectures for MNIST and CIFAR-10, and a pre-trained Inception v3 network for ImageNet. 

**4. What method is used?**
The authors model adversarial example generation as an optimization problem designed to minimize the perturbation under three distance metrics ($L_0$, $L_2$, and $L_\infty$) while forcing misclassification. To solve the optimization efficiently, they:
*   Empirically evaluate seven different objective functions, finding that a margin-based function utilizing model logits performs best.
*   Introduce a change-of-variables method using the $\tanh$ function to natively handle box constraints (keeping pixel values between 0 and 1) without getting stuck during gradient descent.
*   Utilize the Adam optimizer to quickly navigate the optimization space.

**5. What are the two most important findings?**
*   **Finding 1:** Defensive distillation provides almost no security benefit against strong attacks; the newly developed $L_0$, $L_2$, and $L_\infty$ attacks found adversarial examples on distilled networks with a 100% success probability.
*   **Finding 2:** The choice of the objective function drastically impacts the efficacy of adversarial attacks. Standard loss functions like cross-entropy (recommended by prior work) cause gradients to vanish or perform in an overly greedy manner, making them a poor choice for optimization-based evasion. 

**6. What is one important limitation?**
The attacks rely heavily on a white-box threat model, assuming the adversary has complete access to the neural network's architecture and weights to compute gradients. Furthermore, finding optimized minimal perturbations (especially for the $L_0$ attack) is highly computationally expensive on large inputs like ImageNet.

**7. What will we use, change, test, or avoid because of this paper?**
> **Impact on our Adversarial Robustness Harness:** As we design the transformation taxonomy and budgeted combination search for our phishing detectors, we will **avoid** relying on simple, single-step attacks (like FGSM) to evaluate the robustness of our detector zoo, as this can lead us to "fool ourselves" into overestimating our security. We will **use** C&W's rigorous approach to defining perturbation budgets (similar to their precise use of $L_p$ norms) and **test** our models using an optimized, combination-based search strategy to ensure our fragility map measures the *actual* limits of the detector, not just the weakness of the attack[cite: 1, 2].