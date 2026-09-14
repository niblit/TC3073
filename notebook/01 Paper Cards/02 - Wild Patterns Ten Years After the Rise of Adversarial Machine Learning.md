- Full citation and link.

> **Citation:** Biggio, B., & Roli, F. (2018). Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning. [1712.03141v2](https://arxiv.org/abs/1712.03141).

- What problem is studied?

The paper reviews the evolution of adversarial machine learning over the past decade. It connects early research on the security of non-deep learning algorithms to modern deep learning vulnerabilities, addressing common misconceptions about how security should be evaluated.

- What evidence or data is used?

The authors synthesize historical literature and present empirical application examples. Specific use cases explored include spam filtering, PDF malware detection, MNIST digit classification, and the iCub humanoid robot vision system.

- What method is used?

The authors systematize the field by proposing a proactive "security-by-design" cycle. They introduce a comprehensive threat model framework that categorizes attacks based on the attacker's goal, knowledge (e.g., perfect vs. zero-knowledge), and capability (e.g., evasion vs. poisoning).

- What are the two most important findings?

	- Evasion attacks and adversarial machine learning predated the deep learning boom, originating with early spam and malware filters.

	- Robustness should be measured using "security evaluation curves" against maximum-confidence attacks, rather than relying solely on minimally perturbed adversarial examples.

- What is one important limitation?

Existing proactive defenses struggle to account for "unknown unknowns" when deployed in open-world adversarial environments, as threat modeling inherently relies on predicting known attacker behaviors.

- What will we use, change, test, or avoid because of this paper?

Following the project guidelines, we will use this framework to firmly establish test-time evasion as our core security threat. We will avoid the pitfall of using exclusively weak or minimally perturbed attacks, and instead build security evaluation degradation curves to measure how detectors break under specific perturbation budgets.


---
## Notes

### The Threat Model Framework

- **Goal:** What does the attacker want? They might target system integrity (evading detection), availability (causing maximum misclassifications), or privacy (reverse-engineering the model).
    
- **Knowledge:** How much do they know? This ranges from perfect knowledge (a "white-box" attack) to zero knowledge (a "black-box" attack).
    
- **Capability:** When do they attack? They can manipulate data during the testing phase (exploratory) or the training phase (causative).


### Evasion vs. Poisoning (The Core Attacks)

- **Evasion Attacks:** This happens at _test time_. The attacker manipulates input data (like altering an image's pixels) to bypass a trained classifier without actually touching the model itself.
    
- **Poisoning Attacks:** This happens at _training time_. The attacker injects a small fraction of malicious samples into the training data so the model learns the wrong boundaries from the start.

