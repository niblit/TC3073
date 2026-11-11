---
title: "Dos and Don'ts of Machine Learning in Computer Security"
authors: [Daniel Arp, Erwin Quiring, Feargus Pendlebury, Alexander Warnecke, Fabio Pierazzi, Christian Wressnegger, Lorenzo Cavallaro, Konrad Rieck]
year: 2022
tags: [adversarial-ml, evaluation-methodology, security-pitfalls, usenix2022]
project: "Adversarial Robustness Harness"
---

# Paper Card: Dos and Don'ts of Machine Learning in Computer Security

**1. Full citation and link.**
> Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., & Rieck, K. (2022). Dos and Don'ts of Machine Learning in Computer Security. *31st USENIX Security Symposium*, 3971–3988. https://www.usenix.org/conference/usenixsecurity22/presentation/arp

**2. What problem is studied?**
- **Methodological Flaws in ML Security:** The paper investigates ten common but subtle pitfalls in the design, implementation, and evaluation of learning-based security systems that lead to over-optimistic performance estimations and impractical real-world deployments.

**3. What evidence or data is used?**
- **Literature Review:** 30 papers from top-tier security conferences (ACM CCS, IEEE S&P, USENIX Security, NDSS) over the past decade.
- **Author Survey:** Feedback from 49 authors of the evaluated papers regarding their awareness and agreement with the identified pitfalls.
- **Empirical Case Studies:** Four replication studies on Android malware (AndroZoo/DREBIN), vulnerability discovery (VulDeePecker/SARD), authorship attribution (Google Code Jam), and network intrusion detection (Mirai/KITSUNE).

**4. What method is used?**
- **Prevalence Analysis:** Two independent reviewers assessed each of the 30 papers for the ten pitfalls, using Krippendorff's alpha to ensure high inter-rater reliability.
- **Impact Analysis:** The researchers reproduced state-of-the-art models and applied rigorous experimental controls (e.g., separating Chinese vs. GooglePlay malware sources, removing unused code templates) to quantify how much performance degraded when pitfalls were neutralized.

**5. What are the two most important findings?**
- **Endemic Methodological Failures:** Every single one of the 30 analyzed top-tier papers suffered from at least three pitfalls, with "Sampling Bias" and "Data Snooping" present in 90% and 73% of papers, respectively.
- **Severe Performance Inflation:** When methodological flaws are corrected, state-of-the-art performance collapses. For example, correcting for unused code templates (spurious correlations) in authorship attribution caused a 48% drop in test accuracy.

**6. What is one important limitation?**
- **Conservative Pitfall Assessment & Unavoidable Biases:** The prevalence analysis gave authors the benefit of the doubt; if mitigation wasn't clearly stated in the text, it was assumed absent, meaning actual pitfall rates might be higher. Furthermore, some pitfalls (like sampling bias in malware collection or lab-only evaluations) are inherently unavoidable due to ethical constraints or lack of ground-truth distributions.

**7. What will we use, change, test, or avoid because of this paper?**
> **For the Adversarial Robustness Harness:**
> - **Test:** We will actively test the "Detector Zoo" for **Spurious Correlations (P4)**. Our intent-preserving transformations must verify if detectors are learning actual semantic threats or just dataset artifacts.
> - **Avoid:** We will strictly prevent **Biased Parameter Selection (P5)** and **Data Snooping (P3)**. Perturbation budgets and evasion thresholds will be calibrated on a strictly isolated validation set, not the test set.
> - **Use:** We will implement an **Appropriate Threat Model (P10)**. Instead of standard black-box metrics, our harness will assume an adaptive adversary leveraging white-box evasion techniques.
