---
title: "Real Attackers Don't Compute Gradients: Bridging the Gap Between Adversarial ML Research and Practice"
authors: [Giovanni Apruzzese, Hyrum S. Anderson, Savino Dambra, David Freeman, Fabio Pierazzi, Kevin Roundy]
year: 2022
tags: [adversarial-ml, evasion-engine, threat-modeling, ml-systems, intent-preserving, industry-practice]
project: "Adversarial Robustness Harness"
---

# Paper Card: "Real Attackers Don't Compute Gradients"

**1. Full citation and link.**
> **Citation:** Apruzzese, G., Anderson, H. S., Dambra, S., Freeman, D., Pierazzi, F., & Roundy, K. (2022). "Real Attackers Don't Compute Gradients": Bridging the Gap Between Adversarial ML Research and Practice. *University of Liechtenstein, Robust Intelligence, Norton Research Group, Meta, King's College London*. 
> **Link:** (PDF Reference) 08 - “Real Attackers Don’t Compute Gradients”: Bridging the Gap Between Adversarial ML Research and Practice - real_attackers.pdf

---

**2. What problem is studied?**
- **The Research-Practice Disconnect:** There is a significant gap between adversarial ML research and operational security practice. 
- **Misaligned Priorities:** Researchers focus heavily on algorithmic, gradient-based evasion attacks against isolated machine learning models. In contrast, industry practitioners do not prioritize these defenses because actual attackers use simple, heuristic tactics (like domain-specific obfuscation) to bypass entire ML pipelines.

---

**3. What evidence or data is used?**
- **Systematic Literature Review:** An analysis of 88 adversarial ML papers published in top-4 security conferences (CCS, SEC, NDSS, SP) between 2019 and 2021.
- **Industry Case Studies:** Three real-world scenarios: 
  - Facebook's multi-layered spam detection pipeline ("The four A's of abuse fighting").
  - A commercial phishing detector's triage logs of 100 evasive webpages (analyzing over 4,600 samples).
  - Submission and query logs from the 42-day 2021 Machine Learning Security Evasion Competition (MLSEC).

---

**4. What method is used?**
- **Qualitative Forensic Analysis:** Visual inspection and categorization of real-world evasive phishing webpages to identify non-gradient evasion techniques (e.g., cropping, blurring, masking).
- **Temporal/Behavioral Analysis:** Examining the submission timelines and API query counts of the top four MLSEC teams to deduce the actual human cost and domain expertise required for successful evasion.
- **Structured Literature Classification:** Categorizing 88 academic papers by attack family, ML paradigm (deep vs. shallow learning), data type, system complexity (model vs. pipeline), and threat model consistency.

---

**5. What are the two most important findings?**
- **Finding 1: Real attackers rely on cheap, domain-specific heuristics, not gradients.** The commercial phishing logs showed attackers successfully evade ML by using simple intent-preserving tactics—such as cropping, company name obfuscation, masking, and stretching—which break Optical Character Recognition (OCR) rather than exploiting neural network gradients.
- **Finding 2: Complex ML systems require holistic threat models.** Real ML deployments are complex pipelines, not standalone models. Attackers must bypass multiple defense layers (e.g., Automation, Access, Activity, Application layers at Facebook). Furthermore, literature evaluations often ignore this: 80% of reviewed papers evaluated only single models rather than complex pipelines, and 89% focused exclusively on deep learning despite shallow learning's prevalence in practice. 

---

**6. What is one important limitation?**
- **The "Probatio Diabolica" of Gradient Usage:** The authors admit it is exceedingly difficult to definitively prove whether a wild adversarial sample was generated via gradient-based algorithms or just educated guessing. Without asking the attacker directly, forensic deduction is the only available, albeit imperfect, method.

---

**7. What will we use, change, test, or avoid because of this paper?**
> **Application to the Adversarial Robustness Harness Project:**
> - **Change our Scope:** We must evaluate our detector zoo against the *entire processing pipeline* (including input preprocessing, OCR, feature extraction), rather than passing mathematical perturbations directly to the model's tensor inputs.
> - **Test New Perturbations:** We will prioritize intent-preserving, domain-specific transformations (e.g., HTML DOM manipulation, image blurring, spatial cropping) under our perturbation budget, rather than strictly relying on standard $L_p$-norm pixel noise.
> - **Update Metrics:** We will modify our harness's efficiency tracking. Query count alone is misleading; we should incorporate human/compute time estimates to accurately reflect attacker cost-benefit economics.
