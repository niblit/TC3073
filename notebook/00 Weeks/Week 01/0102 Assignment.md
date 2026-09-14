- Read the Week 1 paper by Goodfellow et al. and complete one paper card.

[[01 - EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES]]

---

- Write the threat model in three sentences: attacker knowledge, allowed transformations, and the success criterion.


	- **Attacker Knowledge:** The attacker operates under a realistic, limited-knowledge or black-box assumption, lacking direct access to the detector's internal weights, gradients, or training data.
	
	
	- **Allowed Transformations:** The attacker applies problem-space transformations to the email or SMS—specifically content paraphrasing, structural obfuscation, and channel-metadata manipulation—strictly within an explicit perturbation budget to remain realistic.
	
	
	- **Success Criterion:** The evasion attack succeeds if the modified message bypasses the target detector while preserving its original malicious intent, meaning the message remains functional and persuasive enough that a victim would act on it.
	
---

- Take two safe sample attacks and hand-craft one intent-preserving paraphrase of each.

	**Sample Attack 1: Business Email Compromise (BEC) - Urgent Payment**
	
	- **Original:** "Hi, I am stuck in a meeting right now. Please process this wire transfer of $50,000 to our new vendor immediately to avoid late fees. Account details are attached."
	    
	- **Paraphrase:** "Hello, I'm tied up in conference calls all morning. We need to route a 50k payment to the updated supplier account right away so we don't incur penalties; please see the attached document for the routing info."
	    
	
	**Sample Attack 2: Phishing - Credential Harvesting**
	
	- **Original:** "Your account password will expire in 24 hours. Click the secure link below to verify your credentials and maintain access to your email."
	    
	- **Paraphrase:** "Action required: your current login token is set to expire by tomorrow. Please follow the provided URL to validate your login details and ensure uninterrupted service."


---

- List two ways a detector could look accurate for the wrong reason (two plausible shortcuts).

	- **Over-reliance on Specific Vocabulary:** A model might simply memorize a list of "malicious" keywords (e.g., "wire transfer," "click here," "urgent") from its training data. It looks highly accurate on the test set, but breaks completely when an attacker substitutes those exact terms with intent-preserving synonyms (e.g., "route payment," "follow the URL," "time-sensitive").
	    
	- **Overfitting to Structural Artifacts:** A detector might learn to classify an email as malicious based purely on formatting quirks—such as specific HTML tags, strange capitalization, or poor spelling that happened to be prevalent in the training data's attack class. If an attacker cleans up the HTML structure and fixes the typos while leaving the core malicious request intact, the detector will likely classify it as safe.


---

