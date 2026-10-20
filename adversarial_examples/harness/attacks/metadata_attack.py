import re
from harness.taxonomy import MetadataTransformation
from harness.budget import PerturbationBudget

class TimestampSpoofing(MetadataTransformation):
    """
    Alters delivery metadata (e.g., email Date headers) to manipulate 
    temporal analysis or timestamp-based heuristics in detection engines.
    """
    def __init__(self, name: str = "timestamp_spoofing", target_date: str = "Thu, 01 Jan 1970 00:00:00 +0000"):
        super().__init__(name)
        self.target_date = target_date

    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        # Modifying a header costs 1 edit
        budget.consume(1) 

        fake_header = f"Date: {self.target_date}\n"
        
        # If a Date header already exists, replace it
        if re.search(r"^Date:.*$", text, flags=re.MULTILINE):
            transformed_text = re.sub(
                r"^Date:.*$", 
                fake_header.strip(), 
                text, 
                count=1, 
                flags=re.MULTILINE
            )
        else:
            # Otherwise, prepend the new header (simulating raw email/HTTP structure)
            transformed_text = fake_header + text
            
        return transformed_text
