#this will give you both the company name and its sic code
import re
import os
from groq import Groq
from typing import Dict, List, Set, Optional


#get the industry code through prompt
def get_sic_code(industry: str, api_key: str = None) -> str:
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Set it as environment variable or pass as parameter.")

    client = Groq(api_key=api_key)
    prompt = f"""Given the industry field: "{industry}"

Provide ONLY the 4-digit SIC (Standard Industrial Classification) code number.
The code must be a valid SIC code that appears in SEC EDGAR's classification system.
Return only the number, nothing else. No text, no explanation, just the 4-digit code."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in SEC EDGAR SIC codes. Return ONLY the 4-digit SIC code number that SEC EDGAR uses in its filings. Do not return codes that exist in general SIC manuals but are rarely or never used in EDGAR — prefer the parent or most commonly filed code. Return nothing else."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


#get company name through prompt
class CompanyDetector:

    def __init__(self, llm_fn=None):
        self._call_llm = llm_fn
        self.company_aliases: Set[str] = set()
        self.main_company: str = "the Company"

    def detect_from_sections(self, flat_sections: list) -> str:
        """Detect the main company name from the first pages of the document."""
        print("  Auto-detecting main company from parsed sections...")

        pages = []
        for section in flat_sections:
            for page in section.get('page_contents', []):
                pages.append(page['content'])
                if len(pages) >= 4:
                    break
            if len(pages) >= 4:
                break

        if not pages:
            raise RuntimeError("No page content found in sections; cannot detect company name.")

        if not self._call_llm:
            raise RuntimeError("No LLM function provided for company detection.")
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            raise RuntimeError("AWS_ACCESS_KEY_ID not set; cannot detect company name via LLM.")

        context = "\n\n---\n\n".join(pages)
        prompt = (
            "Read the following excerpts from an annual report and return ONLY "
            "the full legal name of the main company this report is about. "
            "No explanation, just the name.\n\n" + context
        )
        name = self._call_llm(prompt).strip().strip('"').strip("'")
        if not name or len(name) <= 3:
            raise RuntimeError(f"LLM returned an unusable company name: {name!r}")

        print(f"  ✓ Detected main company: {name}")
        self.company_aliases = {name}
        self.main_company = name
        return name

#class for sicode
class SICLookup:

    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}

    def lookup(self, sector: str) -> Optional[str]:
        if not sector:
            return None

        key = sector.strip().lower()
        if key not in self._cache:
            try:
                code = get_sic_code(sector)
                self._cache[key] = str(code).strip()
                print(f"    ✓ SIC lookup: '{sector}' → {self._cache[key]}")
            except Exception as e:
                print(f"    ⚠ SIC lookup failed for '{sector}': {e}")
                self._cache[key] = None

        return self._cache[key]
