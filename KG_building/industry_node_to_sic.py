"""
Industry to SIC Code Converter using Groq API
Converts industry names/fields to their corresponding SIC (Standard Industrial Classification) codes
"""

import os
from groq import Groq


def get_sic_code(industry: str, api_key: str = None) -> dict:
    """
    Get SIC code for a given industry using Groq API
    
    Args:
        industry: Industry name or field (e.g., "finance", "healthcare", "manufacturing")
        api_key: Groq API key (optional, will use GROQ_API_KEY env var if not provided)
    
    Returns:
        str: The 4-digit SIC code
    """
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Set it as environment variable or pass as parameter.")
    
    client = Groq(api_key=api_key)
    
    prompt = f"""Given the industry field: "{industry}"

Provide ONLY the 4-digit SIC (Standard Industrial Classification) code number. 
Return only the number, nothing else. No text, no explanation, just the 4-digit code."""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in SIC codes. Return ONLY the 4-digit SIC code number, nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=10
    )
    
    sic_code = response.choices[0].message.content.strip()
    
    return sic_code


def main():
    """Example usage"""
    # Example industries
    industries = [
        "finance",
        "healthcare",
        "technology",
        "retail",
        "manufacturing"
    ]
    
    print("Industry to SIC Code Converter\n" + "="*50)
    
    for industry in industries:
        try:
            sic_code = get_sic_code(industry)
            print(f"{industry}: {sic_code}")
        except Exception as e:
            print(f"{industry}: Error - {str(e)}")
    

if __name__ == "__main__":
    main()
