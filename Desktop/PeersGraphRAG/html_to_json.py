import json
import re
from bs4 import BeautifulSoup

def html_to_paged_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Split the body on page-break <hr> tags
    body = soup.find("body")
    pages = {}
    current_page = 1
    current_chunks = []

    for element in body.children:
        tag = getattr(element, "name", None)
        style = element.get("style", "") if tag else ""

        if tag == "hr" and "page-break" in style:
            text = " ".join(current_chunks).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                pages[f"{current_page}"] = text
            current_page += 1
            current_chunks = []
        else:
            chunk = element.get_text(separator=" ", strip=True) if tag else str(element).strip()
            if chunk:
                current_chunks.append(chunk)

    # Last page (no trailing <hr>)
    text = " ".join(current_chunks).strip()
    text = re.sub(r"\s+", " ", text)
    if text:
        pages[f"{current_page}"] = text

    # If only one page found, try alternative splitting method
    if len(pages) == 1:
        print(f"Only 1 page found with page-break method. Trying alternative splitting...")
        pages = split_by_chunks(text)

    result = {
        "source": input_path,
        "total_pages": len(pages),
        "pages": pages
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"{input_path} -> {output_path} ({len(pages)} pages)")


def split_by_chunks(text, chunk_size=5000):
    """
    Split text into chunks of approximately chunk_size characters.
    Tries to break at sentence boundaries.
    """
    pages = {}
    words = text.split()
    current_chunk = []
    current_length = 0
    page_num = 1
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            pages[f"{page_num}"] = chunk_text
            page_num += 1
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        pages[f"{page_num}"] = chunk_text
    
    return pages

html_to_paged_json("input/NYSE_MTX_2024.htm",  "output/NYSE_MTX_2024.json")
