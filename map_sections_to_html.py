import json
import re


def load_json_file(filepath):
    """Load JSON file and return data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_for_section(section, html_pages, total_html_pages, total_pdf_pages):
    """
    Extract text content from HTML pages for a given PDF section.
    
    Maps PDF page numbers to HTML page numbers using proportional scaling.
    
    Args:
        section: Section dict with start_page and end_page (PDF page numbers)
        html_pages: Dict of HTML pages with text content
        total_html_pages: Total number of HTML pages
        total_pdf_pages: Total number of PDF pages
    
    Returns:
        Concatenated text from the corresponding HTML pages
    """
    pdf_start = section['start_page']
    pdf_end = section['end_page']
    
    # Map PDF pages to HTML pages using proportional scaling
    # Formula: html_page = (pdf_page / total_pdf_pages) * total_html_pages
    html_start = max(1, int((pdf_start / total_pdf_pages) * total_html_pages))
    html_end = min(total_html_pages, int((pdf_end / total_pdf_pages) * total_html_pages))
    
    # Collect text from HTML pages
    text_chunks = []
    for page_num in range(html_start, html_end + 1):
        page_key = str(page_num)
        if page_key in html_pages:
            text_chunks.append(html_pages[page_key])
    
    return {
        'text': ' '.join(text_chunks).strip(),
        'html_pages': f"{html_start}-{html_end}",
        'pdf_pages': f"{pdf_start}-{pdf_end}"
    }


def process_sections_recursive(sections, html_pages, total_html_pages, total_pdf_pages):
    """
    Recursively process sections and their subsections to add text content.
    
    Args:
        sections: List of section dicts (potentially with subsections)
        html_pages: Dict of HTML pages with text content
        total_html_pages: Total number of HTML pages
        total_pdf_pages: Total number of PDF pages
    
    Returns:
        List of sections with added text content
    """
    result = []
    
    for section in sections:
        # Extract text for this section
        text_data = extract_text_for_section(
            section, 
            html_pages, 
            total_html_pages, 
            total_pdf_pages
        )
        
        # Create new section with text
        new_section = {
            'level': section['level'],
            'title': section['title'],
            'original_title': section['original_title'],
            'start_page': section['start_page'],
            'end_page': section['end_page'],
            'html_pages': text_data['html_pages'],
            'text': text_data['text'],
            'text_length': len(text_data['text'])
        }
        
        # Process subsections recursively if they exist
        if 'subsections' in section and section['subsections']:
            new_section['subsections'] = process_sections_recursive(
                section['subsections'],
                html_pages,
                total_html_pages,
                total_pdf_pages
            )
        else:
            new_section['subsections'] = []
        
        result.append(new_section)
    
    return result


def map_sections_to_html_text(
    sections_json_path='output/parsed_sections.json',
    html_json_path='output/NYSE_MTX_2024.json',
    output_path='output/sections_with_text.json'
):
    """
    Map PDF sections to HTML text content and create enriched output.
    
    Args:
        sections_json_path: Path to parsed sections JSON (from PDF TOC)
        html_json_path: Path to HTML-based JSON with page text
        output_path: Path to save the enriched output
    """
    print(f"Loading sections from: {sections_json_path}")
    sections_data = load_json_file(sections_json_path)
    
    print(f"Loading HTML pages from: {html_json_path}")
    html_data = load_json_file(html_json_path)
    
    total_pdf_pages = sections_data['num_pages']
    total_html_pages = html_data['total_pages']
    html_pages = html_data['pages']
    
    print(f"\nPDF has {total_pdf_pages} pages")
    print(f"HTML has {total_html_pages} pages")
    print(f"Mapping ratio: {total_html_pages / total_pdf_pages:.2f} HTML pages per PDF page\n")
    
    # Process all sections recursively
    print("Processing sections...")
    enriched_sections = process_sections_recursive(
        sections_data['sections'],
        html_pages,
        total_html_pages,
        total_pdf_pages
    )
    
    # Create output structure
    output = {
        'filename': sections_data['filename'],
        'pdf_pages': total_pdf_pages,
        'html_pages': total_html_pages,
        'num_sections': sections_data['num_sections'],
        'sections': enriched_sections
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully mapped {sections_data['num_sections']} sections")
    print(f"Output saved to: {output_path}")
    
    # Print sample
    print("\nSample sections:")
    for i, section in enumerate(enriched_sections[:3]):
        print(f"\n{i+1}. {section['title']}")
        print(f"   PDF pages: {section['start_page']}-{section['end_page']}")
        print(f"   HTML pages: {section['html_pages']}")
        print(f"   Text length: {section['text_length']} characters")
        if section['text']:
            preview = section['text'][:100].replace('\n', ' ')
            print(f"   Preview: {preview}...")


if __name__ == "__main__":
    map_sections_to_html_text()
