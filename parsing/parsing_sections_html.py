import argparse
import json
import re
import os
from typing import Dict, List
from bs4 import BeautifulSoup, NavigableString


def extract_text_with_markers(html_path: str) -> Dict:
    """
    Extract text content from HTML using actual page breaks (BRPFPageHeader divs).
    """
    print(f"Parsing HTML: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove hidden XBRL data
    for hidden in soup.find_all(id='DSPFiXBRLHidden'):
        hidden.decompose()
    
    # Get the main content div
    main_div = soup.find('div', style=lambda x: x and 'line-height: initial' in x)
    
    if not main_div:
        print("No main content div found")
        return {"pages": [], "total_pages": 0}
    
    # Find all page break markers (BRPFPageHeader divs)
    page_markers = main_div.find_all('div', class_='BRPFPageHeader')
    
    if not page_markers:
        print("No page markers found, falling back to word-based pagination")
        # Fallback to word-based pagination
        all_text = main_div.get_text(separator=' ', strip=True)
        words = all_text.split()
        WORDS_PER_PAGE = 3000
        pages = []
        page_num = 1
        
        for i in range(0, len(words), WORDS_PER_PAGE):
            page_words = words[i:i + WORDS_PER_PAGE]
            page_text = ' '.join(page_words)
            pages.append({
                'page': page_num,
                'text': page_text,
                'text_length': len(page_text),
                'word_count': len(page_words),
                'sections': []
            })
            page_num += 1
        
        return {
            "source_html": html_path,
            "total_pages": len(pages),
            "pages": pages
        }
    
    print(f"Found {len(page_markers)} page markers")
    
    # Build a list of all elements with their page numbers
    # Strategy: find parent of page markers and split content by markers
    pages = []
    
    # Get all elements and assign page numbers
    current_page = 0  # Start at 0, will increment when we hit first marker
    page_content = []
    
    for element in main_div.descendants:
        # Check if this is a page marker
        if hasattr(element, 'get') and element.get('class') == ['BRPFPageHeader']:
            # Save previous page if we have content
            if page_content and current_page > 0:
                page_text = ' '.join(page_content)
                if page_text.strip():
                    pages.append({
                        'page': current_page,
                        'text': page_text,
                        'text_length': len(page_text),
                        'word_count': len(page_text.split()),
                        'sections': []
                    })
            # Start new page
            current_page += 1
            page_content = []
        elif isinstance(element, NavigableString):
            text = str(element).strip()
            if text and text != '\xa0':
                page_content.append(text)
    
    # Add last page
    if page_content and current_page > 0:
        page_text = ' '.join(page_content)
        if page_text.strip():
            pages.append({
                'page': current_page,
                'text': page_text,
                'text_length': len(page_text),
                'word_count': len(page_text.split()),
                'sections': []
            })
    
    print(f"Extracted {len(pages)} pages with content")
    
    return {
        "source_html": html_path,
        "total_pages": len(pages),
        "pages": pages
    }


def detect_sections_from_html(html_path: str, pages_data: Dict) -> List[Dict]:
    """
    Detect sections from HTML structure using actual page numbers.
    Looks for PART and Item markers that indicate section boundaries.
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove hidden XBRL data
    for hidden in soup.find_all(id='DSPFiXBRLHidden'):
        hidden.decompose()
    
    # Get the main content div
    main_div = soup.find('div', style=lambda x: x and 'line-height: initial' in x)
    if not main_div:
        return []
    
    sections = []
    
    # Patterns for section headers
    part_pattern = re.compile(r'^PART\s+([IVX]+)', re.IGNORECASE)
    item_pattern = re.compile(r'^Item\s+(\d+[A-Z]?)\.?\s*(.*)', re.IGNORECASE)
    
    # Find all page markers to track which page we're on
    page_markers = main_div.find_all('div', class_='BRPFPageHeader')
    
    if not page_markers:
        # Fallback to word-based page tracking
        cumulative_words = 0
        WORDS_PER_PAGE = 3000
        
        for tag in soup.find_all(['div', 'p']):
            text = tag.get_text(strip=True)
            tag_words = len(text.split())
            cumulative_words += tag_words
            current_page = (cumulative_words // WORDS_PER_PAGE) + 1

            # Skip TOC entries and parent/child duplicates
            if tag.find('a', href=True):
                continue
            child_texts = {child.get_text(strip=True) for child in tag.find_all(['div', 'p'])}
            if text in child_texts:
                continue

            # Check for section headers
            if part_pattern.match(text):
                sections.append({
                    'title': text,
                    'original_title': text,
                    'level': 1,
                    'start_page': current_page,
                    'end_page': current_page,
                })
            elif item_pattern.match(text):
                style = tag.get('style', '')
                if 'font-weight: bold' in style or tag.find('strong') or tag.find('b'):
                    sections.append({
                        'title': text,
                        'original_title': text,
                        'level': 2,
                        'start_page': current_page,
                        'end_page': current_page,
                    })
    else:
        # Use actual page markers
        # Build a map of elements to page numbers
        element_to_page = {}
        current_page = 0  # Start at 0, will increment when we hit first marker

        for element in main_div.descendants:
            if hasattr(element, 'get') and element.get('class') == ['BRPFPageHeader']:
                current_page += 1
            if current_page > 0:  # Only assign page numbers after first marker
                element_to_page[id(element)] = current_page

        # Find the Table of Contents page to exclude its entries
        toc_page = None
        toc_pattern = re.compile(r'table\s+of\s+contents', re.IGNORECASE)
        for tag in main_div.find_all(['div', 'p']):
            if toc_pattern.search(tag.get_text(strip=True)):
                toc_page = element_to_page.get(id(tag))
                if toc_page:
                    break

        # Now find section headers and assign page numbers
        for tag in main_div.find_all(['div', 'p']):
            text = tag.get_text(strip=True)
            page_num = element_to_page.get(id(tag), 1)

            # Skip entries on the Table of Contents page
            if toc_page and page_num == toc_page:
                continue

            # Skip parent elements whose text is produced entirely by a child
            # that would also match (avoids double-matching parent div + child p)
            child_texts = {child.get_text(strip=True) for child in tag.find_all(['div', 'p'])}
            if text in child_texts:
                continue

            # Check for PART headers
            if part_pattern.match(text):
                sections.append({
                    'title': text,
                    'original_title': text,
                    'level': 1,
                    'start_page': page_num,
                    'end_page': page_num,
                })
                continue

            # Check for Item headers
            if item_pattern.match(text):
                style = tag.get('style', '')
                if 'font-weight: bold' in style or tag.find('strong') or tag.find('b'):
                    sections.append({
                        'title': text,
                        'original_title': text,
                        'level': 2,
                        'start_page': page_num,
                        'end_page': page_num,
                    })
    
    # Update end pages for each section
    total_pages = pages_data.get('total_pages', 1)
    
    for i, section in enumerate(sections):
        if i < len(sections) - 1:
            section['end_page'] = sections[i + 1]['start_page'] - 1
        else:
            section['end_page'] = total_pages
    
    print(f"Detected {len(sections)} sections across {total_pages} pages")
    
    return sections


def build_hierarchy(sections: List[Dict]) -> List[Dict]:
    """
    Build hierarchical structure from flat list of sections.
    """
    if not sections:
        return []
    
    root = []
    stack = []
    
    for section in sections:
        level = section["level"]
        
        # Pop stack until we find the parent level
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        
        # Add subsections list if not present
        if "subsections" not in section:
            section["subsections"] = []
        
        # Add to parent's subsections or root
        if stack:
            parent = stack[-1]
            parent["subsections"].append(section)
        else:
            root.append(section)
        
        # Push current section to stack
        stack.append(section)
    
    return root


def fix_parent_page_ranges(sections: List[Dict]) -> List[Dict]:
    """
    Fix parent section page ranges to match their subsections.
    Parent sections should span from the first subsection's start_page
    to the last subsection's end_page.
    """
    for section in sections:
        if section.get('subsections'):
            # Recursively fix subsections first
            section['subsections'] = fix_parent_page_ranges(section['subsections'])
            
            # Get page range from subsections
            subsection_pages = []
            for subsection in section['subsections']:
                subsection_pages.append(subsection['start_page'])
                subsection_pages.append(subsection['end_page'])
            
            if subsection_pages:
                # Update parent to span all subsections
                section['start_page'] = min(subsection_pages)
                section['end_page'] = max(subsection_pages)
    
    return sections


def add_text_to_sections(sections: List[Dict], pages_data: Dict) -> List[Dict]:
    """
    Populate section text from page data.
    """
    pages_by_num = {p['page']: p for p in pages_data['pages']}
    
    for section in sections:
        section_text = ""
        page_contents = []
        
        for page_num in range(section['start_page'], section['end_page'] + 1):
            page_data = pages_by_num.get(page_num)
            if page_data and page_data.get('text'):
                page_text = page_data['text']
                section_text += page_text + "\n"
                page_contents.append({
                    'page_number': page_num,
                    'content': page_text,
                    'sections': [],
                })
        
        section['text'] = section_text.strip()
        section['text_length'] = len(section_text)
        section['word_count'] = len(section_text.split())
        if page_contents:
            section['page_contents'] = page_contents
        
        if section.get('subsections'):
            section['subsections'] = add_text_to_sections(
                section['subsections'], pages_data
            )
    
    return sections


def sections_parser_html(html_path: str, output_path: str = None) -> Dict:
    """
    Parse an HTML file and extract sections.
    
    Args:
        html_path: Path to the HTML file
        output_path: Optional path to save the output (JSON format)
    
    Returns:
        Dictionary containing parsed sections
    """
    try:
        print(f"Parsing HTML: {html_path}")
        
        # First extract pages
        pages_data = extract_text_with_markers(html_path)
        sh
        # Detect sections from HTML structure
        flat_sections = detect_sections_from_html(html_path, pages_data)
        
        if not flat_sections:
            print("No sections found in HTML")
            return None
        
        print(f"Found {len(flat_sections)} sections")
        
        # Build hierarchical structure
        hierarchical_sections = build_hierarchy(flat_sections)
        
        # Fix parent section page ranges based on subsections
        hierarchical_sections = fix_parent_page_ranges(hierarchical_sections)
        
        # Add text content to sections
        hierarchical_sections = add_text_to_sections(hierarchical_sections, pages_data)
        flat_sections = []
        
        def flatten(secs):
            for s in secs:
                flat_sections.append(s)
                if s.get('subsections'):
                    flatten(s['subsections'])
        
        flatten(hierarchical_sections)
        
        result = {
            "filename": html_path,
            "source_html": html_path,
            "num_pages": pages_data['total_pages'],
            "num_sections": len(flat_sections),
            "sections": hierarchical_sections,
            "flat_sections": flat_sections,
        }
        
        # Save to file if output path provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            serializable = {k: v for k, v in result.items() if k != "flat_sections"}
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Output saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_page_index_from_html(html_path: str, output_path: str = None) -> Dict:
    """
    Extract per-page text from HTML and save a page index JSON.
    """
    print(f"Generating page index from HTML: {html_path}")
    
    result = extract_text_with_markers(html_path)
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Page index saved to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse HTML files and extract sections with page numbers"
    )
    parser.add_argument(
        "html_path",
        nargs='?',
        help="Path to the HTML file to parse",
        default="input/NYSE_MTX_2024.htm"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)",
        default="output/parsed_sections_html.json"
    )
    parser.add_argument(
        "--page-index", "-p",
        help="Output path for page index JSON",
        default="output/page_index_html.json"
    )

    args = parser.parse_args()

    print(f"Parsing HTML: {args.html_path}\n")

    # Generate page index first
    print("Generating page index from HTML...")
    page_index = generate_page_index_from_html(args.html_path, args.page_index)
    
    if page_index:
        print(f"✓ Extracted {page_index['total_pages']} logical pages")
    
    # Parse sections
    result = sections_parser_html(args.html_path, args.output)

    if result:
        print(f"\n✓ Successfully extracted {result['num_sections']} sections")


if __name__ == "__main__":
    main()
