import argparse
import json
import re
import os
from typing import Dict, List
from bs4 import BeautifulSoup, NavigableString


def table_to_text(table) -> str:
    rows = []
    for tr in table.find_all('tr'):
        cells = []
        for cell in tr.find_all(['td', 'th']):
            text = re.sub(r'\s+', ' ', cell.get_text(separator=' ', strip=True))
            colspan = int(cell.get('colspan', 1))
            cells.extend([text] + [''] * (colspan - 1))
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    num_cols = max(len(row) for row in rows)
    col_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    formatted_rows = []
    for i, row in enumerate(rows):
        while len(row) < num_cols:
            row.append('')
        formatted_cells = []
        for j, cell in enumerate(row):
            if j < num_cols:
                if cell and re.match(r'^[\d,.$%()-]+$', cell.replace(' ', '')):
                    formatted_cells.append(cell.rjust(col_widths[j]))
                else:
                    formatted_cells.append(cell.ljust(col_widths[j]))
        formatted_rows.append('| ' + ' | '.join(formatted_cells) + ' |')
        if i == 0:
            formatted_rows.append('|' + '|'.join(['-' * (w + 2) for w in col_widths]) + '|')

    return '[TABLE START]\n' + '\n'.join(formatted_rows) + '\n[TABLE END]'


def _is_page_break_hr(tag) -> bool:
    """Return True if tag is an <hr> that acts as a page break."""
    if not hasattr(tag, 'name') or tag.name != 'hr':
        return False
    style = tag.get('style', '')
    return 'page-break-after' in style or 'page-break-before' in style


def extract_text_with_markers(html_path: str) -> Dict:
    """
    Extract text content from HTML using actual page breaks (BRPFPageHeader divs
    or <hr page-break-after:always> elements in newer HTML formats).
    """

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove hidden XBRL data
    for hidden in soup.find_all(id='DSPFiXBRLHidden'):
        hidden.decompose()

    # Get the main content div (older BRPF format)
    main_div = soup.find('div', style=lambda x: x and 'line-height: initial' in x)

    if not main_div:
        # Newer format: content is directly in body
        main_div = soup.find('body')

    if not main_div:
        print("No main content div found")
        return {"pages": [], "total_pages": 0}
    
    # Find all page break markers — BRPF format or hr-based format
    page_markers = main_div.find_all('div', class_='BRPFPageHeader')
    hr_page_breaks = main_div.find_all('hr', style=lambda x: x and 'page-break-after' in x) if not page_markers else []

    if not page_markers and not hr_page_breaks:
        print("No page markers found, falling back to word-based pagination")
        chunks = []
        processed_tables: set = set()
        for element in main_div.descendants:
            if hasattr(element, 'name') and element.name == 'table':
                table_id = id(element)
                if table_id not in processed_tables and element.parent.name != 'table':
                    table_text = table_to_text(element)
                    if table_text:
                        chunks.append('\n' + table_text + '\n')
                    processed_tables.add(table_id)
                    for desc in element.descendants:
                        if hasattr(desc, 'name'):
                            processed_tables.add(id(desc))
            elif isinstance(element, NavigableString):
                parent_table = element.find_parent('table')
                if parent_table and id(parent_table) in processed_tables:
                    continue
                text = str(element).strip()
                if text and text != '\xa0':
                    chunks.append(re.sub(r'\s+', ' ', text))

        all_text = '\n\n'.join(chunks)
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
            })
            page_num += 1

        return {
            "source_html": html_path,
            "total_pages": len(pages),
            "pages": pages
        }

    if hr_page_breaks:
        # Newer format: <hr page-break-after:always> separates pages
        pages = []
        current_page = 1
        page_chunks = []
        processed_tables: set = set()

        for element in main_div.descendants:
            if _is_page_break_hr(element):
                page_text = re.sub(r'\n{3,}', '\n\n', '\n\n'.join(page_chunks)).strip()
                if page_text:
                    pages.append({
                        'page': current_page,
                        'text': page_text,
                        'text_length': len(page_text),
                        'word_count': len(page_text.split()),
                    })
                current_page += 1
                page_chunks = []
                processed_tables = set()
            elif hasattr(element, 'name') and element.name == 'table':
                table_id = id(element)
                if table_id not in processed_tables and element.parent.name != 'table':
                    table_text = table_to_text(element)
                    if table_text:
                        page_chunks.append('\n' + table_text + '\n')
                    processed_tables.add(table_id)
                    for desc in element.descendants:
                        if hasattr(desc, 'name'):
                            processed_tables.add(id(desc))
            elif isinstance(element, NavigableString):
                parent_table = element.find_parent('table')
                if parent_table and id(parent_table) in processed_tables:
                    continue
                text = str(element).strip()
                if text and text != '\xa0':
                    page_chunks.append(re.sub(r'\s+', ' ', text))

        # Add last page
        if page_chunks:
            page_text = re.sub(r'\n{3,}', '\n\n', '\n\n'.join(page_chunks)).strip()
            if page_text:
                pages.append({
                    'page': current_page,
                    'text': page_text,
                    'text_length': len(page_text),
                    'word_count': len(page_text.split()),
                })

        print(f"Extracted {len(pages)} pages with content (hr-based)")
        return {
            "source_html": html_path,
            "total_pages": len(pages),
            "pages": pages
        }


    pages = []
    current_page = 0
    page_chunks = []
    processed_tables = set()

    for element in main_div.descendants:
        if hasattr(element, 'get') and element.get('class') == ['BRPFPageHeader']:
            if page_chunks and current_page > 0:
                page_text = re.sub(r'\n{3,}', '\n\n', '\n\n'.join(page_chunks)).strip()
                if page_text:
                    pages.append({
                        'page': current_page,
                        'text': page_text,
                        'text_length': len(page_text),
                        'word_count': len(page_text.split()),
                    })
            current_page += 1
            page_chunks = []
            processed_tables = set()
        elif hasattr(element, 'name') and element.name == 'table':
            table_id = id(element)
            if table_id not in processed_tables and element.parent.name != 'table':
                table_text = table_to_text(element)
                if table_text:
                    page_chunks.append('\n' + table_text + '\n')
                processed_tables.add(table_id)
                for desc in element.descendants:
                    if hasattr(desc, 'name'):
                        processed_tables.add(id(desc))
        elif isinstance(element, NavigableString):
            parent_table = element.find_parent('table')
            if parent_table and id(parent_table) in processed_tables:
                continue
            text = str(element).strip()
            if text and text != '\xa0':
                page_chunks.append(re.sub(r'\s+', ' ', text))

    # Add last page
    if page_chunks and current_page > 0:
        page_text = re.sub(r'\n{3,}', '\n\n', '\n\n'.join(page_chunks)).strip()
        if page_text:
            pages.append({
                'page': current_page,
                'text': page_text,
                'text_length': len(page_text),
                'word_count': len(page_text.split()),
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
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove hidden XBRL data
    for hidden in soup.find_all(id='DSPFiXBRLHidden'):
        hidden.decompose()
    
    # Get the main content div (older BRPF format), fall back to body for newer format
    main_div = soup.find('div', style=lambda x: x and 'line-height: initial' in x)
    if not main_div:
        main_div = soup.find('body')
    if not main_div:
        return []

    sections = []

    # Patterns for section headers
    part_pattern = re.compile(r'^PART\s+([IVX]+)', re.IGNORECASE)
    item_pattern = re.compile(r'^Item\s+(\d+[A-Z]?)\.?\s*(.*)', re.IGNORECASE)

    # Find all page markers to track which page we're on
    page_markers = main_div.find_all('div', class_='BRPFPageHeader')
    hr_page_breaks = main_div.find_all('hr', style=lambda x: x and 'page-break-after' in x) if not page_markers else []

    if not page_markers and hr_page_breaks:
        # Newer format: build element→page map using <hr page-break-after> as separators
        element_to_page = {}
        current_page = 1
        for element in main_div.descendants:
            element_to_page[id(element)] = current_page
            if _is_page_break_hr(element):
                current_page += 1

        # Skip elements on the Table of Contents page
        toc_page = None
        toc_pattern = re.compile(r'table\s+of\s+contents', re.IGNORECASE)
        for tag in main_div.find_all(['div', 'p']):
            if toc_pattern.search(tag.get_text(strip=True)):
                toc_page = element_to_page.get(id(tag))
                if toc_page:
                    break

        # Find section headers by text pattern (not just by ID)
        # Look for div/p/span tags outside of tables
        seen_sections = set()  # Track to avoid duplicates
        
        for tag in main_div.find_all(['div', 'p', 'span']):
            # Skip if inside a table (likely TOC)
            if tag.find_parent('table'):
                continue
            
            text = tag.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue
            
            page_num = element_to_page.get(id(tag), 1)
            if toc_page and page_num == toc_page:
                continue
            
            # Check for PART headers
            if part_pattern.match(text):
                # Avoid duplicate entries (div and span both match)
                section_key = (text, page_num, 1)
                if section_key not in seen_sections:
                    sections.append({
                        'title': text,
                        'level': 1,
                        'start_page': page_num,
                        'end_page': page_num,
                    })
                    seen_sections.add(section_key)
            
            # Check for Item headers
            elif item_pattern.match(text) and len(text) < 200:
                # Avoid duplicate entries
                section_key = (text, page_num, 2)
                if section_key not in seen_sections:
                    sections.append({
                        'title': text,
                        'level': 2,
                        'start_page': page_num,
                        'end_page': page_num,
                    })
                    seen_sections.add(section_key)

    elif not page_markers:
        # Fallback to word-based page tracking
        cumulative_words = 0
        WORDS_PER_PAGE = 3000

        for tag in soup.find_all(['div', 'p']):
            text = tag.get_text(strip=True)
            tag_words = len(text.split())
            cumulative_words += tag_words
            current_page = (cumulative_words // WORDS_PER_PAGE) + 1

            if tag.find('a', href=True):
                continue
            child_texts = {child.get_text(strip=True) for child in tag.find_all(['div', 'p'])}
            if text in child_texts:
                continue

            if part_pattern.match(text):
                sections.append({
                    'title': text,
                    'level': 1,
                    'start_page': current_page,
                    'end_page': current_page,
                })
            elif item_pattern.match(text):
                style = tag.get('style', '')
                if 'font-weight: bold' in style or tag.find('strong') or tag.find('b'):
                    sections.append({
                        'title': text,
                        'level': 2,
                        'start_page': current_page,
                        'end_page': current_page,
                    })
    else:
        # Use actual page markers
        # Build a map of elements to page numbers
        element_to_page = {}
        current_page = 0  
        for element in main_div.descendants:
            if hasattr(element, 'get') and element.get('class') == ['BRPFPageHeader']:
                current_page += 1
            if current_page > 0:  
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
            child_texts = {child.get_text(strip=True) for child in tag.find_all(['div', 'p'])}
            if text in child_texts:
                continue

            if part_pattern.match(text):
                sections.append({
                    'title': text,
                    'level': 1,
                    'start_page': page_num,
                    'end_page': page_num,
                })
                continue

            if item_pattern.match(text):
                style = tag.get('style', '')
                if 'font-weight: bold' in style or tag.find('strong') or tag.find('b'):
                    sections.append({
                        'title': text,
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
                    'content': page_text
                })
        
        section['text_length'] = len(section_text)
        section['word_count'] = len(section_text.split())

        has_subsections = bool(section.get('subsections'))
        if page_contents and not has_subsections:
            section['page_contents'] = page_contents

        if has_subsections:
            section['subsections'] = add_text_to_sections(
                section['subsections'], pages_data
            )
        else:
            section.pop('subsections', None)

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
        
        # First extract pages
        pages_data = extract_text_with_markers(html_path)
        # Detect sections from HTML structure
        flat_sections = detect_sections_from_html(html_path, pages_data)
        
        if not flat_sections:
            print("No sections found in HTML")
            return None
        
        
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



def main():
    parser = argparse.ArgumentParser(
        description="Parse HTML files and extract sections with page numbers"
    )
    parser.add_argument(
        "html_path",
        nargs='?',
        help="Path to the HTML file to parse",
        default="input/kimco.html"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)",
        default="parsing/parsed_sections_html.json"
    )

    args = parser.parse_args()

    print(f"Parsing HTML: {args.html_path}\n")


    # Parse sections
    result = sections_parser_html(args.html_path, args.output)


if __name__ == "__main__":
    main()