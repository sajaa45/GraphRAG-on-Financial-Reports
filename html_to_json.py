import json
import re
from bs4 import BeautifulSoup, NavigableString


def table_to_text(table):
    """
    Convert an HTML table to formatted text preserving layout.
    """
    rows = []
    
    # Extract all rows
    for tr in table.find_all('tr'):
        cells = []
        for cell in tr.find_all(['td', 'th']):
            # Get cell text and clean it
            text = cell.get_text(separator=' ', strip=True)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Handle colspan
            colspan = int(cell.get('colspan', 1))
            cells.extend([text] + [''] * (colspan - 1))
        if cells:
            rows.append(cells)
    
    if not rows:
        return ""
    
    # Calculate column widths
    num_cols = max(len(row) for row in rows)
    col_widths = [0] * num_cols
    
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Format table in markdown style
    formatted_rows = []
    
    for i, row in enumerate(rows):
        while len(row) < num_cols:
            row.append('')
        
        formatted_cells = []
        for j, cell in enumerate(row):
            if j < num_cols:
                # Left-align text, right-align if it looks like a number
                if cell and re.match(r'^[\d,.$%()-]+$', cell.replace(' ', '')):
                    formatted_cells.append(cell.rjust(col_widths[j]))
                else:
                    formatted_cells.append(cell.ljust(col_widths[j]))
        
        formatted_row = ' | '.join(formatted_cells)
        formatted_rows.append('| ' + formatted_row + ' |')
        
        # Add separator after header row 
        if i == 0:
            separator = '|' + '|'.join(['-' * (w + 2) for w in col_widths]) + '|'
            formatted_rows.append(separator)
    
    # for LLM context
    table_text = '[TABLE START]\n' + '\n'.join(formatted_rows) + '\n[TABLE END]'
    return table_text


def element_to_text(element):
    """
    Convert an HTML element to text, preserving table layouts.
    """
    if element.name == 'table':
        return table_to_text(element)
    elif element.name:
        text = element.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)
    else:
        return str(element).strip()


def html_to_paged_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Remove hidden XBRL data
    for hidden in soup.find_all(id='DSPFiXBRLHidden'):
        hidden.decompose()

    # Split the body on page-break <hr> tags or BRPFPageHeader divs
    body = soup.find("body")
    
    # Try to find page markers
    main_div = soup.find('div', style=lambda x: x and 'line-height: initial' in x)
    if main_div:
        page_markers = main_div.find_all('div', class_='BRPFPageHeader')
        if page_markers:
            print(f"Found {len(page_markers)} page markers (BRPFPageHeader)")
            pages = extract_pages_with_markers(main_div, page_markers)
        else:
            print("No page markers found, using page-break method")
            pages = extract_pages_by_breaks(body)
    else:
        print("No main div found, using page-break method")
        pages = extract_pages_by_breaks(body)

    result = {
        "source": input_path,
        "total_pages": len(pages),
        "pages": pages
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"{input_path} -> {output_path} ({len(pages)} pages)")


def extract_pages_with_markers(main_div, page_markers):
    """
    Extract pages using BRPFPageHeader markers, preserving table layouts.
    """
    pages = {}
    current_page = 0  
    current_chunks = []
    processed_tables = set()
    
    for element in main_div.descendants:
        # Check if this is a page marker
        if hasattr(element, 'get') and element.get('class') == ['BRPFPageHeader']:
            # Save previous page if we have content
            if current_chunks and current_page > 0:
                text = '\n\n'.join(current_chunks).strip()
                text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
                if text:
                    pages[f"{current_page}"] = text
            # Start new page
            current_page += 1
            current_chunks = []
            processed_tables = set()
        elif element.name == 'table':
            table_id = id(element)
            if table_id not in processed_tables and element.parent.name != 'table':
                table_text = table_to_text(element)
                if table_text:
                    current_chunks.append('\n' + table_text + '\n')
                    processed_tables.add(table_id)
                    # Mark all descendants as processed to avoid duplication
                    for desc in element.descendants:
                        if hasattr(desc, 'name'):
                            processed_tables.add(id(desc))
        elif isinstance(element, NavigableString):
            # Skip if this text is inside a table we've already processed
            parent_table = element.find_parent('table')
            if parent_table and id(parent_table) in processed_tables:
                continue
            
            text = str(element).strip()
            if text and text != '\xa0':
                text = re.sub(r'\s+', ' ', text)
                if text:
                    current_chunks.append(text)
    
    if current_chunks and current_page > 0:
        text = '\n\n'.join(current_chunks).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        if text:
            pages[f"{current_page}"] = text
    
    return pages


def extract_pages_by_breaks(body):
    """
    Extract pages using page-break hr tags, preserving table layouts.
    """
    pages = {}
    current_page = 1
    current_chunks = []
    processed_elements = set()

    for element in body.children:
        tag = getattr(element, "name", None)
        style = element.get("style", "") if tag else ""

        if tag == "hr" and "page-break" in style:
            text = '\n\n'.join(current_chunks).strip()
            text = re.sub(r'\n{3,}', '\n\n', text)
            if text:
                pages[f"{current_page}"] = text
            current_page += 1
            current_chunks = []
            processed_elements = set()
        elif tag == 'table':
            table_id = id(element)
            if table_id not in processed_elements:
                table_text = table_to_text(element)
                if table_text:
                    current_chunks.append('\n' + table_text + '\n')
                    processed_elements.add(table_id)
        elif tag:
            tables = element.find_all('table') if hasattr(element, 'find_all') else []
            if tables:
                # Process element with tables 
                for table in tables:
                    table_id = id(table)
                    if table_id not in processed_elements:
                        table_text = table_to_text(table)
                        if table_text:
                            current_chunks.append('\n' + table_text + '\n')
                            processed_elements.add(table_id)
                # Get text outside tables
                for table in tables:
                    table.decompose()
                text = element.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text)
                if text:
                    current_chunks.append(text)
            else:
                # No tables, just get text
                chunk = element_to_text(element)
                if chunk:
                    current_chunks.append(chunk)
        else:
            # Text node
            chunk = str(element).strip()
            if chunk:
                current_chunks.append(chunk)

    # Last page (no <hr>)
    text = '\n\n'.join(current_chunks).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    if text:
        pages[f"{current_page}"] = text
    
    return pages

html_to_paged_json("input/NYSE_MTX_2024.htm",  "output/NYSE_MTX_2024.json")
