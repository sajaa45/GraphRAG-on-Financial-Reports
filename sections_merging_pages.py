import json
import os
import re
from typing import List, Dict, Tuple

def load_json_pages(json_path: str) -> Dict[str, str]:
    """
    Load pages from JSON file
    
    Args:
        json_path (str): Path to the JSON file
    
    Returns:
        Dict[str, str]: Dictionary with page numbers as keys and text content as values
    """
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
        
        print(f"Loaded {len(pages_data)} pages from JSON")
        return pages_data
    
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}

def should_combine_pages(current_page: str, previous_page: str) -> bool:
    """
    Determine if current page should be combined with previous page based on:
    - Current page doesn't start with uppercase letter
    - Previous page doesn't end with a period
    
    Args:
        current_page (str): Text content of current page
        previous_page (str): Text content of previous page
    
    Returns:
        bool: True if pages should be combined
    """
    if not current_page or not previous_page:
        return False
    
    # Clean and get first non-whitespace character of current page
    current_clean = current_page.strip()
    if not current_clean:
        return False
    
    # Check if current page starts with uppercase letter
    # We need to find the first alphabetic character, not just the first character
    first_alpha_char = None
    for char in current_clean:
        if char.isalpha():
            first_alpha_char = char
            break
    
    # If no alphabetic character found, treat as "starts with uppercase" (don't combine)
    if first_alpha_char is None:
        starts_with_uppercase = True
    else:
        starts_with_uppercase = first_alpha_char.isupper()
    
    # Clean and get last non-whitespace character of previous page
    previous_clean = previous_page.strip()
    if not previous_clean:
        return False
    
    # Check if previous page ends with period
    ends_with_period = previous_clean.endswith('.')
    
    # Combine if current doesn't start with uppercase AND previous doesn't end with period
    should_combine = not starts_with_uppercase and not ends_with_period
    
    return should_combine

def is_junk_heuristic(text: str, min_chars: int = 30, alpha_ratio: float = 0.05, dot_ratio: float = 0.8) -> bool:
    """
    Determine if a page contains mostly junk/noise content (extremely lenient parameters)
    
    Args:
        text: Page text content
        min_chars: Minimum character count (pages below this are junk)
        alpha_ratio: Minimum ratio of alphanumeric characters 
        dot_ratio: Maximum ratio any single character can appear 
    
    Returns:
        bool: True if page is likely junk
    """
    text_clean = text.strip()
    
    # Too short (very conservative)
    if len(text_clean) < min_chars:
        return True
    
    # Check for decorative/table pages with mostly symbols
    #if is_decorative_table_page(text_clean):
    #    return True
    
    # Calculate alpha ratio excluding table formatting characters AND numbers
    # Don't penalize legitimate tables for having | - = characters or numbers
    table_formatting_chars = {'|', '-', '+', '=', '_', '\\', '/', '<', '>', '{', '}', '[', ']', ',', '.', '%', '$'}
    text_for_alpha = ''.join(char for char in text_clean if char not in table_formatting_chars and not char.isdigit())
    
    alpha_count = sum(c.isalpha() for c in text_for_alpha)  # Only count letters, not numbers
    alpha = alpha_count / max(len(text_for_alpha), 1)
    if alpha < alpha_ratio:
        return True
    
    # Any single character appears too frequently (more lenient)
    # Skip common formatting characters that might appear frequently
    skip_chars = {' ', '\n', '\t', '-', '|', '=', '*', '#', '+', '_', ',', '.'}
    for char in set(text_clean):
        if char in skip_chars:
            continue
        char_ratio = text_clean.count(char) / len(text_clean)
        if char_ratio > dot_ratio:
            return True
    
    return False

def is_decorative_table_page(text: str) -> bool:
    """
    Detect pages that are mostly decorative tables with repetitive symbols
    Uses very conservative thresholds to avoid filtering legitimate data tables
    """
    # Count different types of characters
    bullet_chars = {'•', '●', '○', '◦', '▪', '▫', '■', '□', '◆', '◇', '$\\bullet$'}
    arabic_numerals = {'٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '\u0660', '\u06f0'}
    
    # 1. Check for excessive repetition of decorative symbols (very high threshold)
    bullet_count = sum(text.count(char) for char in bullet_chars)
    arabic_num_count = sum(text.count(char) for char in arabic_numerals)
    total_decorative = bullet_count + arabic_num_count
    
    # Only flag if more than 90% is decorative symbols (extremely conservative)
    if total_decorative / max(len(text), 1) > 0.9:
        return True
    
    # 2. Check for tables with mostly repetitive content (very conservative)
    lines = text.split('\n')
    table_lines = [line for line in lines if '|' in line and line.count('|') > 2]
    
    # Only check if it's almost entirely table lines (98%+) and many lines
    if len(table_lines) > 0.98 * len(lines) and len(table_lines) > 20:
        # Analyze table content diversity
        table_content = []
        for line in table_lines:
            # Extract content between | symbols
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Skip first/last empty
            table_content.extend(cells)
        
        if table_content:
            # Remove empty cells and very short cells (likely formatting)
            meaningful_cells = [cell for cell in table_content if len(cell.strip()) > 2]
            
            if len(meaningful_cells) > 20:  # Only analyze if enough content
                # Check for excessive repetition of the same content (very strict)
                unique_cells = set(meaningful_cells)
                diversity_ratio = len(unique_cells) / len(meaningful_cells)
                
                # Only flag if less than 1% unique content (extremely repetitive)
                if diversity_ratio < 0.01:
                    return True
                
                # Check if most cells are just symbols/bullets (very strict)
                symbol_cells = sum(1 for cell in meaningful_cells 
                                 if len(cell) <= 3 and any(char in cell for char in bullet_chars | arabic_numerals))
                symbol_ratio = symbol_cells / len(meaningful_cells)
                
                # Only flag if more than 99% of cells are just symbols
                if symbol_ratio > 0.99:
                    return True
    
    # 3. Check for pages with very low text diversity (very conservative)
    words = text.lower().split()
    if len(words) > 200:  # Only check if substantial content
        unique_words = set(words)
        word_diversity = len(unique_words) / len(words)
        
        # Only flag if less than 1% unique words (extremely repetitive)
        if word_diversity < 0.01:
            return True
    
    return False

def is_short_page(text: str, min_length: int = 50) -> bool:
    """Check if page is too short to be meaningful (more lenient)"""
    return len(text.strip()) < min_length

def analyze_page_content(text: str, page_num: str) -> Dict[str, any]:
    """Analyze page content to understand filtering decisions"""
    text_clean = text.strip()
    
    # Calculate alpha ratio excluding table formatting characters
    table_formatting_chars = {'|', '-', '+', '=', '_', '\\', '/', '<', '>', '{', '}', '[', ']'}
    text_for_alpha = ''.join(char for char in text_clean if char not in table_formatting_chars)
    
    alpha_count = sum(c.isalnum() for c in text_for_alpha)
    alpha_ratio = alpha_count / max(len(text_for_alpha), 1)
    
    # Find most frequent non-whitespace character
    char_counts = {}
    skip_chars = {' ', '\n', '\t', '-', '|', '=', '*', '#', '+', '_'}
    for char in text_clean:
        if char not in skip_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
    
    max_char_ratio = 0
    max_char = None
    if char_counts:
        max_char = max(char_counts, key=char_counts.get)
        max_char_ratio = char_counts[max_char] / len(text_clean)
    
    # Check decorative table detection
    is_decorative = False
    
    # Calculate content diversity (generic measure)
    words = text_clean.lower().split()
    word_diversity = 0
    if len(words) > 5:
        unique_words = set(words)
        word_diversity = len(unique_words) / len(words)
    
    return {
        'page': page_num,
        'length': len(text_clean),
        'alpha_ratio': round(alpha_ratio, 3),
        'max_char': max_char,
        'max_char_ratio': round(max_char_ratio, 3),
        'is_decorative': is_decorative,
        'word_diversity': round(word_diversity, 3),
        'preview': text_clean[:100] + "..." if len(text_clean) > 100 else text_clean
    }

def filter_pages(pages_data: Dict[str, str]) -> Dict[str, str]:
    """
    Filter out junk and short pages before sectioning (more conservative)
    
    Args:
        pages_data: Dictionary of page numbers and content
    
    Returns:
        Dict: Filtered pages dictionary
    """
    filtered_pages = {}
    filtered_count = {'junk': 0, 'short': 0, 'first_page': 0}
    filtered_details = []
    
    # Sort pages by page number for processing
    sorted_pages = sorted(pages_data.items(), key=lambda x: int(x[0]))
    
    print(f"  Filtering {len(sorted_pages)} pages...")
    
    for i, (page_num, page_text) in enumerate(sorted_pages):
        # Skip first page by default
        if i == 0:
            filtered_count['first_page'] += 1
            print(f"    Skipped page {page_num} (first page)")
            continue
        
        # Analyze page content
        analysis = analyze_page_content(page_text, page_num)
        
        # Check if page is junk (more conservative)
        if is_junk_heuristic(page_text):
            filtered_count['junk'] += 1
            reason = "decorative table" if analysis['is_decorative'] else "low quality"
            preview = analysis['preview'].replace('\n', '\\n')[:80] + "..." if len(analysis['preview']) > 80 else analysis['preview'].replace('\n', '\\n')
            filtered_details.append(f"    Page {page_num}: {reason} (len={analysis['length']}, alpha={analysis['alpha_ratio']},  decorative={analysis['is_decorative']})")
            filtered_details.append(f"      Preview: {preview}")
            continue
        
        # Check if page is too short (more conservative)
        if is_short_page(page_text):
            filtered_count['short'] += 1
            preview = analysis['preview'].replace('\n', '\\n')[:80] + "..." if len(analysis['preview']) > 80 else analysis['preview'].replace('\n', '\\n')
            filtered_details.append(f"    Page {page_num}: too short ({analysis['length']} chars)")
            filtered_details.append(f"      Preview: {preview}")
            continue
        
        # Keep this page
        filtered_pages[page_num] = page_text
    
    print(f"  Filtering results:")
    print(f"    Original pages: {len(sorted_pages)}")
    print(f"    Filtered out: {sum(filtered_count.values())} pages")
    print(f"      - First page: {filtered_count['first_page']}")
    print(f"      - Junk content: {filtered_count['junk']}")
    print(f"      - Too short: {filtered_count['short']}")
    print(f"    Kept: {len(filtered_pages)} pages")
    
    # Show details of ALL filtered pages
    if filtered_details:
        print(f"  All filtered pages:")
        for detail in filtered_details:
            print(detail)
    
    return filtered_pages

def create_page_sections(pages_data: Dict[str, str]) -> List[Dict[str, any]]:
    """
    Create sections from pages after filtering out junk and short pages
    - Filter out first page, junk content, and short pages
    - Each remaining page becomes a section
    - Combine with previous page if current doesn't start with uppercase and previous doesn't end with period
    
    Args:
        pages_data (Dict[str, str]): Dictionary of page numbers and content
    
    Returns:
        List[Dict]: List of sections with metadata
    """
    
    if not pages_data:
        return []
    
    # Filter pages before sectioning
    filtered_pages = filter_pages(pages_data)
    
    if not filtered_pages:
        print("  No pages remaining after filtering")
        return []
    
    # Sort filtered pages by page number (convert to int for proper sorting)
    sorted_pages = sorted(filtered_pages.items(), key=lambda x: int(x[0]))
    
    sections = []
    current_section_text = ""
    current_section_pages = []
    
    for i, (page_num, page_text) in enumerate(sorted_pages):
        page_text = page_text.strip()
        
        if i == 0:
            # First filtered page always starts a new section
            current_section_text = page_text
            current_section_pages = [page_num]
        else:
            # Check if we should combine with previous
            previous_page_text = sorted_pages[i-1][1].strip()
            
            if should_combine_pages(page_text, previous_page_text):
                # Combine with current section
                current_section_text += "\n\n" + page_text
                current_section_pages.append(page_num)
                print(f"  Combining page {page_num} with previous section (pages: {current_section_pages})")
            else:
                # Finish current section and start new one
                if current_section_text:
                    sections.append({
                        'text': current_section_text,
                        'pages': current_section_pages.copy(),
                        'page_range': f"{current_section_pages[0]}-{current_section_pages[-1]}" if len(current_section_pages) > 1 else current_section_pages[0],
                        'length': len(current_section_text),
                        'section_id': len(sections) + 1
                    })
                
                # Start new section
                current_section_text = page_text
                current_section_pages = [page_num]
    
    # Don't forget the last section
    if current_section_text:
        sections.append({
            'text': current_section_text,
            'pages': current_section_pages.copy(),
            'page_range': f"{current_section_pages[0]}-{current_section_pages[-1]}" if len(current_section_pages) > 1 else current_section_pages[0],
            'length': len(current_section_text),
            'section_id': len(sections) + 1
        })
    
    return sections

def save_sections_to_file(sections: List[Dict], output_path: str = None):
    """
    Save page sections to a text file for inspection
    
    Args:
        sections (List[Dict]): List of section dictionaries
        output_path (str, optional): Output file path
    """
    
    if output_path is None:
        output_dir = "/app/output" if os.path.exists("/app/output") else "."
        output_path = os.path.join(output_dir, "page_sections.txt")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"JSON Page Sections Results\n")
            f.write(f"Total sections: {len(sections)}\n")
            f.write("=" * 80 + "\n\n")
            
            for section in sections:
                f.write(f"SECTION {section['section_id']}\n")
                f.write(f"Pages: {section['page_range']}\n")
                f.write(f"Length: {section['length']} characters\n")
                f.write(f"Page numbers: {', '.join(section['pages'])}\n")
                f.write("-" * 40 + "\n")
                f.write(section['text'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"Sections saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving sections: {e}")

def analyze_sections(sections: List[Dict]) -> Dict[str, any]:
    """
    Analyze the section results and provide statistics
    
    Args:
        sections (List[Dict]): List of section dictionaries
    
    Returns:
        Dict: Analysis results
    """
    
    if not sections:
        return {"error": "No sections to analyze"}
    
    # Basic statistics
    total_sections = len(sections)
    section_lengths = [section['length'] for section in sections]
    total_characters = sum(section_lengths)
    
    # Page combination statistics
    single_page_sections = sum(1 for section in sections if len(section['pages']) == 1)
    multi_page_sections = sum(1 for section in sections if len(section['pages']) > 1)
    max_pages_in_section = max(len(section['pages']) for section in sections)
    
    # Length statistics
    avg_length = total_characters / total_sections
    min_length = min(section_lengths)
    max_length = max(section_lengths)
    
    analysis = {
        "total_sections": total_sections,
        "total_characters": total_characters,
        "average_section_length": round(avg_length, 2),
        "min_section_length": min_length,
        "max_section_length": max_length,
        "single_page_sections": single_page_sections,
        "multi_page_sections": multi_page_sections,
        "max_pages_per_section": max_pages_in_section,
        "combination_rate": round(multi_page_sections / total_sections * 100, 2)
    }
    
    return analysis

def main():
    """Main function to process JSON and create sections only"""
    
    print("JSON Text Processor - Page Sectioning Only")
    print("=" * 50)
    
    # Configuration - check for Docker environment
    input_dir = "/app/input" if os.path.exists("/app/input") else "."
    output_dir = "/app/output" if os.path.exists("/app/output") else "."
    
    # Look for JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        
        print(f"\nProcessing: {json_file}")
        print("=" * 50)
        
        # Check if section output files already exist
        sections_file = os.path.join(output_dir, json_file.replace('.json', '_sections.txt'))
        sections_json_file = os.path.join(output_dir, json_file.replace('.json', '_sections.json'))
        
        if os.path.exists(sections_file) and os.path.exists(sections_json_file):
            print(f"Section files already exist for {json_file}")
            print("   Skipping processing (files already exist)")
            continue
        
        # Load pages from JSON
        pages_data = load_json_pages(json_path)
        
        if not pages_data:
            print(f"Failed to load data from {json_file}")
            continue
        
        # Create page sections with filtering
        print("Creating page sections with filtering...")
        print("Filtering criteria (conservative):")
        print("  ✓ Skip first page (cover page)")
        print("  ✓ Remove junk content (< 25% alphanumeric, excessive repetition)")
        print("  ✓ Remove very short pages (< 50 characters)")
        print("  ✓ Ignore common formatting chars in repetition check")
        
        sections = create_page_sections(pages_data)
        
        if not sections:
            print("No sections created")
            continue
        
        # Analyze section results
        analysis = analyze_sections(sections)
        
        print(f"\nSection Results:")
        print(f"  Total sections: {analysis['total_sections']}")
        print(f"  Single-page sections: {analysis['single_page_sections']}")
        print(f"  Multi-page sections: {analysis['multi_page_sections']}")
        print(f"  Page combination rate: {analysis['combination_rate']}%")
        print(f"  Average section length: {analysis['average_section_length']} characters")
        print(f"  Length range: {analysis['min_section_length']} - {analysis['max_section_length']} characters")
        print(f"  Max pages per section: {analysis['max_pages_per_section']}")
        
        # Save sections to file
        save_sections_to_file(sections, sections_file)
        
        # Save sections as JSON for programmatic use
        try:
            with open(sections_json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis': analysis,
                    'sections': sections
                }, f, indent=2, ensure_ascii=False)
            print(f"Sections JSON saved to: {sections_json_file}")
        except Exception as e:
            print(f"Error saving sections JSON: {e}")
        
        print(f"\nSectioning completed for {json_file}")
        print(f"Next step: Run chunking.py to create filtered chunks with embeddings")

if __name__ == "__main__":
    main()