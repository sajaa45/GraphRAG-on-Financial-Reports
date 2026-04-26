import argparse
import json
import re
import os
from typing import Dict, List
import fitz

_PREFIX_PATTERN = re.compile(r'^[A-Z0-9_]+_\d+_')
_VERSION_PATTERN = re.compile(r'_v\d+$')


def flatten_sections(sections: List[Dict]) -> List[Dict]:
    """Recursively flatten nested sections + subsections into a single list."""
    result = []
    for section in sections:
        result.append(section)
        if section.get('subsections'):
            result.extend(flatten_sections(section['subsections']))
    return result


def load_page_index(page_index_file: str) -> Dict:
    """Load page index JSON and return a dict keyed by page number."""
    print(f"✓ Loading page index from: {page_index_file}")
    try:
        with open(page_index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pages = data.get('pages', [])
        print(f"✓ Loaded {len(pages)} pages")
        return {p['page']: p for p in pages}
    except FileNotFoundError:
        print(f"✗ Page index file not found: {page_index_file}")
        return {}
    except Exception as e:
        print(f"✗ Error loading page index file: {e}")
        return {}


def add_text_from_page_index(sections: List[Dict], page_index: Dict) -> List[Dict]:
    """Populate section text and page_contents from a page index dict."""
    for section in sections:
        section_text = ""
        page_contents = []

        for page_num in range(section['start_page'], section['end_page'] + 1):
            page_data = page_index.get(page_num)
            if page_data and page_data.get('text'):
                page_text = page_data['text']
                section_text += page_text + "\n"
                page_contents.append({
                    'page_number': page_num,
                    'content': page_text,
                    'sections': page_data.get('sections', []),
                })

        section['text'] = section_text.strip()
        section['text_length'] = len(section_text)
        section['word_count'] = len(section_text.split())
        if page_contents:
            section['page_contents'] = page_contents

        if section.get('subsections'):
            section['subsections'] = add_text_from_page_index(
                section['subsections'], page_index
            )

    return sections


def normalize_title(title):
    title = _PREFIX_PATTERN.sub('', title)
    title = _VERSION_PATTERN.sub('', title)
    title = title.replace('_', ' ')
    title = ' '.join(title.split())
    return title


def build_hierarchy(sections):
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


def generate_page_index(pdf_path, sections_data, output_path=None):
    """
    Extract per-page text from the PDF and save a page index JSON.
    Uses the already-parsed sections to annotate which sections cover each page.
    """
    try:
        with fitz.open(pdf_path) as doc:
            flat_sections = sections_data.get('flat_sections') or flatten_sections(
                sections_data.get('sections', [])
            )

            # Pre-build page → sections map: one pass over sections instead of
            # re-scanning all sections for every page (O(sections*pages) → O(coverage))
            page_to_sections: dict[int, list] = {}
            for s in flat_sections:
                entry = {"title": s["title"], "level": s["level"]}
                for p in range(s["start_page"], s["end_page"] + 1):
                    page_to_sections.setdefault(p, []).append(entry)

            pages = []
            for page_num in range(1, len(doc) + 1):
                text = doc[page_num - 1].get_text()
                pages.append({
                    "page": page_num,
                    "sections": page_to_sections.get(page_num, []),
                    "text": text,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                })
                del text  # release each page's string before reading the next

        result = {
            "source_pdf": pdf_path,
            "total_pages": len(pages),
            "pages": pages,
        }

        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Page index saved to: {output_path}")

        return result

    except Exception as e:
        print(f"Error generating page index: {e}")
        return None


def sections_parser_pdf(pdf_path, output_path=None):
    """
    Parse a PDF file and extract sections based on TOC.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the output (JSON format)
    
    Returns:
        Dictionary containing parsed sections
    """
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Get TOC from PDF metadata
        toc = doc.get_toc()
        
        if not toc:
            print("No TOC found in PDF")
            doc.close()
            return None
        
        print(f"Found TOC with {len(toc)} entries")
        
        # Build flat list of sections from TOC
        flat_sections = []
        for i, entry in enumerate(toc):
            level, title, page_num = entry[0], entry[1], entry[2]
            
            # Skip sections that start on page 1
            if page_num == 1:
                print(f"Skipping section starting on page 1: {title}")
                continue
            
            # Normalize the title
            normalized_title = normalize_title(title)
            
            # Determine end page by finding the next valid section's start page
            end_page = len(doc)  # Default to last page
            for j in range(i + 1, len(toc)):
                next_page = toc[j][2]
                if next_page > page_num:  # Only consider sections that come after
                    end_page = next_page - 1
                    break
            
            section = {
                "level": level,
                "title": normalized_title,
                "original_title": title,
                "start_page": page_num,
                "end_page": end_page
            }
            
            flat_sections.append(section)
            print(f"Extracted: {normalized_title} (pages {page_num}-{end_page})")
        
        # Build hierarchical structure
        hierarchical_sections = build_hierarchy(flat_sections)
        
        result = {
            "filename": pdf_path,
            "num_pages": len(doc),
            "num_sections": len(flat_sections),
            "sections": hierarchical_sections,
            "flat_sections": flat_sections,
        }
        
        doc.close()
        
        # Save to file if output path provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            serializable = {k: v for k, v in result.items() if k != "flat_sections"}
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"\nOutput saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Parse PDF files using PyMuPDF and extract sections from TOC"
    )
    parser.add_argument(
        "pdf_path",
        nargs='?',
        help="Path to the PDF file to parse",
        default="input/NYSE_MTX_2024.pdf"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)",
        default="output/parsed_sections.json"
    )
    parser.add_argument(
        "--print-sections",
        action="store_true",
        help="Print section summaries to console"
    )
    parser.add_argument(
        "--page-index", "-p",
        help="Output path for page index JSON (default: output/page_index.json)",
        default="output/page_index.json"
    )

    args = parser.parse_args()

    print(f"Parsing PDF: {args.pdf_path}\n")

    result = sections_parser_pdf(args.pdf_path, args.output)

    if result:
        print(f"\n✓ Successfully extracted {result['num_sections']} sections")

        print(f"\nGenerating page index from PDF text...")
        generate_page_index(args.pdf_path, result, args.page_index)

        if args.print_sections:
            print("\n" + "="*50)
            for section in result["sections"]:
                print(f"\n{section['title']}")
                print(f"Pages: {section['start_page']}-{section['end_page']}")


if __name__ == "__main__":
    main()
