import argparse
import json
import re
import fitz  


def normalize_title(title):
    """
    Normalize section title by removing file prefixes and cleaning up.
    """
    # Remove common file prefixes like "UK01_0005821_01_"
    title = re.sub(r'^[A-Z0-9_]+_\d+_', '', title)
    # Remove version suffixes like "_v14"
    title = re.sub(r'_v\d+$', '', title)
    # Replace underscores with spaces
    title = title.replace('_', ' ')
    # Clean up multiple spaces
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
            
            # Determine end page (start of next section or last page)
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = len(doc)
            
            
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
            "sections": hierarchical_sections
        }
        
        doc.close()
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
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
        help="Path to the PDF file to parse"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)",
        default=None
    )
    parser.add_argument(
        "--print-sections",
        action="store_true",
        help="Print section summaries to console"
    )
    
    args = parser.parse_args()
    
    print(f"Parsing PDF: {args.pdf_path}\n")
    
    result = sections_parser_pdf(args.pdf_path, args.output)
    
    if result:
        print(f"\n✓ Successfully extracted {result['num_sections']} sections")
        
        if args.print_sections:
            print("\n" + "="*50)
            for section in result["sections"]:
                print(f"\n{section['title']}")
                print(f"Pages: {section['start_page']}-{section['end_page']}")


if __name__ == "__main__":
    main()
