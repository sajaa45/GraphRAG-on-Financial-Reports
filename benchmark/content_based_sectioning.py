#!/usr/bin/env python3
"""
Content-Based Sectioning Processor
Uses PDF TOC metadata to create hierarchical sections
"""

import json
import re
import fitz  # PyMuPDF
from typing import Dict, List, Any


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


def build_hierarchy(flat_sections):
    """
    Build hierarchical structure from flat list of sections.
    """
    if not flat_sections:
        return []
    
    root = []
    stack = []  # Stack to track parent sections at each level
    
    for section in flat_sections:
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


class ContentBasedSectioning:
    def __init__(self, pdf_path):
        """Initialize with PDF path"""
        self.pdf_path = pdf_path
        self.doc = None
    
    def process_document(self) -> Dict[str, Any]:
        """
        Process the PDF document using TOC-based sectioning
        """
        try:
            print("Content-Based Sectioning Processor (TOC-based)")
            print("=" * 50)
            
            # Open the PDF
            self.doc = fitz.open(self.pdf_path)
            
            # Get TOC from PDF metadata
            toc = self.doc.get_toc()
            
            if not toc:
                print("No TOC found in PDF")
                self.doc.close()
                return self.fallback_sectioning()
            
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
                    end_page = len(self.doc)
                
                # Extract text from section pages
                section_text = []
                for page_idx in range(page_num - 1, end_page):
                    if page_idx < len(self.doc):
                        page = self.doc[page_idx]
                        text = page.get_text()
                        section_text.append(text)
                
                combined_text = '\n'.join(section_text)
                
                section = {
                    "level": level,
                    "title": normalized_title,
                    "original_title": title,
                    "start_page": page_num,
                    "end_page": end_page,
                    "text": combined_text,
                    "char_count": len(combined_text)
                }
                
                flat_sections.append(section)
                print(f"Extracted: {normalized_title} (pages {page_num}-{end_page})")
            
            # Build hierarchical structure
            hierarchical_sections = build_hierarchy(flat_sections)
            
            result = {
                "filename": self.pdf_path,
                "num_pages": len(self.doc),
                "num_sections": len(flat_sections),
                "sections": hierarchical_sections,
                "metadata": {
                    "sectioning_method": "toc_based",
                    "total_sections": len(flat_sections)
                }
            }
            
            self.doc.close()
            
            print(f"\n✓ Successfully extracted {len(flat_sections)} sections")
            
            return result
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            if self.doc:
                self.doc.close()
            return None
    
    def fallback_sectioning(self) -> Dict[str, Any]:
        """
        Fallback method when no TOC is detected
        """
        print("Using fallback sectioning method...")
        
        if self.doc:
            self.doc.close()
        
        return {
            "filename": self.pdf_path,
            "num_pages": 0,
            "num_sections": 0,
            "sections": [],
            "metadata": {
                "sectioning_method": "fallback",
                "error": "No TOC found"
            }
        }


def main():
    """Test the content-based sectioning independently"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract sections from PDF using TOC"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)",
        default="output/content_based_sections.json"
    )
    
    args = parser.parse_args()
    
    try:
        # Process with content-based sectioning
        processor = ContentBasedSectioning(args.pdf_path)
        result = processor.process_document()
        
        if not result:
            print("Failed to process document")
            return
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save text version for easy reading
        text_output = args.output.replace('.json', '.txt')
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write("Content-Based Sections (TOC-based)\n")
            f.write("=" * 50 + "\n\n")
            
            def write_section(section, indent=0):
                prefix = "  " * indent
                f.write(f"{prefix}Section: {section['title']}\n")
                f.write(f"{prefix}Pages: {section['start_page']}-{section['end_page']}\n")
                f.write(f"{prefix}Characters: {section['char_count']}\n")
                f.write(f"{prefix}{'-' * 40}\n")
                
                if section.get('subsections'):
                    f.write(f"{prefix}Subsections:\n")
                    for subsection in section['subsections']:
                        write_section(subsection, indent + 1)
                
                f.write("\n")
            
            for section in result['sections']:
                write_section(section)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {args.output}")
        print(f"  Text: {text_output}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
    def __init__(self):
        """Initialize the content-based sectioning processor"""
        self.toc_patterns = [
            # Pattern for "- 02 **Corporate Information**"
            r'^-\s*(\d+)\s*\*\*([^*]+)\*\*',
            # Pattern for "**Profile of the Board of Directors** 04"
            r'^\*\*([^*]+)\*\*\s+(\d+)',
            # Pattern for "02 Corporate Information"
            r'^(\d+)\s+([A-Z][^0-9\n]+?)(?=\s*$|\s*\d)',
            # Pattern for "Corporate Information 02"
            r'^([A-Z][^0-9\n]+?)\s+(\d+)(?=\s*$)',
            # Pattern for numbered sections "1. Introduction"
            r'^(\d+)\.\s*([A-Z][^0-9\n]+?)(?=\s*$)',
            # Pattern for lettered sections "A. Board Leadership"
            r'^([A-Z])\.\s*([A-Z][^0-9\n]+?)(?=\s*$)'
        ]
    
    def extract_table_of_contents(self, pages_data: Dict[str, str]) -> Tuple[List[Dict], int]:
        """
        Automatically extract table of contents from the document
        Returns: (toc_entries, content_start_page)
        """
        print("Extracting table of contents...")
        
        toc_entries = []
        toc_pages = []
        
        # Look for table of contents in first few pages
        for page_num in range(1, min(6, len(pages_data) + 1)):
            page_key = str(page_num)
            if page_key not in pages_data:
                continue
                
            page_text = pages_data[page_key]
            
            # Check if this page contains table of contents
            if self.is_toc_page(page_text):
                print(f"Found table of contents on page {page_num}")
                toc_pages.append(page_num)
                entries = self.parse_toc_page(page_text)
                toc_entries.extend(entries)
        
        # Sort by page number
        toc_entries.sort(key=lambda x: x['page'])
        
        print(f"Extracted {len(toc_entries)} table of contents entries")
        
        # Determine content start page
        content_start_page = self.determine_content_start_page(toc_entries, toc_pages)
        
        # If no TOC found, create sections based on major headings
        if not toc_entries:
            print("No table of contents found, detecting sections from headings...")
            toc_entries = self.detect_sections_from_headings(pages_data)
            content_start_page = self.determine_content_start_page_fallback(pages_data)
        
        return toc_entries, content_start_page
    
    def determine_content_start_page(self, toc_entries: List[Dict], toc_pages: List[int]) -> int:
        """
        Determine where the actual content starts (after preliminary pages)
        """
        if not toc_entries:
            return 1
        
        # Find the earliest content page from TOC
        min_content_page = min(entry['page'] for entry in toc_entries)
        
        # Content starts at the first actual section, not before
        content_start = min_content_page
        
        # Also exclude TOC pages themselves
        if toc_pages:
            max_toc_page = max(toc_pages)
            content_start = max(content_start, max_toc_page + 1)
        
        print(f"Content starts at page {content_start}")
        return content_start
    
    def determine_content_start_page_fallback(self, pages_data: Dict[str, str]) -> int:
        """
        Fallback method to determine content start when no TOC is found
        Uses structural indicators rather than hardcoded keywords
        """
        for page_num in range(1, min(6, len(pages_data) + 1)):
            page_key = str(page_num)
            if page_key not in pages_data:
                continue
                
            page_text = pages_data[page_key]
            
            # Check if page is very short (likely cover/title page)
            if len(page_text.strip()) < 200:
                continue
            
            # Check if page has substantial content structure
            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
            
            # Look for structured content indicators
            has_headings = any(line.startswith('#') or line.isupper() for line in lines[:5])
            has_paragraphs = len([line for line in lines if len(line) > 50]) >= 3
            
            # If page has structured content, it's likely actual content
            if has_headings and has_paragraphs:
                print(f"Content appears to start at page {page_num} (fallback method)")
                return page_num
        
        # Default to page 3 if can't determine
        print("Using default content start page 3")
        return 3
    
    def filter_preliminary_pages(self, pages_data: Dict[str, str], content_start_page: int) -> Dict[str, str]:
        """
        Remove preliminary pages (cover, TOC, etc.) from processing
        """
        filtered_pages = {}
        excluded_pages = []
        
        for page_key, page_text in pages_data.items():
            page_num = int(page_key)
            
            if page_num >= content_start_page:
                filtered_pages[page_key] = page_text
            else:
                excluded_pages.append(page_num)
        
        if excluded_pages:
            print(f"Excluded preliminary pages: {excluded_pages}")
        
        print(f"Processing {len(filtered_pages)} content pages (from page {content_start_page})")
        
        return filtered_pages
    
    def is_toc_page(self, page_text: str) -> bool:
        """
        Determine if a page contains table of contents
        Uses structural indicators rather than hardcoded keywords
        """
        lines = page_text.split('\n')
        
        # Look for structural patterns that indicate TOC
        page_number_count = 0
        title_like_lines = 0
        dots_or_dashes = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Count lines with page numbers
            if re.search(r'\b\d{1,3}\b', line):
                page_number_count += 1
            
            # Count lines that look like titles (start with capital, reasonable length)
            if re.match(r'^[A-Z]', line) and 10 <= len(line) <= 80:
                title_like_lines += 1
            
            # Count lines with dots or dashes (common in TOC formatting)
            if '...' in line or '---' in line or line.count('.') > 5:
                dots_or_dashes += 1
        
        # Heuristic: likely TOC if has multiple page numbers and title-like lines
        total_lines = len([l for l in lines if l.strip()])
        
        if total_lines < 5:  # Too short to be TOC
            return False
        
        page_number_ratio = page_number_count / total_lines
        title_ratio = title_like_lines / total_lines
        
        # Likely TOC if good ratio of page numbers and titles
        return (page_number_ratio >= 0.3 and title_ratio >= 0.3) or dots_or_dashes >= 3
    
    def parse_toc_page(self, page_text: str) -> List[Dict]:
        """
        Parse table of contents entries from a page
        Only accepts entries that have both title and page number
        """
        entries = []
        lines = page_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try each pattern
            for pattern in self.toc_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    # Determine which group is page number and which is title
                    if len(groups) == 2:
                        page_num = None
                        title = None
                        
                        if groups[0].isdigit():
                            page_num, title = int(groups[0]), groups[1].strip()
                        elif groups[1].isdigit():
                            title, page_num = groups[0].strip(), int(groups[1])
                        else:
                            # Skip if no valid page number found
                            continue
                        
                        # Clean up title
                        title = re.sub(r'\*+', '', title).strip()
                        title = re.sub(r'\s+', ' ', title)
                        
                        # Only accept if we have both title and valid page number
                        if title and page_num and page_num > 0:
                            entries.append({
                                'name': title,
                                'page': page_num,
                                'source': 'toc'
                            })
                            break
        
        return entries
    
    def detect_sections_from_headings(self, pages_data: Dict[str, str]) -> List[Dict]:
        """
        Detect sections by finding major headings in the document
        Only includes headings that appear to be section titles (no hardcoded keywords)
        """
        sections = []
        
        heading_patterns = [
            r'^#\s+(.+)$',  # Markdown heading
            r'^\*\*([^*]+)\*\*$',  # Bold text as heading
            r'^([A-Z][A-Z\s&]+)$',  # ALL CAPS headings
            r'^([A-Z][a-z\s&]+(?:[A-Z][a-z\s&]*)+)$'  # Title Case headings
        ]
        
        for page_num, page_text in pages_data.items():
            lines = page_text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if len(line) < 5 or len(line) > 100:  # Skip very short or long lines
                    continue
                
                for pattern in heading_patterns:
                    match = re.search(pattern, line)
                    if match:
                        title = match.group(1).strip()
                        
                        # Only include if it looks like a section heading
                        # (no hardcoded keywords - just structural checks)
                        if self.is_structural_heading(title, line_idx, lines):
                            sections.append({
                                'name': title,
                                'page': int(page_num),
                                'source': 'heading'
                            })
                            break
        
        # Remove duplicates and sort
        seen = set()
        unique_sections = []
        for section in sections:
            key = (section['name'].lower(), section['page'])
            if key not in seen:
                seen.add(key)
                unique_sections.append(section)
        
        unique_sections.sort(key=lambda x: x['page'])
        return unique_sections
    
    def is_structural_heading(self, title: str, line_idx: int, lines: List[str]) -> bool:
        """
        Determine if a title is likely a section heading based on structure only
        (no hardcoded keywords)
        """
        # Check length (reasonable section titles)
        if not (5 <= len(title) <= 80):
            return False
        
        # Avoid common non-section phrases (minimal list)
        avoid_phrases = ['page', 'continued', 'cont\'d', 'note', 'table', 'figure', 'see page']
        title_lower = title.lower()
        if any(phrase in title_lower for phrase in avoid_phrases):
            return False
        
        # Check if it's at the beginning of the page or after significant whitespace
        # (structural indicator of a section heading)
        if line_idx == 0:  # First line of page
            return True
        
        # Check if preceded by empty lines (section break indicator)
        empty_lines_before = 0
        for i in range(max(0, line_idx - 3), line_idx):
            if i < len(lines) and not lines[i].strip():
                empty_lines_before += 1
        
        # If preceded by 2+ empty lines, likely a section heading
        if empty_lines_before >= 2:
            return True
        
        # Check if it's a standalone line (not part of a paragraph)
        is_standalone = True
        if line_idx > 0 and lines[line_idx - 1].strip():
            is_standalone = False
        if line_idx < len(lines) - 1 and lines[line_idx + 1].strip() and not lines[line_idx + 1].startswith('#'):
            # Next line exists and is not empty and not another heading
            pass
        
        return is_standalone
    
    def should_combine_pages(self, current_page_text: str, previous_page_text: str) -> bool:
        """
        Determine if current page should be combined with previous page
        Logic: Combine if current page doesn't start with uppercase letter AND previous doesn't end with period
        """
        if not current_page_text or not previous_page_text:
            return False
        
        # Find first alphabetic character in current page
        first_alpha_char = None
        for char in current_page_text:
            if char.isalpha():
                first_alpha_char = char
                break
        
        # If no alphabetic character found, don't combine
        if first_alpha_char is None:
            return False
        
        # Check if current page starts with uppercase
        current_starts_uppercase = first_alpha_char.isupper()
        
        # Check if previous page ends with period
        previous_ends_period = previous_page_text.rstrip().endswith('.')
        
        # Combine if current doesn't start with uppercase AND previous doesn't end with period
        return not current_starts_uppercase and not previous_ends_period
    
    def create_sections_from_toc(self, pages_data: Dict[str, str], toc_entries: List[Dict]) -> List[Dict]:
        """
        Create sections based on TOC entries, where each section includes all pages 
        from its title page until the next section title, with individual page content preserved
        """
        sections = []
        section_id = 1
        
        # Get the maximum page number from content pages
        max_page = max([int(k) for k in pages_data.keys()]) if pages_data else 0
        
        for i, main_section in enumerate(toc_entries):
            start_page = main_section["page"]
            
            # Skip sections that don't exist in our filtered pages
            if str(start_page) not in pages_data:
                continue
            
            # Find end page (start of next section - 1, or max page for last section)
            if i + 1 < len(toc_entries):
                # Find next section that exists in our content pages
                end_page = max_page
                for j in range(i + 1, len(toc_entries)):
                    next_start = toc_entries[j]["page"]
                    if str(next_start) in pages_data:
                        end_page = next_start - 1
                        break
            else:
                end_page = max_page
            
            # Ensure end page doesn't exceed document
            end_page = min(end_page, max_page)
            
            print(f"Processing section: {main_section['name']} (pages {start_page}-{end_page})")
            
            # Collect all pages for this section with individual content
            section_pages = []
            page_contents = []
            total_length = 0
            
            for page_num in range(start_page, end_page + 1):
                page_key = str(page_num)
                if page_key in pages_data:
                    page_text = pages_data[page_key]
                    section_pages.append(page_num)
                    page_contents.append({
                        'page_number': page_num,
                        'content': page_text,
                        'length': len(page_text)
                    })
                    total_length += len(page_text)
            
            if section_pages:
                # Create section entry with individual page contents
                section = {
                    'section_id': section_id,
                    'main_section': main_section['name'],
                    'pages': section_pages,
                    'page_contents': page_contents,
                    'page_range': f"{section_pages[0]}-{section_pages[-1]}" if len(section_pages) > 1 else str(section_pages[0]),
                    'total_length': total_length,
                    'detection_method': main_section['source'],
                    'page_count': len(section_pages)
                }
                
                sections.append(section)
                section_id += 1
                
                print(f"  Created section with {len(section_pages)} pages")
        
        return sections
    
    def process_document(self, pages_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Process the entire document using content-based sectioning
        """
        print("Content-Based Sectioning Processor")
        print("=" * 50)
        
        # Extract table of contents and determine content start
        toc_entries, content_start_page = self.extract_table_of_contents(pages_data)
        
        # Filter out preliminary pages
        content_pages = self.filter_preliminary_pages(pages_data, content_start_page)
        
        if not toc_entries:
            print("No sections detected. Using fallback method...")
            return self.fallback_sectioning(content_pages)
        
        # Filter TOC entries to only include those within content pages
        valid_toc_entries = [
            entry for entry in toc_entries 
            if entry['page'] >= content_start_page and str(entry['page']) in content_pages
        ]
        
        if not valid_toc_entries:
            print("No valid TOC entries found in content pages. Using fallback method...")
            return self.fallback_sectioning(content_pages)
        
        print(f"Using {len(valid_toc_entries)} valid TOC entries for sectioning")
        
        # Create sections based on TOC entries
        all_sections = self.create_sections_from_toc(content_pages, valid_toc_entries)
        
        # Create summary
        total_original_pages = len(pages_data)
        total_content_pages = len(content_pages)
        total_sections = len(all_sections)
        excluded_pages = total_original_pages - total_content_pages
        
        print(f"\nContent-Based Sectioning Summary:")
        print(f"  Total original pages: {total_original_pages}")
        print(f"  Excluded preliminary pages: {excluded_pages}")
        print(f"  Content pages processed: {total_content_pages}")
        print(f"  Total sections created: {total_sections}")
        print(f"  Main sections detected: {len(valid_toc_entries)}")
        
        return {
            'sections': all_sections,
            'metadata': {
                'total_original_pages': total_original_pages,
                'total_content_pages': total_content_pages,
                'excluded_pages': excluded_pages,
                'content_start_page': content_start_page,
                'total_sections': total_sections,
                'main_sections_count': len(valid_toc_entries),
                'sectioning_method': 'content_based_auto',
                'detected_toc_entries': valid_toc_entries
            }
        }
    
    def fallback_sectioning(self, pages_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Fallback method when no TOC is detected - use simple page grouping with your combination logic
        """
        print("Using fallback sectioning method...")
        
        all_sections = []
        section_id = 1
        current_section_pages = []
        current_page_contents = []
        
        page_numbers = sorted([int(k) for k in pages_data.keys()])
        
        for i, page_num in enumerate(page_numbers):
            page_text = pages_data[str(page_num)]
            
            if not current_section_pages:
                # Start first section
                current_section_pages = [page_num]
                current_page_contents = [{
                    'page_number': page_num,
                    'content': page_text,
                    'length': len(page_text)
                }]
                continue
            
            # Check if we should combine with previous page
            previous_page_text = pages_data[str(page_numbers[i-1])]
            
            if self.should_combine_pages(page_text, previous_page_text):
                # Combine with current section
                current_section_pages.append(page_num)
                current_page_contents.append({
                    'page_number': page_num,
                    'content': page_text,
                    'length': len(page_text)
                })
            else:
                # Start new section - save current one first
                total_length = sum(page['length'] for page in current_page_contents)
                
                all_sections.append({
                    'section_id': section_id,
                    'main_section': f"Section {section_id}",
                    'pages': current_section_pages.copy(),
                    'page_contents': current_page_contents.copy(),
                    'page_range': f"{current_section_pages[0]}-{current_section_pages[-1]}" if len(current_section_pages) > 1 else str(current_section_pages[0]),
                    'total_length': total_length,
                    'detection_method': 'fallback',
                    'page_count': len(current_section_pages)
                })
                section_id += 1
                
                # Start new section
                current_section_pages = [page_num]
                current_page_contents = [{
                    'page_number': page_num,
                    'content': page_text,
                    'length': len(page_text)
                }]
        
        # Add the last section
        if current_section_pages:
            total_length = sum(page['length'] for page in current_page_contents)
            
            all_sections.append({
                'section_id': section_id,
                'main_section': f"Section {section_id}",
                'pages': current_section_pages.copy(),
                'page_contents': current_page_contents.copy(),
                'page_range': f"{current_section_pages[0]}-{current_section_pages[-1]}" if len(current_section_pages) > 1 else str(current_section_pages[0]),
                'total_length': total_length,
                'detection_method': 'fallback',
                'page_count': len(current_section_pages)
            })
        
        return {
            'sections': all_sections,
            'metadata': {
                'total_original_pages': 'unknown',
                'total_content_pages': len(pages_data),
                'excluded_pages': 'unknown',
                'content_start_page': min([int(k) for k in pages_data.keys()]) if pages_data else 1,
                'total_sections': len(all_sections),
                'main_sections_count': len(all_sections),
                'sectioning_method': 'fallback',
                'detected_toc_entries': []
            }
        }

def main():
    """Test the content-based sectioning independently"""
    
    # Load the original pages data
    input_file = "shell-annual-report-2024.json"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            pages_data = json.load(f)
        
        print(f"Loaded {len(pages_data)} pages from {input_file}")
        
        # Process with content-based sectioning
        processor = ContentBasedSectioning()
        result = processor.process_document(pages_data)
        
        # Save results
        output_file = "output/content_based_sections.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save text version for easy reading
        text_output_file = "output/content_based_sections.txt"
        with open(text_output_file, 'w', encoding='utf-8') as f:
            f.write("Content-Based Sections\n")
            f.write("=" * 50 + "\n\n")
            
            for section in result['sections']:
                f.write(f"Section {section['section_id']}: {section['main_section']}\n")
                f.write(f"Page numbers: {', '.join(map(str, section['pages']))}\n")
                f.write(f"Page range: {section['page_range']}\n")
                f.write(f"Length: {section['length']} characters\n")
                f.write("-" * 40 + "\n")
                f.write(section['text'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        print(f"\nResults saved to:")
        print(f"  JSON: {output_file}")
        print(f"  Text: {text_output_file}")
        
        # Print sample sections
        print(f"\nSample sections:")
        for i, section in enumerate(result['sections'][:3]):
            print(f"  Section {section['section_id']}: {section['main_section']}")
            print(f"    Pages: {section['page_range']}")
            print(f"    Length: {section['length']} chars")
            print(f"    Preview: {section['text'][:100]}...")
            print()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()