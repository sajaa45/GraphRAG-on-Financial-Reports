"""
Convert plain text SEC filing to markdown with proper headers
"""
import re

def txt_to_markdown(txt_path, md_path):
    """Add markdown headers to plain text based on SEC filing patterns"""
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip the XBRL metadata block at the beginning
    lines = content.split('\n')
    
    # Find where actual content starts (after XBRL metadata)
    start_idx = 0
    for i, line in enumerate(lines):
        # Look for "UNITED STATES" which typically starts SEC filings
        if 'UNITED STATES' in line:
            # Check if next line contains SECURITIES
            if i+1 < len(lines) and 'SECURITIES' in lines[i+1]:
                start_idx = i
                break
    
    # If we found the start, skip everything before it
    if start_idx > 0:
        print(f"Skipping {start_idx} lines of XBRL metadata")
        lines = lines[start_idx:]
    
    output_lines = []
    
    # Patterns for different header levels
    part_pattern = re.compile(r'^PART\s+([IVX]+)\.?\s*(.*)$', re.IGNORECASE)
    item_pattern = re.compile(r'^Item\s+(\d+[A-Z]?)\.?\s+(.+)$', re.IGNORECASE)
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            output_lines.append(line)
            continue
        
        # Check for PART headers (# level 1)
        part_match = part_pattern.match(stripped)
        if part_match:
            output_lines.append(f"# {stripped}")
            continue
        
        # Check for Item headers (## level 2)
        item_match = item_pattern.match(stripped)
        if item_match:
            output_lines.append(f"## {stripped}")
            continue
        
        # Check for all-caps headers that are likely section titles
        # Must be short (< 100 chars) and mostly uppercase
        if len(stripped) < 100 and stripped.isupper() and len(stripped.split()) > 1:
            # Check if next line is not also all caps (to avoid tables)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not (next_line and next_line.isupper()):
                    output_lines.append(f"### {stripped}")
                    continue
        
        # Regular line
        output_lines.append(line)
    
    # Write markdown file
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✓ Markdown with headers saved to: {md_path}")

if __name__ == "__main__":
    input_txt = "outputt.txt"  # Your plain text file from pandoc
    output_md = "output.md"
    
    print("Converting plain text to markdown with headers...")
    txt_to_markdown(input_txt, output_md)
    
    print("\n✓ Done! Check output.md")
