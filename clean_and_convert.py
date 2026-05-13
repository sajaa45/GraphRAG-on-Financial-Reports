"""
Clean SEC HTML filing by removing XBRL data, then convert to markdown using pandoc
"""
import subprocess
import re
from bs4 import BeautifulSoup

def clean_html(input_path, output_path):
    """Remove XBRL and other hidden data from SEC HTML filing"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove hidden XBRL data block
    for hidden in soup.find_all(id='DSPFiXBRLHidden'):
        hidden.decompose()
    
    # Remove other common hidden elements
    for element in soup.find_all(style=lambda x: x and 'display: none' in x):
        element.decompose()
    
    # Unwrap all inline XBRL tags (ix:nonnumeric, ix:nonfraction, etc.) - keep their text content
    for tag in soup.find_all(re.compile(r'^ix:')):
        tag.unwrap()
    
    # Remove XBRL namespace declarations from html tag
    html_tag = soup.find('html')
    if html_tag:
        # Keep only basic attributes
        attrs_to_keep = {}
        if 'lang' in html_tag.attrs:
            attrs_to_keep['lang'] = html_tag.attrs['lang']
        html_tag.attrs = attrs_to_keep
    
    # Save cleaned HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    print(f"✓ Cleaned HTML saved to: {output_path}")

def convert_to_markdown(html_path, md_path):
    """Convert HTML to markdown using pandoc"""
    try:
        subprocess.run([
            'pandoc', 
            html_path, 
            '-t', 'markdown',
            '-o', md_path
        ], check=True)
        print(f"✓ Markdown saved to: {md_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running pandoc: {e}")
    except FileNotFoundError:
        print("Error: pandoc not found. Please install pandoc first.")

if __name__ == "__main__":
    input_html = "input/NYSE_MTX_2024.htm"
    cleaned_html = "input/NYSE_MTX_2024_cleaned.htm"
    output_md = "output.md"
    
    print("Step 1: Cleaning HTML...")
    clean_html(input_html, cleaned_html)
    
    print("\nStep 2: Converting to markdown...")
    convert_to_markdown(cleaned_html, output_md)
    
    print("\n✓ Done! Check output.md")
