import argparse
import json
import re
import os
from typing import Dict, List, Optional
import html2text
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------

_PART_PATTERN = re.compile(r'^PART\s+[IVX]+', re.IGNORECASE)
_ITEM_PATTERN = re.compile(r'^Item\s+\d+\s*[A-Z]?\.?\s*', re.IGNORECASE)


def _inject_headings(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Replace div/p elements that match PART or Item section headers with
    <h1>/<h2> tags so html2text emits proper Markdown ATX headings.
    Skips elements inside tables (TOC entries) and deduplicates via a seen-set.
    """
    seen: set = set()

    for tag in soup.find_all(['div', 'p']):
        if tag.find_parent('table'):
            continue

        text = re.sub(r'\s+', ' ', tag.get_text(separator=' ', strip=True))
        if not text or len(text) > 200:
            continue

        if _PART_PATTERN.match(text):
            key = ('part', text[:40])
            if key in seen:
                continue
            seen.add(key)
            h = soup.new_tag('h1')
            h.string = text
            tag.replace_with(h)

        elif _ITEM_PATTERN.match(text):
            # Accept items whose element or any descendant carries heading-like styling
            all_styles = ' '.join(
                str(d.get('style', ''))
                for d in [tag] + list(tag.descendants)
                if hasattr(d, 'get')
            )
            is_heading = (
                'font-weight' in all_styles
                or 'text-decoration:underline' in all_styles.replace(' ', '')
                or bool(tag.find(['strong', 'b', 'u']))
            )
            if not is_heading:
                continue

            key = ('item', text[:40])
            if key in seen:
                continue
            seen.add(key)
            h = soup.new_tag('h2')
            h.string = text
            tag.replace_with(h)

    return soup


_PAGE_BREAK_MARKER = '\n\n<<<PAGE_BREAK>>>\n\n'
_PAGE_NUMBER_RE = re.compile(r'^\s*\d+\s*$')


def _inject_page_breaks(soup: BeautifulSoup) -> BeautifulSoup:
    """Replace <hr> tags (SEC page breaks) with a sentinel that survives html2text."""
    for hr in soup.find_all('hr'):
        marker = soup.new_tag('p')
        marker.string = '<<<PAGE_BREAK>>>'
        hr.replace_with(marker)
    return soup


def html_to_markdown(html_path: str, save_md: str = None) -> str:
    """Convert an HTML file to Markdown, injecting ATX headings for PART/Item sections."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    for hidden in soup.find_all(id='DSPFiXBRLHidden'):
        hidden.decompose()

    soup = _inject_headings(soup)
    soup = _inject_page_breaks(soup)

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0
    converter.unicode_snob = True
    converter.decode_errors = 'ignore'

    markdown_text = converter.handle(str(soup))

    if save_md:
        os.makedirs(os.path.dirname(save_md) or '.', exist_ok=True)
        with open(save_md, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"Markdown saved to: {save_md}")

    return markdown_text


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

def parse_markdown_sections(markdown_text: str) -> List[Dict]:
    """
    Parse Markdown into a flat list of sections using ATX headings (# ## ###...).
    Each entry has title, level, and raw_text (body up to the next heading).
    """
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE)
    matches = list(heading_pattern.finditer(markdown_text))

    if not matches:
        return [{'title': 'Document', 'level': 1, 'raw_text': markdown_text.strip()}]

    sections = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        sections.append({
            'title': title,
            'level': level,
            'raw_text': markdown_text[body_start:body_end].strip(),
        })

    return sections


def build_hierarchy(sections: List[Dict]) -> List[Dict]:
    root: List[Dict] = []
    stack: List[Dict] = []

    for section in sections:
        level = section['level']
        node = {**section, 'subsections': []}

        while stack and stack[-1]['level'] >= level:
            stack.pop()

        if stack:
            stack[-1]['subsections'].append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def _flatten(sections: List[Dict], result: Optional[List] = None) -> List[Dict]:
    if result is None:
        result = []
    for s in sections:
        result.append(s)
        if s.get('subsections'):
            _flatten(s['subsections'], result)
    return result


def add_text_to_sections(sections: List[Dict], chunk_idx: List[int] = None, page_num: List[int] = None) -> List[Dict]:
    if chunk_idx is None:
        chunk_idx = [1]
    if page_num is None:
        page_num = [1]

    for section in sections:
        has_subsections = bool(section.get('subsections'))

        if has_subsections:
            section['subsections'] = add_text_to_sections(section['subsections'], chunk_idx, page_num)
            section['word_count'] = sum(s['word_count'] for s in section['subsections'])
            section['text_length'] = sum(s['text_length'] for s in section['subsections'])
            section.pop('raw_text', None)
        else:
            raw = section.get('raw_text', '')
            pages = []
            for part in raw.split('<<<PAGE_BREAK>>>'):
                content = part.strip()
                # skip bare page-number lines that printers insert
                if _PAGE_NUMBER_RE.fullmatch(content):
                    page_num[0] += 1
                    continue
                if content:
                    pages.append({'page_number': page_num[0], 'content': content})
                page_num[0] += 1

            if not pages:
                pages = [{'page_number': page_num[0], 'content': raw}]

            section['text_length'] = sum(len(p['content']) for p in pages)
            section['word_count'] = sum(len(p['content'].split()) for p in pages)
            section['start_page'] = pages[0]['page_number']
            section['end_page'] = pages[-1]['page_number']
            section['page_contents'] = pages
            chunk_idx[0] += len(pages)
            section.pop('raw_text', None)
            section.pop('subsections', None)

    return sections


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sections_parser_markdown(
    input_path: str,
    output_path: str = None,
    save_md: str = None,
) -> Dict:
    """
    Parse a .md or .html file and return a section hierarchy.
    HTML files are converted to Markdown first (PART/Item headings are injected
    before conversion so ATX headings appear in the output).

    Args:
        input_path:  Path to a .md or .html/.htm file.
        output_path: Optional path to write the JSON result.
        save_md:     Optional path to save the intermediate Markdown (HTML only).
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext in ('.html', '.htm'):
        print(f"Converting HTML to Markdown: {input_path}")
        markdown_text = html_to_markdown(input_path, save_md=save_md)
    elif ext == '.md':
        print(f"Reading Markdown: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .md, .html, or .htm")

    flat_sections = parse_markdown_sections(markdown_text)

    if not flat_sections:
        print("No sections found")
        return None

    hierarchical = build_hierarchy(flat_sections)
    hierarchical = add_text_to_sections(hierarchical)
    all_flat = _flatten(hierarchical)

    result = {
        "filename": input_path,
        "num_sections": len(all_flat),
        "sections": hierarchical,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Output saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse a Markdown (or HTML converted to Markdown) file and extract section hierarchy"
    )
    parser.add_argument(
        "input_path",
        nargs='?',
        default="input/document.md",
        help="Path to a .md or .html file",
    )
    parser.add_argument(
        "--output", "-o",
        default="parsing/parsed_sections_markdown.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--save-md",
        default=None,
        help="Save the intermediate Markdown to this path (HTML input only)",
    )

    args = parser.parse_args()
    print(f"Parsing: {args.input_path}\n")
    sections_parser_markdown(args.input_path, args.output, args.save_md)


if __name__ == "__main__":
    main()
