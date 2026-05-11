import json

with open('output/parsed_sections_html.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

sections = []

def traverse(items, level=0):
    for s in items:
        sections.append({
            'title': s['title'],
            'level': s['level'],
            'text_len': s.get('text_length', 0),
            'words': s.get('word_count', 0)
        })
        if 'subsections' in s:
            traverse(s['subsections'], level+1)

traverse(data['sections'])

print("Top 15 sections by text length:\n")
print(f"{'Section Title':<70} {'Level':<7} {'Chars':<10} {'Words':<10}")
print("=" * 100)

for s in sorted(sections, key=lambda x: x['text_len'], reverse=True)[:15]:
    title = s['title'][:65] + "..." if len(s['title']) > 65 else s['title']
    print(f"{title:<70} {s['level']:<7} {s['text_len']:<10} {s['words']:<10}")
