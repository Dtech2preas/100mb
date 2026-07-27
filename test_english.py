import json

with open('papers.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

english_subject = []
for p in papers:
    name = p.get('name', '')
    if name.lower().startswith('english'):
        english_subject.append(name)

print("Count:", len(english_subject))
print("Sample:", english_subject[:10])
