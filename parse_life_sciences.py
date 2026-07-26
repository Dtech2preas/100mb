import json
import re

with open('papers.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

life_sciences_papers = []
for p in papers:
    name = p.get('name', '')
    url = p.get('file', '')

    if 'life science' in name.lower():
        # try to extract year
        match = re.search(r'\b(19|20)\d{2}\b', name)
        year = match.group(0) if match else "Unknown"
        life_sciences_papers.append((name, year, url))

with open('life_sciences.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total Life Sciences Papers: {len(life_sciences_papers)}\n\n")
    for name, year, url in life_sciences_papers:
        out.write(f"Name: {name}\nYear: {year}\nURL: {url}\n\n")

print(f"Total Life Sciences papers found: {len(life_sciences_papers)}")
