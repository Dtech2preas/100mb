import json
import re

with open('papers.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

mathematics_papers = []
for p in papers:
    name = p.get('name', '')
    url = p.get('file', '')
    name_lower = name.lower()
    if 'mathematics' in name_lower and 'technical' not in name_lower and 'mathematical literacy' not in name_lower:
        # try to extract year
        match = re.search(r'\b(19|20)\d{2}\b', name)
        year = match.group(0) if match else "Unknown"
        mathematics_papers.append((name, year, url))

with open('mathematics.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total Mathematics Papers: {len(mathematics_papers)}\n\n")
    for name, year, url in mathematics_papers:
        out.write(f"Name: {name}\nYear: {year}\nURL: {url}\n\n")

print(f"Total Mathematics papers found: {len(mathematics_papers)}")
