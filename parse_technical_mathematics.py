import json
import re

with open('papers.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

technical_mathematics_papers = []
for p in papers:
    name = p.get('name', '')
    url = p.get('file', '')

    if 'technical mathematics' in name.lower():
        # try to extract year
        match = re.search(r'\b(19|20)\d{2}\b', name)
        year = match.group(0) if match else "Unknown"
        technical_mathematics_papers.append((name, year, url))

with open('technical_mathematics.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total Technical Mathematics Papers: {len(technical_mathematics_papers)}\n\n")
    for name, year, url in technical_mathematics_papers:
        out.write(f"Name: {name}\nYear: {year}\nURL: {url}\n\n")

print(f"Total Technical Mathematics papers found: {len(technical_mathematics_papers)}")
