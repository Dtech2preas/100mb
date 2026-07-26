import json
import re

with open('papers.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

accounting_papers = []
for p in papers:
    name = p.get('name', '')
    url = p.get('file', '')

    if 'accounting' in name.lower():
        # try to extract year
        match = re.search(r'\b(19|20)\d{2}\b', name)
        year = match.group(0) if match else "Unknown"
        accounting_papers.append((name, year, url))

with open('accounting_papers.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total Accounting Papers: {len(accounting_papers)}\n\n")
    for name, year, url in accounting_papers:
        out.write(f"Name: {name}\nYear: {year}\nURL: {url}\n\n")

print(f"Total Accounting papers found: {len(accounting_papers)}")
