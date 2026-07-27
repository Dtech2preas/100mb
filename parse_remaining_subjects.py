import json
import re

def extract_papers(search_term, filename, search_condition=None):
    with open('papers.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)

    extracted = []
    for p in papers:
        name = p.get('name', '')
        url = p.get('file', '')
        name_lower = name.lower()

        # Determine if it matches based on custom condition or general term
        matches = False
        if search_condition:
            matches = search_condition(name_lower)
        else:
            matches = search_term in name_lower

        if matches:
            # extract year
            match = re.search(r'\b(19|20)\d{2}\b', name)
            year = match.group(0) if match else "Unknown"
            extracted.append((name, year, url))

    with open(filename, 'w', encoding='utf-8') as out:
        title_header = search_term.title() if not search_condition else filename.replace('.txt', '').replace('_', ' ').title()
        out.write(f"Total {title_header} Papers: {len(extracted)}\n\n")
        for name, year, url in extracted:
            out.write(f"Name: {name}\nYear: {year}\nURL: {url}\n\n")

    print(f"Total {title_header} papers found: {len(extracted)}")

extract_papers("english", "english.txt", lambda n: n.startswith("english"))
extract_papers("engineering graphics", "egd.txt")
extract_papers("technical science", "technical_science.txt")
extract_papers("civil technology", "civil_technology.txt")
