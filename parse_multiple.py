import json
import re

def extract_papers(search_term, filename, exclude_terms=None):
    if exclude_terms is None:
        exclude_terms = []

    with open('papers.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)

    extracted = []
    for p in papers:
        name = p.get('name', '')
        url = p.get('file', '')
        name_lower = name.lower()

        # Check if search term is in name, and none of exclude terms are in name
        if search_term in name_lower:
            should_exclude = any(ext in name_lower for ext in exclude_terms)
            if not should_exclude:
                # Special handling for "English" as many papers say "(English)"
                # To get the English subject papers, they typically start with "English"
                if search_term == "english" and not name_lower.startswith("english"):
                    continue

                # try to extract year
                match = re.search(r'\b(19|20)\d{2}\b', name)
                year = match.group(0) if match else "Unknown"
                extracted.append((name, year, url))

    with open(filename, 'w', encoding='utf-8') as out:
        out.write(f"Total {search_term.title()} Papers: {len(extracted)}\n\n")
        for name, year, url in extracted:
            out.write(f"Name: {name}\nYear: {year}\nURL: {url}\n\n")

    print(f"Total {search_term.title()} papers found: {len(extracted)}")

extract_papers("english", "english.txt")
extract_papers("egd", "egd.txt", exclude_terms=["memo"]) # Might have to tweak if EGD is expanded
extract_papers("engineering graphics and design", "egd_full.txt")
extract_papers("technical science", "technical_science.txt")
extract_papers("civil technology", "civil_technology.txt")
