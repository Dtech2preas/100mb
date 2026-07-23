import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json

requests.packages.urllib3.disable_warnings()

BASE_URL = "https://www.education.gov.za/"

def fetch_links(url, filter_str=None):
    try:
        response = requests.get(url, verify=False, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')

        items = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'LinkClick.aspx' in href and 'fileticket' in href:
                name = a.text.strip()
                if not name or name.lower() == 'download':
                    # sometimes the link is just 'Download', try to get preceding text
                    parent = a.find_parent('td') or a.find_parent('li') or a.find_parent('p')
                    if parent:
                        # naive attempt to extract name
                        text = parent.text.strip()
                        text = text.replace('Download', '').strip()
                        if text:
                            name = text
                if not name or name.lower() == 'download':
                    continue

                full_url = urljoin(BASE_URL, href)
                if '&forcedownload' in full_url:
                    full_url = full_url.split('&forcedownload')[0]

                # Deduplicate by URL
                items.append({"name": name, "file": full_url})
        return items
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def update_json(filename, new_items):
    import os
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = []

    existing_urls = {i['file'] for i in existing if 'file' in i}
    added = 0

    for item in new_items:
        if item['file'] not in existing_urls:
            existing.append(item)
            existing_urls.add(item['file'])
            added += 1

    if added > 0:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=4)
        print(f"Added {added} new items to {filename}.")
    else:
        print(f"No new items for {filename}.")


# We know tabid 4684 is Grade 12 PATs 2024 (and 2025 is there probably?).
# Tab 611 is SBA Tasks.
# Mind the gap is at /Curriculum/LearningandTeachingSupportMaterials(LTSM)/MindtheGapStudyGuides.aspx

print("Scraping Study Guides...")
guides = fetch_links("https://www.education.gov.za/Curriculum/LearningandTeachingSupportMaterials(LTSM)/MindtheGapStudyGuides.aspx")
update_json('guide.json', guides)

print("Scraping SBA Tasks...")
# Let's check SBA Tasks 2014 URL. There might be a newer one, but let's grab this one first.
sbas = fetch_links("https://www.education.gov.za/Default.aspx?tabid=611")
update_json('SBA.json', sbas)

print("Scraping PATs...")
# Let's check Grade 12 PATs.
pats = fetch_links("https://www.education.gov.za/Default.aspx?tabid=4684")
update_json('PAT.json', pats)
