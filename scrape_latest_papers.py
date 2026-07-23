import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib.parse
import json
import os
import re

# Disable warnings for unverified HTTPS requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.education.gov.za/"
INDEX_URL = "https://www.education.gov.za/Curriculum/NationalSeniorCertificate(NSC)Examinations/NSCPastExaminationpapers.aspx"
JSON_FILE = "papers.json"

def fetch_soup(url):
    try:
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_year_pages():
    soup = fetch_soup(INDEX_URL)
    if not soup:
        return []

    pages = []
    # Target 2024, 2025, 2026. (Also 2023 just in case some are missing)
    target_years = [str(year) for year in range(2008, 2027)]

    for link in soup.find_all('a'):
        text = link.text.strip()
        if any(year in text for year in target_years):
            href = link.get('href', '')
            if href:
                full_url = urljoin(BASE_URL, href)
                # Handle encoded redirect URLs
                if 'link=' in href:
                    parsed = urllib.parse.urlparse(href)
                    query = urllib.parse.parse_qs(parsed.query)
                    if 'link' in query:
                        actual_link = query['link'][0]
                        if actual_link.isdigit():
                            actual_link = full_url
                        elif not actual_link.startswith('http'):
                            actual_link = urljoin(BASE_URL, actual_link)
                        pages.append((text, actual_link))
                else:
                    pages.append((text, full_url))
    return pages

def extract_papers_from_page(page_name, url):
    print(f"Scraping page: {page_name} ({url})")
    soup = fetch_soup(url)
    if not soup:
        return []

    # Extract contextual suffix from page_name (e.g. "2024 November NSC Examination Papers" -> "November 2024")
    # or just use the page name as a fallback suffix
    match = re.search(r'(20\d{2})\s+([A-Za-z/]+)', page_name)
    if match:
        year, month = match.groups()
        suffix = f"{month} {year}"
    else:
        suffix = page_name

    papers = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        name = a.text.strip()

        # We look for file links which usually have LinkClick.aspx and fileticket
        if 'LinkClick.aspx' in href and 'fileticket' in href:
            # Skip forcedownload generic links if they are named "Download"
            if name.lower() == 'download' or name == '':
                continue

            full_url = urljoin(BASE_URL, href)
            # Remove any trailing url parameters like forcedownload=true just in case, or leave it.
            # Current JSON has fileticket, tabid, portalid, mid.
            # We'll strip forcedownload to be cleaner
            if '&forcedownload' in full_url:
                full_url = full_url.split('&forcedownload')[0]

            formatted_name = f"{name} {suffix}"
            papers.append({
                "name": formatted_name,
                "file": full_url
            })

    return papers

def main():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    print(f"Loaded {len(existing_data)} existing papers.")

    # Track existing URLs to prevent duplicates
    existing_urls = {item['file'] for item in existing_data if 'file' in item}
    existing_names = {item['name'] for item in existing_data if 'name' in item}

    year_pages = get_year_pages()
    new_papers = []

    for page_name, page_url in year_pages:
        scraped = extract_papers_from_page(page_name, page_url)
        for paper in scraped:
            # Prevent duplicates by exact URL or exact Name
            if paper['file'] not in existing_urls and paper['name'] not in existing_names:
                new_papers.append(paper)
                existing_urls.add(paper['file'])
                existing_names.add(paper['name'])

    print(f"Found {len(new_papers)} new papers.")

    if new_papers:
        existing_data.extend(new_papers)
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
        print("Successfully updated papers.json.")
    else:
        print("No new papers to add.")

if __name__ == "__main__":
    main()
