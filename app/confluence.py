import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

class ConfluenceClient:
    def __init__(self):
        self.base_url = os.getenv("CONFLUENCE_URL")
        self.auth = (os.getenv("CONFLUENCE_EMAIL"), os.getenv("CONFLUENCE_TOKEN"))

    def fetch_page_content(self, page_id: str):
        """Загружает страницу и разбивает на логические куски по заголовкам."""
        url = f"{self.base_url}/rest/api/content/{page_id}?expand=body.storage"
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        
        raw_html = response.json()['body']['storage']['value']
        soup = BeautifulSoup(raw_html, 'html.parser')
        
        chunks = []
        current_section = "General"
        
        # Логика разбивки: ищем заголовки и группируем текст под ними
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
            if element.name in ['h1', 'h2', 'h3']:
                current_section = element.get_text()
            else:
                chunks.append({
                    "page_id": page_id,
                    "topic": current_section,
                    "content": element.get_text(),
                    "metadata": {"source": "confluence"}
                })
        
        return chunks