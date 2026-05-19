import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenceClient:
    def __init__(self):
        self.base_url = os.getenv("CONFLUENCE_URL", "").rstrip("/")

        # SSO через cookies (как AI_BATTLEGROUND_COOKIES — обновлять когда истекают)
        cookies_str = os.getenv("CONFLUENCE_COOKIES", "")
        self.cookies = {}
        for part in cookies_str.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                self.cookies[k.strip()] = v.strip()

    def _get(self, url: str, params: dict = None) -> requests.Response:
        """Общий метод запроса с cookies."""
        response = requests.get(url, params=params, cookies=self.cookies)
        response.raise_for_status()
        return response

    # -------------------------------------------------------------------------
    # 1. SEARCH
    # -------------------------------------------------------------------------
    def search_pages(self, query: str, limit: int = 20) -> List[Dict]:
        url = f"{self.base_url}/rest/api/content/search"
        params = {
            "cql": f'title ~ "{query}" AND type = page',
            "limit": limit,
            "expand": "space,ancestors"
        }

        data = self._get(url, params).json()
        results = data.get("results", [])

        pages = []
        for r in results:
            space = r.get("space", {})
            space_name = space.get("name", space.get("key", ""))
            space_key = space.get("key", "")
            webui = r.get("_links", {}).get("webui", "")

            pages.append({
                "page_id": r["id"],
                "title": r["title"],
                "space": space_name,
                "space_key": space_key,
                "breadcrumb": " > ".join(
                    [a["title"] for a in r.get("ancestors", [])] + [r["title"]]
                ),
                "url": f"{self.base_url}{webui}"
            })

        return pages

    # -------------------------------------------------------------------------
    # 2. BULK SYNC
    # -------------------------------------------------------------------------
    def bulk_sync_space(self, space_key: str) -> List[Dict]:
        logger.info(f"Starting bulk sync for space: {space_key}")
        all_chunks = []
        start = 0
        limit = 50

        while True:
            url = f"{self.base_url}/rest/api/content"
            params = {
                "spaceKey": space_key,
                "type": "page",
                "status": "current",
                "expand": "body.storage,ancestors,version,space",
                "start": start,
                "limit": limit
            }

            data = self._get(url, params).json()
            pages = data.get("results", [])
            if not pages:
                break

            for page in pages:
                chunks = self._parse_and_chunk(page)
                all_chunks.extend(chunks)
                logger.info(f"  Synced: '{page['title']}' → {len(chunks)} chunks")

            if len(pages) < limit:
                break
            start += limit

        logger.info(f"Bulk sync complete. Total chunks: {len(all_chunks)}")
        return all_chunks

    # -------------------------------------------------------------------------
    # 3. FETCH SELECTED PAGES
    # -------------------------------------------------------------------------
    def fetch_selected_pages(self, page_ids: List[str]) -> List[Dict]:
        all_chunks = []

        for page_id in page_ids:
            url = f"{self.base_url}/rest/api/content/{page_id}"
            params = {"expand": "body.storage,ancestors,version,space"}

            page = self._get(url, params).json()
            chunks = self._parse_and_chunk(page)
            all_chunks.extend(chunks)
            logger.info(f"Fetched: '{page['title']}' → {len(chunks)} chunks")

        return all_chunks

    # -------------------------------------------------------------------------
    # 4. CHUNKING
    # -------------------------------------------------------------------------
    def _parse_and_chunk(self, page: Dict) -> List[Dict]:
        page_id = page["id"]
        page_title = page["title"]
        raw_html = page.get("body", {}).get("storage", {}).get("value", "")
        space_key = page.get("space", {}).get("key", "")
        ancestors = page.get("ancestors", [])
        breadcrumb_parts = [a["title"] for a in ancestors] + [page_title]

        soup = BeautifulSoup(raw_html, "html.parser")
        chunks = []
        current_section = "General"
        current_text_parts = []
        chunk_index = 0

        def save_chunk(section: str, text_parts: List[str]):
            nonlocal chunk_index
            text = " ".join(text_parts).strip()
            if not text or len(text) < 30:
                return
            chunks.append({
                "page_id": page_id,
                "page_title": page_title,
                "section": section,
                "content": text,
                "breadcrumb": " > ".join(breadcrumb_parts + [section]),
                "chunk_index": chunk_index,
                "metadata": {
                    "source": "confluence",
                    "space_key": space_key,
                    "version": page.get("version", {}).get("number", 1),
                }
            })
            chunk_index += 1

        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
            if element.name in ["h1", "h2", "h3", "h4"]:
                save_chunk(current_section, current_text_parts)
                current_section = element.get_text(strip=True)
                current_text_parts = []
            else:
                text = element.get_text(strip=True)
                if text:
                    current_text_parts.append(text)

        save_chunk(current_section, current_text_parts)

        if not chunks:
            full_text = soup.get_text(separator=" ", strip=True)
            if full_text:
                chunks.append({
                    "page_id": page_id,
                    "page_title": page_title,
                    "section": "General",
                    "content": full_text[:2000],
                    "breadcrumb": " > ".join(breadcrumb_parts),
                    "chunk_index": 0,
                    "metadata": {
                        "source": "confluence",
                        "space_key": space_key,
                        "version": page.get("version", {}).get("number", 1),
                    }
                })

        return chunks