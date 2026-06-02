import os
import re
import logging
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

FRAME_LIMIT = 100


def parse_figma_url(url: str) -> Optional[Dict[str, str]]:
    pattern = r"figma\.com/(?:file|design|proto)/([a-zA-Z0-9]+)"
    match = re.search(pattern, url)
    if not match:
        return None

    file_key = match.group(1)
    node_id_match = re.search(r"node-id=([^&]+)", url)
    node_id = node_id_match.group(1).replace("-", ":") if node_id_match else None

    return {"file_key": file_key, "node_id": node_id}


def fetch_figma_frames(url: str) -> List[Dict[str, str]]:
    load_dotenv(override=True)
    FIGMA_TOKEN = os.getenv("FIGMA_TOKEN", "")
    proxy_password = os.getenv("CORP_PROXY_PASSWORD", "")

    if not FIGMA_TOKEN:
        raise ValueError("FIGMA_TOKEN not set in .env")

    parsed = parse_figma_url(url)
    if not parsed:
        raise ValueError(f"Invalid Figma URL: {url}")

    file_key = parsed["file_key"]
    node_id = parsed["node_id"]

    headers = {"X-Figma-Token": FIGMA_TOKEN}
    proxies = {
        "http":  f"http://garayevh:{proxy_password}@proxy.azercell.com:8080",
        "https": f"http://garayevh:{proxy_password}@proxy.azercell.com:8080",
    }

    if node_id:
        # Конкретный node — берём его children
        api_url = f"https://api.figma.com/v1/files/{file_key}/nodes?ids={node_id}"
        resp = requests.get(api_url, headers=headers, proxies=proxies, verify=False, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        frames = []
        for nid, node_data in data.get("nodes", {}).items():
            doc = node_data.get("document", {})
            for child in doc.get("children", []):
                if child.get("type") in ("FRAME", "COMPONENT", "COMPONENT_SET", "GROUP"):
                    frames.append({
                        "id": child["id"],
                        "name": child["name"],
                        "type": child["type"],
                    })
                if len(frames) >= FRAME_LIMIT:
                    break
        logger.info(f"[figma] node mode: {len(frames)} frames from node {node_id}")
        return frames

    else:
        # Весь файл — только top-level frames, лимит по страницам
        api_url = f"https://api.figma.com/v1/files/{file_key}?depth=2"
        resp = requests.get(api_url, headers=headers, proxies=proxies, verify=False, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        frames = []
        pages = data.get("document", {}).get("children", [])
        for page in pages:
            for child in page.get("children", []):
                if child.get("type") in ("FRAME", "COMPONENT", "COMPONENT_SET"):
                    frames.append({
                        "id": child["id"],
                        "name": child["name"],
                        "type": child["type"],
                        "page": page["name"],
                    })
                if len(frames) >= FRAME_LIMIT:
                    break
            if len(frames) >= FRAME_LIMIT:
                break

        logger.info(f"[figma] file mode: {len(frames)} frames from {len(pages)} pages")
        return frames