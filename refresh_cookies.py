"""
refresh_cookies.py — Автоматически обновляет cookies в .env через Selenium

Использование:
    python refresh_cookies.py          # оба
    python refresh_cookies.py --ai     # только AI Battleground
    python refresh_cookies.py --conf   # только Confluence
"""

import sys
import time
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.keys import Keys
except ImportError:
    print("❌ Установи: pip install selenium")
    sys.exit(1)

ENV_FILE = Path(__file__).parent / ".env"
CHROMEDRIVER = Path(__file__).parent / "chromedriver.exe"


def get_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    if CHROMEDRIVER.exists():
        service = Service(str(CHROMEDRIVER))
        driver = webdriver.Chrome(service=service, options=options)
    else:
        print(f"⚠️  chromedriver.exe не найден в {CHROMEDRIVER}")
        print("   Положи chromedriver.exe в папку проекта")
        sys.exit(1)

    return driver


def cookies_to_string(cookies: list) -> str:
    return "; ".join(f"{c['name']}={c['value']}" for c in cookies)


def update_env(key: str, value: str):
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"✅ {key} обновлен ({len(value)} символов)")


def refresh_ai():
    print("\n🤖 Обновление AI Battleground cookies...")
    print("=" * 50)
    driver = get_driver()

    try:
        driver.get("https://ai-battleground.azercell.com")
        print("\n👆 Залогинься: email → Continue → подтверди на телефоне → Stay signed in")
        print("   Нажми Enter когда окажешься на главной странице с чатом...")
        input("   [Enter] ")

        # Отправляем сообщение чтобы обновить session cookies
        try:
            textarea = driver.find_element("tag name", "textarea")
            textarea.send_keys("hi")
            time.sleep(1)
            textarea.send_keys(Keys.RETURN)
            time.sleep(3)
            print("   Сообщение отправлено — session cookies обновлены")
        except Exception:
            print("   Используем текущие cookies...")

        update_env("AI_BATTLEGROUND_COOKIES", cookies_to_string(driver.get_cookies()))

    finally:
        driver.quit()


def refresh_confluence():
    print("\n📚 Обновление Confluence cookies...")
    print("=" * 50)
    driver = get_driver()

    try:
        driver.get("https://confluence.azercell.com")
        print("\n👆 Залогинься если нужно")
        print("   Нажми Enter когда страница загрузится...")
        input("   [Enter] ")

        time.sleep(2)
        update_env("CONFLUENCE_COOKIES", cookies_to_string(driver.get_cookies()))

    finally:
        driver.quit()


def main():
    args = sys.argv[1:]
    if "--ai" in args:
        refresh_ai()
    elif "--conf" in args:
        refresh_confluence()
    else:
        refresh_ai()
        refresh_confluence()

    print("\n🎉 Готово! Перезапусти backend:")
    print("   python -m app.main")


if __name__ == "__main__":
    main()