# src/crawler/crawler.py
from __future__ import annotations

import os
import re
import time
from io import BytesIO
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse, urlsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ====== 경로 설정 ======
SEED_FILE = "data/raw/gyms_seed.csv"          # ← 전체 경로 수정
OUTPUT_FILE = "data/processed/gym_prices.csv"

# ====== 네트워크 설정 ======
try:
    import cloudscraper  # 선택: cf 우회 (uv add cloudscraper)
    SESSION = cloudscraper.create_scraper()
except Exception:
    SESSION = requests.Session()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.co.kr/",
    "Cache-Control": "no-cache",
}

# 지도/플랫폼 도메인(가격 없음) 스킵
SKIP_DOMAINS = (
    "google.co", "google.com", "goo.gl",
    "naver.com", "map.naver.com", "m.place.naver.com",
    "place.map.kakao.com", "kko.to"
)

# ====== OCR 사용 여부 (원하면 True로) ======
OCR_ENABLED = False  # True 로 바꾸면 이미지 가격 OCR 시도
# OCR 사용 시 필요:
#   uv add pillow pytesseract
#   brew install tesseract
if OCR_ENABLED:
    from PIL import Image
    import pytesseract  # type: ignore
    # 필요 시 명시:
    # pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ====== 가격 정규화/추출 ======
# “3개월 400,000원”, “6개월 70만원”, “12개월 1,200,000 원”, “월 15만원”, “3개월-29만9천원” 등 포용
PRICE_PATTERNS = [
    re.compile(r'(\d{1,2})\s*개월[^0-9\n]*([0-9][0-9,\.]*\s*만?\s*원)', re.I),
    re.compile(r'(?:월\s*권|월\s*이용|1\s*개월|월)[^0-9\n]*([0-9][0-9,\.]*\s*만?\s*원)', re.I),
    re.compile(r'(\d{1,2})\s*개월[^\n]*?([0-9][0-9,\.]*\s*만?\s*원)', re.I),
]
WON_NUMBER = re.compile(r'([0-9][0-9,\.]*)\s*원', re.I)
MANWON = re.compile(r'([0-9]+)(?:[.]([0-9]+))?\s*만\s*원', re.I)

def normalize_price_to_krw(text: str) -> Optional[int]:
    """'15만원', '150,000원', '1,200,000 원' 등을 정수 KRW로 변환."""
    t = text.replace(" ", "")
    m = MANWON.search(t)
    if m:
        whole = int(m.group(1))
        frac = int(m.group(2)) if m.group(2) else 0
        return whole * 10_000 + (frac * (10 ** (4 - len(m.group(2)))) if m.group(2) else 0)
    m = WON_NUMBER.search(t)
    if m:
        return int(m.group(1).replace(",", "").replace(".", ""))
    return None

def robust_get(url: str, max_retry: int = 3, sleep_sec: float = 1.5) -> Optional[str]:
    """헤더/리트라이/HTTPS 재시도/리퍼러 교체 포함 GET. 최종 실패 시 None."""
    for i in range(max_retry):
        try:
            resp = SESSION.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
            # http → https 재시도
            if resp.status_code == 403 and url.startswith("http://"):
                https_url = "https://" + url.split("://", 1)[1]
                resp = SESSION.get(https_url, headers=HEADERS, timeout=20, allow_redirects=True)
            # 리퍼러를 해당 사이트 루트로 교체
            if resp.status_code == 403:
                host = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
                h2 = dict(HEADERS); h2["Referer"] = host
                resp = SESSION.get(url, headers=h2, timeout=20, allow_redirects=True)

            if resp.status_code in (403, 429) and i < max_retry - 1:
                time.sleep(sleep_sec); continue
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] Failed to fetch {url}: {e}")
                return None
            time.sleep(sleep_sec)
    return None

def extract_prices_from_text(html: str) -> List[Dict]:
    """페이지 전체 텍스트에서 가격 패턴 추출."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    found: List[Dict] = []

    for pat in PRICE_PATTERNS:
        for m in pat.finditer(text):
            if len(m.groups()) == 2:
                term, price_txt = m.group(1), m.group(2)
            else:
                term, price_txt = "", m.group(1)
            price_krw = normalize_price_to_krw(price_txt)
            found.append({
                "plan_type": "",
                "term_months": int(term) if term.isdigit() else "",
                "price_text": price_txt,
                "price_krw": price_krw if price_krw is not None else "",
            })

    # 중복 제거
    uniq = {(r["term_months"], r["price_text"]): r for r in found}
    return list(uniq.values())

def find_price_image_urls(html: str, base_url: str) -> List[str]:
    """가격표로 보이는 이미지 src 후보 수집."""
    soup = BeautifulSoup(html, "html.parser")
    imgs = soup.find_all("img")
    candidates: List[str] = []
    for img in imgs:
        alt = (img.get("alt") or "").lower()
        src = img.get("src") or ""
        if not src:
            continue
        if any(k in alt for k in ("가격", "요금", "이용요금", "멤버십", "membership", "price", "fee")) \
           or any(k in src.lower() for k in ("price", "fee", "membership", "요금", "가격")):
            candidates.append(urljoin(base_url, src))
    return list(dict.fromkeys(candidates))  # 중복 제거

def ocr_image_url(image_url: str) -> str:
    """이미지 URL에서 텍스트 OCR (OCR_ENABLED=True일 때만)."""
    if not OCR_ENABLED:
        return ""
    try:
        r = SESSION.get(image_url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        text = pytesseract.image_to_string(img, lang="kor+eng")
        return text.strip()
    except Exception as e:
        print(f"[WARN] OCR failed {image_url}: {e}")
        return ""

def extract_prices_from_images(html: str, page_url: str) -> List[Dict]:
    """이미지 가격 추출 (OCR). OCR 비활성화면 빈 리스트."""
    if not OCR_ENABLED:
        return []
    results: List[Dict] = []
    for img_url in find_price_image_urls(html, page_url):
        print(f"[INFO] OCR price image: {img_url}")
        text = ocr_image_url(img_url)
        if not text:
            continue
        for pat in PRICE_PATTERNS:
            for m in pat.finditer(text):
                if len(m.groups()) == 2:
                    term, price_txt = m.group(1), m.group(2)
                else:
                    term, price_txt = "", m.group(1)
                price_krw = normalize_price_to_krw(price_txt)
                results.append({
                    "plan_type": "",
                    "term_months": int(term) if term.isdigit() else "",
                    "price_text": price_txt,
                    "price_krw": price_krw if price_krw is not None else "",
                })
    uniq = {(r["term_months"], r["price_text"]): r for r in results}
    return list(uniq.values())

def looks_like_map(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(d in host for d in SKIP_DOMAINS)

# ----- 도메인 전용 파서 (수율↑) -----
def parse_spoany(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict] = []
    for node in soup.find_all(["li", "p", "span", "div"]):
        t = (node.get_text(" ", strip=True) or "")
        if "원" in t and any(k in t for k in ["개월", "월"]):
            price_krw = normalize_price_to_krw(t) or ""
            m = re.search(r'(\d{1,2})\s*개월', t)
            term = int(m.group(1)) if m else ""
            out.append({"plan_type": "", "term_months": term, "price_text": t, "price_krw": price_krw})
    return out

def parse_modoo(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict] = []
    # 표 형태 선호
    rows = soup.select("table tr")
    if not rows:
        rows = soup.select(".se-table tr, .se_sectionArea table tr")
    for tr in rows:
        txt = tr.get_text(" ", strip=True)
        if "원" in txt:
            price_krw = normalize_price_to_krw(txt) or ""
            m = re.search(r'(\d{1,2})\s*개월', txt)
            term = int(m.group(1)) if m else ""
            out.append({"plan_type": "", "term_months": term, "price_text": txt, "price_krw": price_krw})
    return out

DOMAIN_PARSERS = {
    "m.spoany.co.kr": parse_spoany,
    "spoany.co.kr": parse_spoany,
    "modoo.at": parse_modoo,
}

def extract_prices(html: str, page_url: str) -> List[Dict]:
    host = urlparse(page_url).netloc.lower()
    # 1) 도메인 전용 파서
    for domain, fn in DOMAIN_PARSERS.items():
        if domain in host:
            data = fn(html)
            if data:
                return data
    # 2) 일반 텍스트 정규식
    data = extract_prices_from_text(html)
    if data:
        return data
    # 3) 이미지 OCR
    return extract_prices_from_images(html, page_url)

def choose_url(row: dict) -> Optional[str]:
    """price_page_url > website_url > (website_url 기반 price 후보 경로) 순.
       지도/플랫폼은 스킵."""
    # 1순위: 명시된 가격 페이지
    for key in ("price_page_url", "website_url"):
        url = (row.get(key) or "").strip()
        if url.startswith("http") and not looks_like_map(url):
            return url
    # 2순위: website_url 기반 후보 경로 추정
    base = (row.get("website_url") or "").strip()
    if base.startswith("http") and not looks_like_map(base):
        for p in ["price", "prices", "membership", "멤버십", "요금", "이용요금", "program", "programs"]:
            cand = urljoin(base.rstrip("/") + "/", p)
            return cand
    # 3순위: source_url(지도)은 스킵
    return None

def main():
    if not os.path.exists(SEED_FILE):
        print(f"[ERROR] Seed file not found: {SEED_FILE}")
        return

    # 전화번호/우편번호 등의 선행 0 유지를 위해 전부 문자열로
    df = pd.read_csv(SEED_FILE, dtype=str).fillna("")
    # website_url 없는 행은 일단 제외(수율↑). 필요시 주석 처리.
    if "website_url" in df.columns:
        df = df[df["website_url"].str.startswith("http")]

    rows: List[Dict] = []
    for _, row in df.iterrows():
        name = (row.get("name") or "").strip()
        if not name:
            continue

        crawl_url = choose_url(row)
        display_url = row.get("price_page_url") or row.get("website_url") or row.get("source_url") or ""

        if not crawl_url:
            print(f"[SKIP] No crawlable URL for {name}")
            rows.append({
                "name": name,
                "address": row.get("address", ""),
                "phone": row.get("phone", ""),
                "plan_type": "",
                "term_months": "",
                "price_krw": "",
                "price_text": "NO_PRICE_FOUND",
                "source_url": display_url,
            })
            continue

        print(f"[INFO] Fetching prices for: {name} ({crawl_url})")
        html = robust_get(crawl_url)
        prices = extract_prices(html or "", crawl_url) if html else []

        if not prices:
            rows.append({
                "name": name,
                "address": row.get("address", ""),
                "phone": row.get("phone", ""),
                "plan_type": "",
                "term_months": "",
                "price_krw": "",
                "price_text": "NO_PRICE_FOUND",
                "source_url": crawl_url,
            })
        else:
            for p in prices:
                rows.append({
                    "name": name,
                    "address": row.get("address", ""),
                    "phone": row.get("phone", ""),
                    "plan_type": p.get("plan_type", ""),
                    "term_months": p.get("term_months", ""),
                    "price_krw": p.get("price_krw", ""),
                    "price_text": p.get("price_text", ""),
                    "source_url": crawl_url,
                })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
