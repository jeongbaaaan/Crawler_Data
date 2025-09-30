# src/crawler/crawler.py
from __future__ import annotations

import os
import re
import time
from collections import deque
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ====== 경로 설정 ======
SEED_FILE = "data/raw/gyms_seed.csv"            # 입력(seed)
OUTPUT_FILE = "data/processed/gym_prices.csv"   # 출력(result)

# ====== 디버그 설정 ======
DEBUG_SAVE_HTML = True
DEBUG_DIR = "data/debug_html"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug_html(name: str, url: str, html: str):
    if not DEBUG_SAVE_HTML or not html:
        return
    host = urlparse(url).netloc.replace(":", "_")
    fname = f"{name}_{host}.html".replace(" ", "_")
    path = os.path.join(DEBUG_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DEBUG] saved html -> {path}")

# ====== 네트워크 설정 ======
try:
    import cloudscraper  # 선택: cf 우회 (pip install cloudscraper / uv add cloudscraper)
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

# SSL 인증서 이슈 대비: verify=False 경고 끄기
import urllib3
from requests.exceptions import SSLError, HTTPError
from urllib3.exceptions import SSLError as URLLibSSLError, InsecureRequestWarning
urllib3.disable_warnings(category=InsecureRequestWarning)

# 지도/플랫폼 도메인(가격 없음) 스킵
SKIP_DOMAINS = (
    "google.co", "google.com", "goo.gl",
    "naver.com", "map.naver.com", "m.place.naver.com",
    "place.map.kakao.com", "kko.to"
)

# 가격/요금 페이지 힌트 키워드 (내부 링크 탐색용)
PRICE_URL_HINTS = [
    "price", "prices", "membership", "멤버십", "요금", "이용요금",
    "program", "programs", "fee", "이용권"
]

# BFS 탐색에서 제외할 링크(로그인/약관/뉴스 등)
BLOCK_URL_PARTS = (
    "login", "signin", "signup", "join", "cart", "checkout",
    "terms", "privacy", "policy", "recruit", "career", "jobs",
    "story", "news", "board", "review", "event", "notice", "faq",
    "blog", "press", "instagram", "kakao", "facebook", "youtube"
)

# ====== OCR 사용 여부 (원하면 True로) ======
OCR_ENABLED = False  # True 로 바꾸면 이미지 가격 OCR 시도
# OCR 사용 시 필요:
#   pip install pillow pytesseract
#   brew install tesseract  # mac
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

# 긴 문장 속에서 가격 구문만 뽑기 위한 패턴 (예: "3개월 9만원" / "9만원")
PRICE_PHRASE = re.compile(r'((\d{1,2})\s*개월)?[^0-9\n]*([0-9][0-9,\.]*\s*만?\s*원)', re.I)

# 너무 긴 줄(리뷰/공지)을 거르기 위한 길이 제한 (완화)
MAX_LINE_LEN = 160

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

def extract_price_phrase(text: str):
    """
    긴 문장 안에서 '3개월 9만원', '9만원' 같은 가격 구문만 잘라서 반환.
    (term_months, price_text, price_krw) 형태로 리턴. 없으면 None.
    """
    m = PRICE_PHRASE.search(text)
    if not m:
        return None
    term = m.group(2) or ""
    price_txt = m.group(3)
    price_krw = normalize_price_to_krw(price_txt)
    return (int(term) if term.isdigit() else "", price_txt, price_krw if price_krw is not None else "")

def robust_get(url: str, max_retry: int = 3, sleep_sec: float = 1.5) -> Optional[str]:
    """헤더/리트라이/HTTPS 재시도/리퍼러 교체 포함 GET.
       ★ bytes→str 디코딩을 직접 처리해 한글 깨짐 방지."""
    for i in range(max_retry):
        try:
            resp = SESSION.get(url, headers=HEADERS, timeout=20, allow_redirects=True, verify=True)
            resp.raise_for_status()

            raw = resp.content  # bytes 우선

            # 1) HTTP 헤더에서 charset 찾기
            enc = None
            ct = resp.headers.get("Content-Type", "")
            m = re.search(r"charset=([^\s;]+)", ct, re.I)
            if m:
                enc = m.group(1).strip('\'"').lower()

            # 2) 헤더에 없으면 <meta charset="..."> 스니프
            if not enc:
                m = re.search(rb'<meta[^>]+charset=["\']?\s*([a-zA-Z0-9_\-]+)', raw, re.I)
                if m:
                    try:
                        enc = m.group(1).decode("ascii", "ignore").lower()
                    except Exception:
                        enc = None

            # 3) 그래도 없으면 requests의 apparent_encoding 또는 utf-8
            if not enc:
                enc = (getattr(resp, "apparent_encoding", None) or "utf-8").lower()

            try:
                text = raw.decode(enc, errors="replace")
            except Exception:
                text = raw.decode("utf-8", errors="replace")

            return text

        except (SSLError, URLLibSSLError):
            # SSL 재시도 (verify=False)
            try:
                resp = SESSION.get(url, headers=HEADERS, timeout=20, allow_redirects=True, verify=False)
                resp.raise_for_status()
                raw = resp.content
                m = re.search(rb'<meta[^>]+charset=["\']?\s*([a-zA-Z0-9_\-]+)', raw, re.I)
                enc = m.group(1).decode("ascii", "ignore").lower() if m else "utf-8"
                return raw.decode(enc, errors="replace")
            except Exception as e_inner:
                if i == max_retry - 1:
                    print(f"[ERROR] Failed to fetch {url} after retries: {e_inner}")
                    return None

        except HTTPError as e:
            if e.response.status_code == 403:
                host = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}/"
                h2 = dict(HEADERS); h2["Referer"] = host
                try:
                    resp = SESSION.get(url, headers=h2, timeout=20, allow_redirects=True, verify=True)
                    resp.raise_for_status()
                    raw = resp.content
                    m = re.search(rb'<meta[^>]+charset=["\']?\s*([a-zA-Z0-9_\-]+)', raw, re.I)
                    enc = m.group(1).decode("ascii", "ignore").lower() if m else (getattr(resp, "apparent_encoding", None) or "utf-8")
                    return raw.decode(enc, errors="replace")
                except Exception as e_referer:
                    if i == max_retry - 1:
                        print(f"[ERROR] Failed to fetch {url} with modified referer: {e_referer}")
                        return None
            if i == max_retry - 1:
                print(f"[ERROR] Failed to fetch {url}: {e}")
                return None
            time.sleep(sleep_sec)

        except Exception as e:
            if i == max_retry - 1:
                print(f"[ERROR] Failed to fetch {url}: {e}")
                return None
            time.sleep(sleep_sec)
    return None


def cleanup_soup(soup: BeautifulSoup):
    """리뷰/이벤트/뉴스 등 노이즈 영역과 불필요 태그 제거."""
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    noisy = soup.select(
        '[class*="review"], [id*="review"], '
        '[class*="event"], [id*="event"], '
        '[class*="news"], [id*="news"], '
        '[class*="magazine"]'
    )
    for n in noisy:
        n.decompose()

def extract_prices_from_text(html: str) -> List[Dict]:
    """페이지 전체 텍스트에서 '줄 단위'로 가격 구문만 추출."""
    soup = BeautifulSoup(html, "html.parser")
    cleanup_soup(soup)
    text = soup.get_text("\n", strip=True)

    found: List[Dict] = []
    for line in text.splitlines():
        if "원" not in line:
            continue
        if not ("월" in line or "개월" in line):
            continue
        if len(line) > MAX_LINE_LEN:
            continue
        got = extract_price_phrase(line)
        if not got:
            continue
        term, price_txt, price_krw = got
        found.append({
            "plan_type": "",
            "term_months": term,
            "price_text": price_txt,   # 라인 전체가 아니라 '가격 구문만'
            "price_krw": price_krw,
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

# ----- 도메인 전용 파서 (수율↑ / 오탐↓) -----
def parse_spoany(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    cleanup_soup(soup)

    out: List[Dict] = []
    candidates = []
    candidates += soup.select('[class*="price"], [id*="price"]')
    candidates += soup.select('[class*="membership"], [id*="membership"]')
    candidates += soup.select('table, ul, ol')
    if not candidates:
        candidates = [soup]

    for node in candidates:
        txt = node.get_text("\n", strip=True)
        for line in txt.splitlines():
            if "원" not in line:
                continue
            if not ("월" in line or "개월" in line):
                continue
            if len(line) > MAX_LINE_LEN:
                continue
            got = extract_price_phrase(line)
            if not got:
                continue
            term, price_txt, price_krw = got
            out.append({
                "plan_type": "",
                "term_months": term,
                "price_text": price_txt,
                "price_krw": price_krw,
            })

    uniq = {(r["term_months"], r["price_text"]): r for r in out}
    return list(uniq.values())

def parse_modoo(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    cleanup_soup(soup)

    out: List[Dict] = []
    blocks = []
    rows = soup.select("table tr, .se-table tr, .se_sectionArea table tr")
    if rows:
        for tr in rows:
            txt = tr.get_text(" ", strip=True)
            blocks.append(txt)
    else:
        blocks = [b.get_text(" ", strip=True) for b in soup.find_all(["li", "p", "span", "div"])]

    for txt in blocks:
        if "원" not in txt:
            continue
        for line in txt.splitlines():
            if "원" not in line:
                continue
            if not ("월" in line or "개월" in line):
                continue
            if len(line) > MAX_LINE_LEN:
                continue
            got = extract_price_phrase(line)
            if not got:
                continue
            term, price_txt, price_krw = got
            out.append({
                "plan_type": "",
                "term_months": term,
                "price_text": price_txt,
                "price_krw": price_krw,
            })

    uniq = {(r["term_months"], r["price_text"]): r for r in out}
    return list(uniq.values())

DOMAIN_PARSERS = {
    "m.spoany.co.kr": parse_spoany,
    "spoany.co.kr": parse_spoany,
    "modoo.at": parse_modoo,
}

# ====== 가격 페이지 탐색 (BFS) ======
def _score_link(url_or_text: str) -> int:
    s = url_or_text.lower()
    return sum(1 for k in PRICE_URL_HINTS if k in s)

def _same_host(u1: str, u2: str) -> bool:
    return urlparse(u1).netloc == urlparse(u2).netloc

def _looks_blocked(u: str) -> bool:
    s = u.lower()
    return any(b in s for b in BLOCK_URL_PARTS)

_NUMBER_WON = re.compile(r'\d[\d,\.]*\s*원')
_NEAR_MONTH = re.compile(r'(?:\d{1,2}\s*개월)|(?:월[^가-힣0-9]*\d)', re.I)

def _pricey_score_from_text(text: str) -> int:
    score = 0
    if _NUMBER_WON.search(text):
        score += 3
    if _NEAR_MONTH.search(text):
        score += 2
    return score

def discover_price_page_bfs(seed_url: str, seed_html: str, max_depth: int = 2, max_saved: int = 20) -> Optional[str]:
    """
    seed_url에서 시작해 같은 도메인의 내부 링크를 BFS로 탐색.
    링크/URL 힌트 스코어 + 본문 가격 신호 점수(숫자+원, 개월/월)로 우선순위.
    로그인/약관/뉴스 등은 블록리스트로 제외.
    """
    visited = set([seed_url])
    q = deque([(seed_url, seed_html, 0)])
    best_candidate, best_score = None, -1
    saved_count = 0

    while q:
        cur_url, cur_html, depth = q.popleft()

        soup_cur = BeautifulSoup(cur_html, "html.parser")
        page_text = soup_cur.get_text(" ", strip=True)
        page_score = _score_link(cur_url) + _pricey_score_from_text(page_text)

        if page_score > best_score:
            best_score, best_candidate = page_score, cur_url

        if depth < max_depth:
            anchors = soup_cur.find_all("a", href=True)
            children = []
            for a in anchors:
                full = urljoin(cur_url, a.get("href") or "")
                if not _same_host(seed_url, full):
                    continue
                if full in visited:
                    continue
                if _looks_blocked(full):
                    continue
                text_score = _score_link(a.get_text(" ", strip=True))
                url_score = _score_link(full)
                total = text_score + url_score
                children.append((total, full))

            children.sort(key=lambda x: x[0], reverse=True)

            for score, child in children:
                if child in visited:
                    continue
                visited.add(child)
                sub_html = robust_get(child)
                if not sub_html:
                    continue
                if score <= 0 and len(sub_html) < 2000:
                    continue
                if saved_count < max_saved:
                    save_debug_html("SUBPAGE", child, sub_html)
                    saved_count += 1
                q.append((child, sub_html, depth + 1))

    if best_candidate and best_candidate != seed_url:
        print(f"[HINT] discovered price page (depth≤{max_depth}): {best_candidate}")
        return best_candidate
    return None

# ====== 가격 추출 엔트리(이제 URL도 같이 반환) ======
def extract_prices(html: str, page_url: str) -> Tuple[List[Dict], str]:
    """
    Returns:
        (prices, used_url)
    """
    host = urlparse(page_url).netloc.lower()

    # 1) 도메인 전용 파서
    for domain, fn in DOMAIN_PARSERS.items():
        if domain in host:
            data = fn(html)
            if data:
                return data, page_url

    # 2) 일반 텍스트 파서
    data = extract_prices_from_text(html)
    if data:
        return data, page_url

    # 2-추가) 내부 링크 BFS로 가격 페이지 추정 후 재시도
    price_page = discover_price_page_bfs(page_url, html, max_depth=2, max_saved=20)
    if price_page and price_page != page_url:
        sub_html = robust_get(price_page)
        if sub_html:
            # 도메인 파서 → 일반 파서 순으로 한번 더
            for domain, fn in DOMAIN_PARSERS.items():
                if domain in host:
                    d2 = fn(sub_html)
                    if d2:
                        return d2, price_page
            d2 = extract_prices_from_text(sub_html)
            if d2:
                return d2, price_page

    # 3) 이미지 OCR (옵션)
    img_data = extract_prices_from_images(html, page_url)
    return img_data, page_url

def choose_url(row: dict) -> Optional[str]:
    """price_page_url > website_url > (website_url 기반 price 후보 경로) 순. 지도/플랫폼은 스킵."""
    for key in ("price_page_url", "website_url"):
        url = (row.get(key) or "").strip()
        if url.startswith("http") and not looks_like_map(url):
            return url
    base = (row.get("website_url") or "").strip()
    if base.startswith("http") and not looks_like_map(base):
        for p in ["price", "prices", "membership", "멤버십", "요금", "이용요금", "program", "programs"]:
            cand = urljoin(base.rstrip("/") + "/", p)
            return cand
    return None

def main():
    if not os.path.exists(SEED_FILE):
        print(f"[ERROR] Seed file not found: {SEED_FILE}")
        return

    df = pd.read_csv(SEED_FILE, dtype=str).fillna("")
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
                "price_text": "NO_URL",
                "source_url": display_url,
            })
            continue

        print(f"[INFO] Fetching prices for: {name} ({crawl_url})")
        html = robust_get(crawl_url)
        if html is None:
            rows.append({
                "name": name,
                "address": row.get("address", ""),
                "phone": row.get("phone", ""),
                "plan_type": "",
                "term_months": "",
                "price_krw": "",
                "price_text": "FETCH_FAILED",
                "source_url": crawl_url,
            })
            continue

        # 디버그 저장 (메인 페이지)
        save_debug_html(name, crawl_url, html)

        prices, used_url = extract_prices(html, crawl_url)

        if not prices:
            reason = "NO_MATCH_ON_PAGE"
            rows.append({
                "name": name,
                "address": row.get("address", ""),
                "phone": row.get("phone", ""),
                "plan_type": "",
                "term_months": "",
                "price_krw": "",
                "price_text": reason,
                # 가격 페이지 못 찾았으면 메인 URL / 찾았는데도 없으면 그 페이지
                "source_url": used_url or crawl_url,
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
                    # 실제 가격을 뽑은 URL 기록
                    "source_url": used_url or crawl_url,
                })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

