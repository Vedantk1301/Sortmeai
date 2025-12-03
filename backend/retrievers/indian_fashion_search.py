import os
import json
import concurrent.futures
from functools import partial
import re
import ast
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI, BadRequestError

from config import Config

OPENAI_API_KEY = Config.OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# Use gpt-5-nano with Responses API + web_search tool
MODEL_NAME = "gpt-5-nano"

# ===================== URL heuristics (India e-com + PDP vs listing) =====================

# Known Indian marketplaces + popular Indian / India-focused brands
INDIA_DOMAINS = {
    # Marketplaces / multi-brand
    "amazon.in",
    "www.amazon.in",
    "flipkart.com",
    "www.flipkart.com",
    "myntra.com",
    "www.myntra.com",
    "ajio.com",
    "www.ajio.com",
    "nykaafashion.com",
    "www.nykaafashion.com",
    "tatacliq.com",
    "www.tatacliq.com",
    "meesho.com",
    "www.meesho.com",

    # Fast-fashion / western-leaning, India-targeted
    "urbanic.com",
    "www.urbanic.com",
    "in.urbanic.com",
    "urbanicindia.com",
    "www.urbanicindia.com",
    "urbanicindia.in",
    "www.urbanicindia.in",
    "savana.com",
    "www.savana.com",
    "newme.asia",
    "www.newme.asia",
    "newmefashion.com",
    "www.newmefashion.com",
    "littleboxindia.com",
    "www.littleboxindia.com",

    # Ethnic / fusion Indian brands
    "libas.in",
    "www.libas.in",
    "shoplibas.com",
    "www.shoplibas.com",
    "biba.in",
    "www.biba.in",
    "globaldesi.in",
    "www.globaldesi.in",
    "fabindia.com",
    "www.fabindia.com",
    "wforwoman.com",
    "www.wforwoman.com",
    "houseofindya.com",
    "www.houseofindya.com",
}


def is_india_ecom_url(url: str, price_hint: Optional[Any] = None) -> bool:
    """
    Heuristic: check if URL belongs to a known Indian e-commerce / fashion domain
    OR looks India-focused (e.g. .in / .co.in / in.<brand>.com).

    price_hint is an optional numeric INR-looking price from the LLM output; if present
    in a reasonable INR range, we treat it as an extra (weak) signal that this is
    India-oriented.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
    except Exception:
        return False

    # 1) Explicit allow-list of popular Indian marketplaces/brands
    if netloc in INDIA_DOMAINS:
        return True

    # 2) Generic .in / .co.in domains (usually India-focused)
    if netloc.endswith(".in") or netloc.endswith(".co.in"):
        return True

    # 3) Subdomain-based India sites, e.g. in.urbanic.com
    if netloc.startswith("in.") and netloc.endswith(".com"):
        return True

    # 4) Optional numeric INR price hint (very loose sanity bounds)
    if isinstance(price_hint, (int, float)):
        if 50 <= price_hint <= 200_000:
            return True

    return False


def looks_like_product_page(url: str) -> bool:
    """
    Return True only if the URL looks like a single product detail page (PDP)
    on a known / likely fashion e-commerce site.

    We do NOT treat search / listing / category URLs as product pages.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.lower()
        query = (parsed.query or "").lower()
    except Exception:
        return False

    # ---------- Amazon.in ----------
    if "amazon.in" in netloc:
        # Listings/search pages use /s, product pages use /dp/ or /gp/product
        if path.startswith("/s") or path.endswith("/s") or "/s/" in path:
            return False
        return ("/dp/" in path) or ("/gp/product" in path)

    # ---------- Flipkart ----------
    if "flipkart.com" in netloc:
        # Search/listing URLs
        if "/search" in path or "q=" in query:
            return False
        # Product pages
        return ("/p/" in path) or ("pid=" in query)

    # ---------- Myntra ----------
    if "myntra.com" in netloc:
        # Product pages usually have /buy/
        if "/buy/" in path:
            return True
        # Treat category URLs like /women-white-trousers as listing → not a single PDP
        return False

    # ---------- Ajio ----------
    if "ajio.com" in netloc:
        # PDPs generally have /p/
        return "/p/" in path

    # ---------- Nykaa Fashion ----------
    if "nykaafashion.com" in netloc:
        # PDPs usually have /p/ (and an ID/slug)
        return "/p/" in path

    # ---------- TataCliq ----------
    if "tatacliq.com" in netloc:
        # PDP patterns
        return ("/p-" in path) or ("/product/" in path)

    # ---------- Meesho ----------
    if "meesho.com" in netloc:
        # PDPs use /p/<id>
        return "/p/" in path

    # ---------- Urbanic ----------
    if "urbanic" in netloc:
        if "/product/" in path or "/products/" in path:
            return True

    # ---------- NEWME ----------
    if "newme.asia" in netloc or "newmefashion.com" in netloc:
        if "/product/" in path or "/products/" in path:
            return True

    # ---------- Littlebox ----------
    if "littleboxindia.com" in netloc:
        if "/products/" in path:
            return True

    # ---------- Libas / Biba / Global Desi / W / Fabindia / Indya ----------
    if any(
        brand in netloc
        for brand in (
            "libas.in",
            "shoplibas.com",
            "biba.in",
            "globaldesi.in",
            "wforwoman.com",
            "fabindia.com",
            "houseofindya.com",
        )
    ):
        if "/product/" in path or "/products/" in path:
            return True

    # ---------- Generic PDP heuristic ----------
    if "/product/" in path or "/products/" in path:
        return True

    # Any other domain: we don't know the PDP pattern → be safe and say False
    return False


def looks_like_listing_page(url: str) -> bool:
    """
    Heuristic: True if this looks like a *search/listing* page (we only handle Amazon for now),
    which we can scrape to get multiple PDP links.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.lower()
        query = (parsed.query or "").lower()
    except Exception:
        return False

    # Amazon search/listing:
    # - classic: /s?k=...
    # - SEO slug: /something/s?k=...
    if "amazon.in" in netloc:
        if (
            path == "/s"
            or path.startswith("/s/")
            or path.endswith("/s")
            or "/s/" in path
            or "k=" in query
        ):
            return True

    # Extend here later for Myntra/Ajio/etc if you want.
    return False


PRICE_RE = re.compile(r"[₹\s]*([\d,]{2,})")


def extract_inr_price(text: str) -> Optional[int]:
    """
    Extract an INR-like number from text (e.g. '₹1,999' -> 1999).
    Very simple heuristic – used only as a hint when scraping.
    """
    if not text:
        return None
    m = PRICE_RE.search(text)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return int(num)
    except ValueError:
        return None


# ===================== Scraping Amazon listing pages into PDP URLs =====================

def scrape_amazon_listing_to_products(
    url: str, max_products: int = 5, timeout: int = 8
) -> List[Dict[str, Any]]:
    """
    Given an Amazon search/listing URL, scrape it and extract up to max_products
    product detail URLs (PDPs). Returns product dicts in the same shape as LLM
    output (but without fancy description / tone).
    """
    # print(f"[DEBUG] Scraping Amazon listing: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        # print(f"[DEBUG] Failed to fetch listing page: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, Any]] = []

    # Amazon search result items: div.s-result-item[data-asin]
    for div in soup.select("div.s-result-item[data-asin]"):
        if len(results) >= max_products:
            break

        asin = div.get("data-asin")
        if not asin:
            continue

        # Product link inside this result item
        a_tag = (
            div.select_one("a.a-link-normal.s-no-outline")
            or div.select_one(
                "a.a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal"
            )
        )
        if not a_tag:
            continue

        href = a_tag.get("href")
        if not href:
            continue

        full_url = urljoin("https://www.amazon.in", href)

        # Only keep if it's clearly a product page
        if not looks_like_product_page(full_url):
            continue

        # Extract title & price
        title_el = (
            div.select_one("span.a-size-medium.a-color-base.a-text-normal")
            or div.select_one("span.a-text-normal")
        )
        title = title_el.get_text(strip=True) if title_el else "Amazon product"

        price_el = div.select_one("span.a-price span.a-price-whole")
        price_text = price_el.get_text(strip=True) if price_el else ""
        price = extract_inr_price(price_text)

        # Build simple product dict (description = title for now)
        results.append(
            {
                "name": title,
                "description": title,
                "price": price,
                "imageUrl": None,
                "sourceUrl": full_url,
                "tone": "casual",
            }
        )

    # print(f"[DEBUG] Extracted {len(results)} PDP URLs from Amazon listing")
    return results


# ===================== LLM + web_search part =====================

def search_fashion_with_web(
    user_query: str,
    max_results: int = 5,
    image_max_workers: int = 10,
    image_timeout: int = 6,
) -> List[Dict[str, Any]]:
    """
    Use the Responses API with the built-in web_search tool to find fashion products.
    Then post-process:
      - Keep only India / India-focused e-com domains.
      - Keep only URLs that look like PDPs.
      - If some URLs are listing/search pages (especially Amazon), scrape them
        into multiple PDP URLs.
    Returns a list of product dicts with keys:
      name, description, price, imageUrl, sourceUrl, tone
    """

    system_prompt = """
You are a fashion shopping assistant for Indian users.

You can use the `web_search` tool to find real products on live e-commerce websites.

The user will give a fashion shopping query (e.g., "white wide leg pants under 2000 for women").

GOALS:
- Find **real, purchasable products** that match the user's constraints:
  - product type/category (e.g., wide leg pants),
  - gender (e.g., women),
  - color (e.g., white),
  - and budget (e.g., under 2000, below 1500, etc.).
- Focus ONLY on Indian / India-focused e-commerce sites:
  - Marketplaces: Amazon.in, Flipkart, Myntra, Ajio, Nykaa Fashion, TataCliq, Meesho, etc.
  - Fast-fashion / western: Urbanic, Savana, NEWME, Littlebox India, etc.
  - Ethnic / fusion: Libas, Biba, W for Woman, Fabindia, Global Desi, House of Indya, etc.
- Prefer pages showing prices in Indian Rupees (INR).
- When the query is general (not strictly ethnic or western), mix both Indian-ethnic
  and western silhouettes/brands in the results if possible.

TOOL USAGE:
- You MAY call `web_search` multiple times if helpful, but keep calls reasonable.
- When you construct queries:
  - Include the user's key constraints (color, type, gender, budget).
  - Bias towards domains like:
    amazon.in, flipkart.com, myntra.com, ajio.com,
    nykaafashion.com, tatacliq.com, meesho.com,
    urbanic.com, in.urbanic.com, savana.com, newme.asia, littleboxindia.com,
    libas.in, biba.in, globaldesi.in, fabindia.com, wforwoman.com, houseofindya.com.
  - You may also use other `.in` or `.co.in` fashion sites that clearly show INR prices.
  - Always prefer individual product detail pages (PDPs) rather than generic listings.

URL RULES (VERY IMPORTANT):
- You should **strongly prefer** URLs that look like a single product detail page:
  - Amazon: URLs containing "/dp/" or "/gp/product".
  - Flipkart: URLs containing "/p/" or "pid=".
  - Myntra: URLs containing "/buy/".
  - Ajio: URLs containing "/p/".
  - Nykaa Fashion: URLs containing "/p/".
  - TataCliq: URLs containing "/p-" or "/product/".
  - Meesho: URLs containing "/p/" with a product ID.
  - Urbanic / NEWME / Littlebox / Libas / Biba / Global Desi / W / Fabindia / Indya:
    URLs containing "/product/" or "/products/".
- Avoid generic search or category URLs like:
  - Amazon search: URLs with "/s" and "k=".
  - Flipkart search: URLs with "/search".
  - Myntra category URLs that do not have "/buy/".
- However, if you absolutely cannot find enough PDP URLs, you may return some listing URLs
  as a fallback. The client will scrape those later.

PRICE & INR:
- Only select products that appear to have a price in INR.
- When you set the `price` field, use the numeric INR value (no currency symbol).
- If you can't confidently find an INR price, set `price` to null.

MATCHING QUALITY:
- Treat color and fit (e.g., "white", "wide leg") as strong filters.
- Prefer exact matches to the user's text.
- If exact matches are too few, you may include close variants:
  - e.g., off-white/ivory for white; clearly wide-leg palazzos for wide leg.
- If a budget is mentioned (e.g., "under 2000"), prefer items whose price is <= that.
- If you still don't have enough items, you may include some slightly above the budget,
  but only if they strongly match other constraints.
- Try NOT to over-index on a single brand or site when multiple good options exist.

STRICT OUTPUT FORMAT:
- You MUST respond with ONLY a single JSON object.
- No explanations, no markdown, no backticks, no commentary outside the JSON.
- Inside JSON strings, do NOT use raw double quotes; if needed, escape them as \".
- The JSON MUST have exactly this shape:

{
  "products": [
    {
      "name": string,
      "description": string,
      "price": number | null,
      "imageUrl": string | null,
      "sourceUrl": string,
      "tone": string | null
    }
  ]
}

- `name`: cleaned-up product name.
- `description`: 1–2 sentences describing the product (color, fit, occasion, etc.).
- `price`: integer INR (no symbol) or null.
- `imageUrl`: MUST be null (the client will scrape product pages for images).
- `sourceUrl`: the product URL (ideally a PDP on an Indian e-commerce site).
- `tone`: simple style label like "casual", "festive", "ethnic", or null.

Return AT MOST max_products items. If fewer truly match, return fewer.
"""

    user_payload = {
        "user_query": user_query,
        "max_products": max_results,
    }

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "low",
                }
            ],
            tool_choice="auto",
        )
    except BadRequestError as e:
        print("OpenAI BadRequestError:", e)
        raise

    # Aggregate all assistant text into a single string
    reply_text = (response.output_text or "").strip()

    # Parse JSON from assistant text robustly
    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        # Try to slice between outermost braces
        start = reply_text.find("{")
        end = reply_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = reply_text[start : end + 1]
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # As a last resort, try Python literal eval (handles single quotes, etc.)
                try:
                    parsed = ast.literal_eval(json_str)
                except Exception:
                    # If all fails, return empty
                    return []
        else:
            return []

    products = parsed.get("products", [])
    if not isinstance(products, list):
        return []

    # Normalize initial output
    normalized: List[Dict[str, Any]] = []
    for p in products:
        normalized.append(
            {
                "name": (p.get("name") or "").strip(),
                "description": (p.get("description") or "").strip(),
                "price": p.get("price"),
                "imageUrl": p.get("imageUrl"),
                "sourceUrl": (p.get("sourceUrl") or "").strip(),
                "tone": p.get("tone"),
            }
        )

    # ===================== Post-processing: enforce India PDP + scrape listings =====================

    processed: List[Dict[str, Any]] = []
    scraped_listing_count = 0
    MAX_LISTINGS_TO_SCRAPE = 2  # safety limit so it doesn't get too slow

    for p in normalized:
        if len(processed) >= max_results:
            break

        url = p["sourceUrl"]
        if not url:
            continue

        price_hint = p.get("price")

        # Must be on Indian / India-focused e-com domain
        if not is_india_ecom_url(url, price_hint=price_hint):
            continue

        # If already looks like a product page, keep as is
        if looks_like_product_page(url):
            processed.append(p)
            continue

        # If looks like listing (Amazon search), scrape it into PDPs
        if looks_like_listing_page(url) and scraped_listing_count < MAX_LISTINGS_TO_SCRAPE:
            scraped_listing_count += 1
            scraped_products = scrape_amazon_listing_to_products(
                url,
                max_products=max_results - len(processed),
            )
            # Extend processed with scraped PDPs
            for sp in scraped_products:
                if len(processed) >= max_results:
                    break
                if is_india_ecom_url(sp["sourceUrl"], price_hint=sp.get("price")) and looks_like_product_page(
                    sp["sourceUrl"]
                ):
                    processed.append(sp)

    # Fallback: if scraping/filters removed everything, fall back to normalized
    if not processed:
        processed = normalized[:max_results]

    # Finally, cap to max_results
    processed = processed[:max_results]
    
    # Enrich with images
    processed = enrich_products_with_images(
        processed,
        max_workers=image_max_workers,
        fetch_timeout=image_timeout,
    )
    
    return processed


# ===================== Scraping PDPs for image URLs =====================

def _extract_image_url_from_html(html: str, base_url: str) -> Optional[str]:
    """
    Given HTML for a product page, try to extract a good product image URL.
    Priority:
      1. <meta property="og:image"> / twitter:image
      2. <link rel="image_src">
      3. A large-ish <img> that doesn't look like a logo / sprite
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) OpenGraph / Twitter image
    for prop in ["og:image", "twitter:image", "twitter:image:src"]:
        tag = soup.find("meta", property=prop) or soup.find(
            "meta", attrs={"name": prop}
        )
        if tag and tag.get("content"):
            return urljoin(base_url, tag["content"].strip())

    # 2) link rel="image_src"
    link_tag = soup.find("link", rel="image_src")
    if link_tag and link_tag.get("href"):
        return urljoin(base_url, link_tag["href"].strip())

    # 3) Fallback: pick a likely main <img>
    candidates = []
    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-srcset")
        )
        if not src:
            continue

        # If srcset-style, take the first URL
        src = str(src).split()[0]

        lower_src = src.lower()
        # Skip obvious non-product images
        if any(bad in lower_src for bad in ["sprite", "logo", "icon", "placeholder"]):
            continue

        # Try get size info (helps choose largest image)
        try:
            width = int(img.get("width") or 0)
            height = int(img.get("height") or 0)
        except ValueError:
            width = height = 0

        area = width * height
        if area == 0:
            # some sites don't set width/height; still consider them but at low priority
            area = 1

        candidates.append((area, src))

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        best_src = candidates[0][1]
        return urljoin(base_url, best_src)

    return None


BLOCKED_IMAGE_FETCH_DOMAINS = {
    # These domains tend to block HTML fetches from scripts; go straight to image search
    "nykaafashion.com",
    "ajio.com",
    "meesho.com",
}


def _fetch_image_for_product(
    product: Dict[str, Any],
    timeout: int = 6,
    blocked_domains: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    For a single product dict, try to fill imageUrl by scraping sourceUrl.
    If anything fails, we leave imageUrl as-is.
    """
    url = product.get("sourceUrl")
    if not url:
        return product

    # Don't re-scrape if we already have an imageUrl
    if product.get("imageUrl"):
        return product

    # Decide if we should skip direct PDP fetch for known-blocking domains
    parsed_domain = ""
    try:
        parsed_domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        parsed_domain = ""

    should_skip_direct_fetch = False
    if blocked_domains:
        for d in blocked_domains:
            if parsed_domain.endswith(d):
                should_skip_direct_fetch = True
                break

    # Try fetching the PDP directly first (if not blocked)
    if not should_skip_direct_fetch:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                ),
                # Some CDNs get picky about missing headers even if we aren't rendering JS
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()

            img_url = _extract_image_url_from_html(resp.text, url)
            if img_url:
                product["imageUrl"] = img_url
                return product
        except Exception:
            # It's fine if we can't get an image from the PDP; try a fallback search.
            pass

    # Fallback: image search by product name (helpful when PDPs block scraping)
    fallback_query = (product.get("name") or product.get("description") or "").strip()
    if not fallback_query:
        return product

    # Include the domain in the query to bias towards the right product imagery
    try:
        parsed = urlparse(url)
        domain_hint = parsed.netloc.replace("www.", "")
    except Exception:
        domain_hint = ""

    if domain_hint:
        fallback_query = f"{fallback_query} {domain_hint}"

    try:
        from duckduckgo_search import DDGS  # type: ignore

        # Keep timeout tight; we only need one good image URL
        with DDGS(timeout=6) as ddgs:
            for result in ddgs.images(fallback_query, max_results=3):
                img_url = result.get("image") or result.get("thumbnail")
                if img_url:
                    product["imageUrl"] = img_url
                    break
    except Exception:
        # If the fallback search fails (no internet, dependency missing, etc.), leave imageUrl unset.
        pass

    return product


def enrich_products_with_images(
    products: List[Dict[str, Any]],
    max_workers: int = 10,
    fetch_timeout: int = 6,
    blocked_domains: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run image scraping in parallel for speed.
    Only touches imageUrl field; everything else stays the same.
    """
    if not products:
        return products

    worker = partial(
        _fetch_image_for_product,
        timeout=fetch_timeout,
        blocked_domains=blocked_domains or BLOCKED_IMAGE_FETCH_DOMAINS,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        enriched = list(executor.map(worker, products))

    return enriched
