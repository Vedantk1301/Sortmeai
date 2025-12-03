import requests
from urllib.parse import urlparse
import sys


def product_url_to_product_js(url: str) -> str:
    """
    Convert a product URL like:
      https://offduty.in/products/desert-blue-dust-straight-leg-jeans
    to:
      https://offduty.in/products/desert-blue-dust-straight-leg-jeans.js
    (Shopify Ajax Product API)
    """
    parsed = urlparse(url)
    path = parsed.path

    if "/products/" not in path:
        raise ValueError(f"Not a product URL: {url}")

    if path.endswith("/"):
        path = path[:-1]

    if not path.endswith(".js"):
        path = path + ".js"

    return f"{parsed.scheme}://{parsed.netloc}{path}"


def fetch_product_js(product_js_url: str) -> dict:
    resp = requests.get(product_js_url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def print_product_with_discounts(product: dict):
    print("\n=== {} ===".format(product.get("title", "Unknown product")))
    print("Handle:", product.get("handle"))
    print("URL   :", f"https://offduty.in/products/{product.get('handle')}")

    variants = product.get("variants", [])
    if not variants:
        print("No variants found.")
        return

    print(f"\n{'Variant':25} {'MRP':>10} {'Price':>10} {'Disc%':>7} {'In Stock':>10}")
    print("-" * 70)

    for v in variants:
        title = v.get("title", "Variant")
        price = v.get("price")              # current price, in paise
        cap = v.get("compare_at_price")     # original price, in paise or None
        available = v.get("available", False)

        # Shopify prices in js are usually in subunits (e.g. paise), so /100
        def money(p):
            if p is None:
                return "-"
            return f"₹{p/100:.2f}"

        # Discount %
        if cap and cap > price:
            disc_pct = round((cap - price) * 100 / cap, 1)
            disc_str = f"{disc_pct}%"
        else:
            disc_str = "-"

        print(f"{title:25} {money(cap):>10} {money(price):>10} {disc_str:>7} {str(available):>10}")


def test_products(product_urls):
    for url in product_urls:
        try:
            print(f"\nChecking: {url}")
            js_url = product_url_to_product_js(url)
            product = fetch_product_js(js_url)
            print_product_with_discounts(product)
        except Exception as e:
            print(f"Error for {url}: {e}")


if __name__ == "__main__":
    default_urls = [
        "https://offduty.in/products/desert-blue-dust-straight-leg-jeans",
        "https://offduty.in/products/90s-blue-ripped-baggy-fit-jeans",
    ]

    urls = sys.argv[1:] if len(sys.argv) > 1 else default_urls
    test_products(urls)

