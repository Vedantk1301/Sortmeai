from pathlib import Path
text = Path('page.tsx').read_text()
idx = text.find('<div className="product-price-row">')
print(text[idx:idx+800])
