from pathlib import Path
text = Path('page.tsx').read_text()
start = text.find('<div className="product-price-row">')
end = text.find('</div>', start)
# find next closing div line after original block maybe not enough, so extend to after two </div>
segment = text[start:start+200]
print(segment)
