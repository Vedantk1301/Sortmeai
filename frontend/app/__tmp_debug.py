from pathlib import Path
text = Path('page.tsx').read_text()
idx = text.find('product-price-row')
print('idx', idx)
print(text[idx-20:idx+120])
