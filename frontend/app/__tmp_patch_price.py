from pathlib import Path
lines = Path('page.tsx').read_text().splitlines()
start = 501-1  # 1-based to 0-based
end = 515      # slice end index (1-based inclusive -> exclusive = 515)
new_lines = [
'                                      <div className="product-price-row">',
'                                        {(() => {',
'                                          const priceMeta = formatPrice(product.price);',
'                                          return (',
'                                            <>',
'                                              {priceMeta.display && (',
'                                                <div className="product-price">{priceMeta.display}</div>',
'                                              )}',
'                                              {priceMeta.compareAt && (',
'                                                <div className="product-price-original">{priceMeta.compareAt}</div>',
'                                              )}',
'                                            </>',
'                                          );',
'                                        })()}',
'                                      </div>',
]
lines[start:end] = new_lines
Path('page.tsx').write_text('\n'.join(lines) + '\n')
print('patched lines', start+1, 'to', end)
