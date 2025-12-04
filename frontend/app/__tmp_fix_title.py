from pathlib import Path
lines = Path('page.tsx').read_text().splitlines()
insert_at = 501  # zero-based index? lines list length? we want after line 500 (1-based), index 500
lines.insert(500, '                                      </div>')
Path('page.tsx').write_text('\n'.join(lines) + '\n')
print('inserted closing title div')
