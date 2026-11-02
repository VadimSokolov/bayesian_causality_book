"""Drop a plain index reference when the same key is marked bold on the same page.

makeindex prints a page twice when one key has a plain reference and a |textbf reference on that page
(the defining occurrence and a nearby mention). The build runs this filter on main.idx before makeindex:
for every key and page that carry a bold reference, the plain references to that key and page are removed.
Run from book_tex/: python3 ../tools/idx_dedupe.py main.idx
"""
import re
import sys

ENTRY = re.compile(r'^\\indexentry\{(.*)\}\{([^{}]*)\}\s*$')


def split_encap(arg):
    """Split the argument of \\indexentry into key and encap at the last top level |."""
    depth = 0
    cut = None
    for i, ch in enumerate(arg):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        elif ch == '|' and depth == 0:
            cut = i
    if cut is None:
        return arg, ''
    return arg[:cut], arg[cut + 1:]


def main(path):
    lines = open(path, encoding='utf-8', errors='surrogateescape').read().split('\n')
    parsed = []
    bold = set()
    for ln in lines:
        m = ENTRY.match(ln)
        if not m:
            parsed.append((ln, None, None, None))
            continue
        key, encap = split_encap(m.group(1))
        page = m.group(2)
        parsed.append((ln, key, encap, page))
        if 'textbf' in encap:
            bold.add((key, page))
    out, dropped = [], 0
    for ln, key, encap, page in parsed:
        if key is not None and (key, page) in bold and encap in ('', 'hyperpage'):
            dropped += 1
            continue
        out.append(ln)
    open(path, 'w', encoding='utf-8', errors='surrogateescape').write('\n'.join(out))
    print('idx_dedupe: dropped %d plain references that share a page with a bold one' % dropped)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'main.idx')
