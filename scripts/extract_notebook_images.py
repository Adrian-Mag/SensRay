#!/usr/bin/env python3
"""
Extract image/png outputs from Jupyter notebooks and save them as PNG files.

Usage:
    python scripts/extract_notebook_images.py [notebook_paths ...]

If no notebooks are given, the script will scan the repo `demos/` folder and
the `ray_tracing_tutorial.ipynb` file (if present).

This script only extracts existing PNG outputs embedded in the notebooks. To
re-generate images run the notebooks (e.g. with nbconvert) before extraction.
"""
import sys
import json
import base64
from pathlib import Path


def extract_images_from_notebook(nb_path: Path, out_dir: Path):
    with nb_path.open('r', encoding='utf-8') as fh:
        nb = json.load(fh)

    saved = []
    cells = nb.get('cells', [])
    for ci, cell in enumerate(cells, start=1):
        outputs = cell.get('outputs', []) if cell.get('outputs') else []
        for oi, out in enumerate(outputs, start=1):
            # Jupyter stores PNGs in data -> 'image/png'
            data = out.get('data') or {}
            img_b64 = data.get('image/png')
            if img_b64:
                # Some notebooks store as list of lines
                if isinstance(img_b64, list):
                    img_b64 = ''.join(img_b64)

                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{nb_path.stem}_cell{ci:03d}_out{oi:03d}.png"
                out_path = out_dir / fname
                with out_path.open('wb') as of:
                    of.write(base64.b64decode(img_b64))
                saved.append(out_path)

    return saved


def main(args):
    repo_root = Path(__file__).resolve().parents[1]
    if args:
        nb_paths = [Path(a) for a in args]
    else:
        # default locations to search
        nb_dir = repo_root / 'demos'
        candidates = []
        if nb_dir.exists():
            candidates.extend(sorted(nb_dir.glob('*.ipynb')))
        rt = repo_root / 'ray_tracing_tutorial.ipynb'
        if rt.exists():
            candidates.append(rt)
        nb_paths = candidates

    if not nb_paths:
        print('No notebooks found to scan.')
        return 1

    out_dir = repo_root / 'docs' / 'images'
    all_saved = []
    for nb in nb_paths:
        if not nb.exists():
            print(f'Skipping missing notebook: {nb}')
            continue
        saved = extract_images_from_notebook(nb, out_dir)
        for p in saved:
            print(f'Saved: {p.relative_to(repo_root)}')
        all_saved.extend(saved)

    if not all_saved:
        print(
            'No embedded PNG outputs found. To generate images run the '
            'notebooks first.'
        )
        return 2

    print(f'Extracted {len(all_saved)} images to {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
