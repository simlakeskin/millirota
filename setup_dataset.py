import zipfile, json, os, random
from pathlib import Path
from collections import defaultdict

random.seed(42)
BASE = Path('/Users/balim/Desktop/millirota_dataset')

for d in ['train/images','train/labels','val/images','val/labels']:
    (BASE/d).mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile('/Users/balim/Desktop/10/329_Hybrid_Detection_Project.coco.zip') as z:
    zip_map = {Path(n).name: n for n in z.namelist() if n.lower().endswith('.jpg')}
    with z.open('train/_annotations.coco.json') as f:
        coco = json.load(f)

    id2info = {img['id']: img for img in coco['images']}
    id2anns = defaultdict(list)
    for ann in coco['annotations']:
        id2anns[ann['image_id']].append(ann)
    cat_map = {c['id']: c['id']-1 for c in coco['categories'] if c['id'] > 0}

    all_ids = list(id2info.keys())
    random.shuffle(all_ids)
    train_ids = set(all_ids[:int(len(all_ids)*0.8)])

    converted = 0
    for img_id, img_info in id2info.items():
        fname = img_info['file_name']
        zip_path = zip_map.get(fname)
        if not zip_path:
            continue
        w, h = img_info['width'], img_info['height']
        split = 'train' if img_id in train_ids else 'val'

        with open(BASE/split/'images'/fname, 'wb') as f:
            f.write(z.read(zip_path))

        lines = []
        for ann in id2anns[img_id]:
            cid = ann['category_id']
            if cid not in cat_map:
                continue
            x, y, bw, bh = [float(v) for v in ann['bbox']]
            cx = max(0, min(1, (x+bw/2)/w))
            cy = max(0, min(1, (y+bh/2)/h))
            nw = max(0, min(1, bw/w))
            nh = max(0, min(1, bh/h))
            lines.append(f'{cat_map[cid]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        with open(BASE/split/'labels'/fname.replace('.jpg','.txt'), 'w') as f:
            f.write('\n'.join(lines))
        converted += 1

print(f'Tamamlandi: {converted} goruntu')
print(f'Train: {len(list((BASE/"train"/"images").glob("*.jpg")))}')
print(f'Val  : {len(list((BASE/"val"/"images").glob("*.jpg")))}')
