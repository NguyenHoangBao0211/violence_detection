import ast, csv, hashlib, json
from pathlib import Path
import cv2
root = Path(__file__).resolve().parents[2]
items = json.loads((root/'reports/label_review/webcam_inventory.json').read_text(encoding='utf-8'))
rows=[]
for item in items:
    p=Path(item['path']); cap=cv2.VideoCapture(str(p))
    fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok,frame=cap.read()
    cap.release()
    label=0 if p.parent.name=='non_violence' else 1
    rows.append(dict(path=str(p),label=label,class_name=p.parent.name,fps=fps,frames=count,duration=count/fps if fps>0 else 0,decode_ok=bool(ok),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),reviewed_label='',review_notes=''))
with (root/'reports/label_review/webcam_label_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
n=json.loads((root/'notebooks/notebook7d69f2165d_no_rwf.ipynb').read_text(encoding='utf-8'))
for c in n['cells']:
    if c['cell_type']=='code': ast.parse(''.join(c['source']))
summary={'clips':len(rows),'decode_failures':sum(not r['decode_ok'] for r in rows),'shorter_than_16_frames':sum(r['frames']<16 for r in rows),'duplicate_files':len(rows)-len(set(r['sha256'] for r in rows)),'content_labels_verified':False}
(root/'reports/label_review/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(summary)
