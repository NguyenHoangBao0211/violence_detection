import ast,json
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[2]
n=json.loads((root/'notebooks/notebook7d69f2165d_no_rwf.ipynb').read_text(encoding='utf-8'))
for c in n['cells']:
    if c['cell_type']=='code': ast.parse(''.join(c['source']))
tree=ast.parse(''.join(n['cells'][3]['source']))
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='apply_reviewed_labels')
scope={'pd':pd}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'labels','exec'),scope)
df=pd.DataFrame([dict(source='mywebcam',path=s+str(y),split=s,label=y) for s in ['train','val','test'] for y in [0,1]])
review=df.copy(); review['reviewed_label']=1-review.label
p=root/'storage/label_review_test.csv'
try:
    review.to_csv(p,index=False)
    out=scope['apply_reviewed_labels'](df,p)
    assert (out.label==1-df.label).all()
    review.loc[0,'reviewed_label']=None; review.to_csv(p,index=False)
    try: scope['apply_reviewed_labels'](df,p)
    except ValueError: pass
    else: raise AssertionError('Incomplete labels accepted')
finally:
    p.unlink(missing_ok=True)
print('PASS: notebook syntax, corrected labels applied, incomplete review rejected')
