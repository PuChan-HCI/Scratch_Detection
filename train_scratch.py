"""Reproducible native-resolution training; temporal holdouts and guard frames."""
import os, json, random, time, argparse
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from scratch_model import ScratchNet

ROOT=Path(__file__).resolve().parent
def load_data(base):
    d=json.load(open(base/'annotations/instances_train.json'))
    anns={x['id']:[] for x in d['images']}
    for a in d['annotations']: anns[a['image_id']].append(a)
    data=[]
    for im in sorted(d['images'],key=lambda x:x['file_name']):
        rgb=cv2.cvtColor(cv2.imread(str(base/'images'/im['file_name'])),cv2.COLOR_BGR2RGB)
        mask=np.zeros(rgb.shape[:2],np.uint8); ignore=mask.copy(); objects=[]
        for a in anns[im['id']]:
            m=np.zeros_like(mask)
            cv2.fillPoly(m,[np.rint(np.array(p).reshape(-1,2)).astype(np.int32) for p in a['segmentation']],1)
            if a['category_id']==1: mask|=m; objects.append(a)
            else: ignore|=m
        ignore[mask>0]=0
        data.append(dict(info=im,rgb=rgb,mask=mask,ignore=ignore,objects=objects))
    return data

def split_data(data):
    ts=[int(x['info']['file_name'].split('_')[-1].split('.')[0]) for x in data]
    boundaries=[0]+[i for i in range(1,len(ts)) if ts[i]-ts[i-1]>2e7]+[len(ts)]
    splits={k:[] for k in ['train','validation','test','guard']}
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        n=b-a; v=max(5,int(n*.15)); t=b-v; q=t-3-v
        splits['train']+=list(range(a,q-3)); splits['guard']+=list(range(q-3,q))+list(range(t-3,t))
        splits['validation']+=list(range(q,t-3)); splits['test']+=list(range(t,b))
    return splits

def batch(data, ids, rng, n=4, size=512):
    xs=[]; ys=[]; vs=[]
    pos=[i for i in ids if data[i]['objects']]
    for _ in range(n):
        focus=rng.random()<.65; d=data[rng.choice(pos if focus else ids)]; h,w=d['mask'].shape
        if focus:
            a=rng.choice(d['objects']); x,y,bw,bh=a['bbox']; cx=x+rng.random()*bw; cy=y+rng.random()*bh
            x=int(np.clip(cx-rng.uniform(.2,.8)*size,0,w-size)); y=int(np.clip(cy-rng.uniform(.2,.8)*size,0,h-size))
        else: x=rng.randint(0,w-size); y=rng.randint(0,h-size)
        im=d['rgb'][y:y+size,x:x+size].copy(); m=d['mask'][y:y+size,x:x+size].copy(); v=1-d['ignore'][y:y+size,x:x+size].copy()
        k=rng.randrange(4); im=np.rot90(im,k); m=np.rot90(m,k); v=np.rot90(v,k)
        if rng.random()<.5: im=im[:,::-1]; m=m[:,::-1]; v=v[:,::-1]
        im=np.clip(im.astype(np.float32)/255*rng.uniform(.8,1.2)+rng.uniform(-.04,.04),0,1)
        xs.append(im.transpose(2,0,1).copy()); ys.append(m[None].copy()); vs.append(v[None].copy())
    return [torch.from_numpy(np.stack(a)).float().cuda() for a in [xs,ys,vs]]

@torch.no_grad()
def predict(model,rgb,size=512,stride=384):
    h,w=rgb.shape[:2]; out=np.zeros((h,w),np.float32); count=np.zeros_like(out)
    yy=list(range(0,max(h-size,0)+1,stride)); xx=list(range(0,max(w-size,0)+1,stride))
    if yy[-1]!=h-size: yy.append(h-size)
    if xx[-1]!=w-size: xx.append(w-size)
    coords=[(y,x) for y in yy for x in xx]
    for k in range(0,len(coords),8):
        loc=coords[k:k+8]; x=np.stack([rgb[y:y+size,x:x+size].transpose(2,0,1) for y,x in loc]).astype(np.float32)/255
        with torch.autocast('cuda',dtype=torch.bfloat16): p=model(torch.from_numpy(x).cuda()).sigmoid().float().cpu().numpy()[:,0]
        for (y,x),z in zip(loc,p): out[y:y+size,x:x+size]+=z; count[y:y+size,x:x+size]+=1
    return out/count

def measure(predictions,data,ids,threshold):
    tp=fp=fn=0; groups={k:[0,0] for k in ['all','small','thin']}; pixel_groups={k:[0,0] for k in groups}
    for p,i in zip(predictions,ids):
        d=data[i]; m=d['mask']>0; v=d['ignore']==0; b=p>=threshold
        tp+=int((b&m&v).sum()); fp+=int((b&~m&v).sum()); fn+=int((~b&m&v).sum())
        for a in d['objects']:
            z=np.zeros(m.shape,np.uint8); cv2.fillPoly(z,[np.rint(np.array(s).reshape(-1,2)).astype(np.int32) for s in a['segmentation']],1)
            # Thin: effective width 2*area/perimeter <= 6 native pixels.
            contours,_=cv2.findContours(z,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            perimeter=sum(cv2.arcLength(c,True) for c in contours); area=int(z.sum())
            keys=['all']+(['small'] if area<=1024 else [])+(['thin'] if 2*area/max(perimeter,1)<=6 else [])
            hit=int((b&(z>0)).sum()/max(area,1)>=.25)
            for key in keys:
                groups[key][0]+=hit; groups[key][1]+=1
                pixel_groups[key][0]+=int((b&(z>0)).sum()); pixel_groups[key][1]+=area
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    return dict(precision=precision,recall=recall,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),fp_pixels=fp,object_recall={k:dict(hits=a,total=b,recall=a/max(b,1)) for k,(a,b) in groups.items()},annotated_pixel_recall={k:a/max(b,1) for k,(a,b) in pixel_groups.items()})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path(r'D:\InnCoreTech\Scratch_Detection\coco_instance_dataset')); p.add_argument('--epochs',type=int,default=12); p.add_argument('--steps',type=int,default=80); p.add_argument('--resume',action='store_true'); p.add_argument('--lr',type=float,default=2e-4); args=p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.backends.cudnn.benchmark=True
    data=load_data(args.data); splits=split_data(data)
    json.dump({k:[data[i]['info']['file_name'] for i in v] for k,v in splits.items()},open(ROOT/'split_manifest.json','w'),indent=2)
    print('SPLITS', {k:len(v) for k,v in splits.items()},flush=True)
    model=ScratchNet(str(Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); rng=random.Random(142 if args.resume else 42); best=-1; history=[]; offset=0
    if args.resume:
        ck=torch.load(ROOT/'scratch_detail_best.pt',weights_only=True,map_location='cpu'); model.load_state_dict(ck['state_dict'])
        history=json.load(open(ROOT/'training_history.json')); offset=max(x['epoch'] for x in history)
    start=time.time()
    for epoch in range(args.epochs):
        model.train(); losses=[]
        # Keep pretrained BatchNorm statistics stable for small patch batches.
        for m in model.modules():
            if isinstance(m,torch.nn.BatchNorm2d): m.eval()
        for step in range(args.steps):
            x,y,v=batch(data,splits['train'],rng); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x)
            logits=logits.float(); prob=logits.sigmoid(); bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=torch.tensor(3.,device='cuda'))
            loss=(bce*v).sum()/v.sum().clamp_min(1)+1-(2*(prob*y*v).sum()+1)/((prob*v).sum()+(y*v).sum()+1)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; training stopped')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(loss.item())
        model.eval(); preds=[predict(model,data[i]['rgb']) for i in splits['validation']]
        options=[(t,measure(preds,data,splits['validation'],t)) for t in [.2,.3,.4,.5,.6,.7,.8,.9]]
        threshold,metrics=max(options,key=lambda a:a[1]['dice']); score=metrics['dice']
        row=dict(epoch=epoch+1+offset,loss=float(np.mean(losses)),threshold=threshold,validation=metrics,elapsed_seconds=round(time.time()-start,1),learning_rate=args.lr); history.append(row)
        print(json.dumps(row),flush=True); json.dump(history,open(ROOT/'training_history.json','w'),indent=2)
        if score>best:
            best=score; torch.save(dict(state_dict=model.cpu().state_dict(),threshold=threshold,epoch=epoch+1+offset,tile_size=512,stride=384),ROOT/'scratch_detail_best.pt'); model.cuda()
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__': main()
