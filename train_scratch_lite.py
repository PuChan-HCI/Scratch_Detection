"""Train a 2,741-parameter native-resolution refiner; keep original backbone frozen."""
import sys,json,random,time,argparse
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from scratch_lite_model import ScratchLite
from train_scratch import load_data,split_data,measure

ROOT=Path(__file__).resolve().parent

def cached_batch(data,ids,rng,n=8,size=320):
    positives=[i for i in ids if data[i]['objects']]; xs=[]; ps=[]; ys=[]; vs=[]
    for _ in range(n):
        focus=rng.random()<.5; d=data[rng.choice(positives if focus else ids)]; h,w=d['mask'].shape
        if focus:
            a=rng.choice(d['objects']); x,y,bw,bh=a['bbox']
            x=int(np.clip(x+rng.random()*bw-rng.uniform(.15,.85)*size,0,w-size)); y=int(np.clip(y+rng.random()*bh-rng.uniform(.15,.85)*size,0,h-size))
        else: x=rng.randint(0,w-size); y=rng.randint(0,h-size)
        arrays=[d['rgb'][y:y+size,x:x+size],d['prior'][y:y+size,x:x+size],d['mask'][y:y+size,x:x+size],1-d['ignore'][y:y+size,x:x+size]]
        k=rng.randrange(4); flip=rng.random()<.5
        arrays=[np.rot90(a,k)[:,::-1].copy() if flip else np.rot90(a,k).copy() for a in arrays]
        im,prior,mask,valid=arrays
        # No RGB-only lighting augmentation: cached prior must correspond to the same image.
        xs.append(im.transpose(2,0,1).astype(np.float32)/255); ps.append(prior[None]); ys.append(mask[None]); vs.append(valid[None])
    return [torch.from_numpy(np.stack(a)).float().cuda() for a in [xs,ps,ys,vs]]

@torch.inference_mode()
def predict_cached(model,d):
    x=torch.from_numpy(d['rgb'].transpose(2,0,1).copy()[None]).float().cuda()/255
    prior=torch.from_numpy(d['prior'][None,None]).cuda()
    return model.refiner(x,prior).sigmoid()[0,0].cpu().numpy()

def load_checkpoint(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=True); model=ScratchLite(); model.load_state_dict(ck['state_dict']); model.to(device).eval(); return model,ck

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,default=Path(r'D:\InnCoreTech\Scratch_Detection')); p.add_argument('--epochs',type=int,default=40); p.add_argument('--steps',type=int,default=60); args=p.parse_args()
    sys.path.insert(0,str(ROOT.parent/'work/python_deps'))
    torch.set_num_threads(4); torch.manual_seed(83); np.random.seed(83); random.seed(83)
    torch.backends.cudnn.benchmark=False
    model=ScratchLite(); model.backbone.load_original_onnx(args.source/'one_ai_model.onnx'); model.cuda()
    for parameter in model.backbone.parameters(): parameter.requires_grad=False
    data=load_data(args.source/'coco_instance_dataset'); splits=split_data(data)
    model.eval()
    with torch.inference_mode():
        for d in data:
            x=torch.from_numpy(d['rgb'].transpose(2,0,1).copy()[None]).float().cuda()/255
            d['prior']=model.prior(x)[0,0].cpu().numpy()
    print('PARAMETERS',sum(p.numel() for p in model.parameters()),'TRAINABLE',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
    json.dump({k:[data[i]['info']['file_name'] for i in v] for k,v in splits.items()},open(ROOT/'lite_split_manifest.json','w'),indent=2)
    opt=torch.optim.AdamW(model.refiner.parameters(),lr=.001,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.00005)
    rng=random.Random(83); best=-1; history=[]; started=time.time()
    for epoch in range(args.epochs+1):
        if epoch:
            model.refiner.train(); losses=[]
            for _ in range(args.steps):
                x,prior,y,v=cached_batch(data,splits['train'],rng); opt.zero_grad(set_to_none=True)
                logits=model.refiner(x,prior); prob=logits.sigmoid()
                bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
                loss=(bce*v).sum()/v.sum().clamp_min(1)+1-(2*(prob*y*v).sum()+1)/((prob*v).sum()+(y*v).sum()+1)
                loss=loss+.0001*((logits-prior)**2).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.refiner.parameters(),1.,error_if_nonfinite=True); opt.step(); losses.append(loss.item())
            schedule.step()
        model.eval(); predictions=[predict_cached(model,data[i]) for i in splits['validation']]
        options=[(t,measure(predictions,data,splits['validation'],t)) for t in [.2,.3,.4,.5,.6,.7,.8,.9]]
        threshold,metrics=max(options,key=lambda v:v[1]['dice'])
        row=dict(epoch=epoch,loss=float(np.mean(losses)) if epoch else None,threshold=threshold,validation=metrics,elapsed_seconds=round(time.time()-started,2))
        history.append(row); json.dump(history,open(ROOT/'lite_training_history.json','w'),indent=2); print(json.dumps(row),flush=True)
        if metrics['dice']>best:
            best=metrics['dice']; torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},threshold=threshold,epoch=epoch,architecture='ScratchLite',parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.refiner.parameters())),ROOT/'scratch_lite.pt')
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__': main()
