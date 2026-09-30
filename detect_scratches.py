"""Native-resolution overlapping tile inference, preserving thin scratch pixels."""
import argparse, json
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

class TorchSession:
    def __init__(self,path,device):
        import torch
        from types import SimpleNamespace
        from scratch_model import ScratchNet
        torch.set_num_threads(4)
        self.torch=torch
        self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        checkpoint=torch.load(path,weights_only=True,map_location='cpu')
        self.model=ScratchNet(); self.model.load_state_dict(checkpoint['state_dict']); self.model.to(self.device).eval()
        self.config={'threshold':checkpoint['threshold']}
        size=checkpoint.get('tile_size',320); self.meta=SimpleNamespace(name='rgb',shape=[1,3,size,size])
    def get_inputs(self): return [self.meta]
    def run(self,outputs,inputs):
        with self.torch.inference_mode():
            return [self.model(self.torch.from_numpy(inputs['rgb']).to(self.device)).sigmoid().cpu().numpy()]

def probability(session,rgb,size=None,stride=None):
    size=int(session.get_inputs()[0].shape[-1]) if size is None else size
    stride=int(size*.75) if stride is None else stride
    h,w=rgb.shape[:2]; rgb=cv2.copyMakeBorder(rgb,0,max(0,size-h),0,max(0,size-w),cv2.BORDER_REFLECT_101)
    ph,pw=rgb.shape[:2]; out=np.zeros((ph,pw),np.float32); count=np.zeros_like(out)
    yy=sorted(set(list(range(0,ph-size+1,stride))+[ph-size])); xx=sorted(set(list(range(0,pw-size+1,stride))+[pw-size]))
    for y in yy:
        for x in xx:
            tile=np.ascontiguousarray(rgb[y:y+size,x:x+size].transpose(2,0,1)[None],dtype=np.float32)/255
            p=session.run(None,{session.get_inputs()[0].name:tile})[0][0,0]
            out[y:y+size,x:x+size]+=p; count[y:y+size,x:x+size]+=1
    return (out/count)[:h,:w]

def overlay(frame,p,threshold,min_pixels):
    binary=(p>=threshold).astype(np.uint8); n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    kept=np.zeros_like(binary); boxes=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if area<min_pixels: continue
        kept[labels==i]=1; boxes.append(dict(x=x,y=y,w=w,h=h,pixels=area))
    out=frame.copy(); out[kept>0]=(out[kept>0]*.65+np.array([0,0,255])*.35).astype(np.uint8)
    for b in boxes: cv2.rectangle(out,(b['x'],b['y']),(b['x']+b['w'],b['y']+b['h']),(0,255,255),1)
    return out,boxes

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(); p.add_argument('--model',default=str(root/'scratch_detail.onnx')); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--threshold',type=float); p.add_argument('--min-pixels',type=int,default=3); p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args()
    if Path(a.model).suffix.lower()=='.pt':
        session=TorchSession(a.model,a.device); config=session.config
        print(f'PyTorch device: {session.device}',flush=True)
    else:
        config=json.load(open(Path(a.model).with_suffix('.json')))
        options=ort.SessionOptions(); options.intra_op_num_threads=4
        providers=['CPUExecutionProvider']
        if a.device=='cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers(): raise RuntimeError('CUDA ONNX Runtime unavailable. Use --model scratch_detail_best.pt --device cuda.')
            providers=['CUDAExecutionProvider','CPUExecutionProvider']
        session=ort.InferenceSession(a.model,sess_options=options,providers=providers)
    threshold=config['threshold'] if a.threshold is None else a.threshold
    src=Path(a.input); target=Path(a.output); target.parent.mkdir(parents=True,exist_ok=True)
    if src.resolve()==target.resolve(): raise ValueError('Input and output must differ')
    def process(frame): return overlay(frame,probability(session,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),threshold,a.min_pixels)
    if src.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']:
        frame=cv2.imread(str(src))
        if frame is None: raise ValueError('Cannot read image')
        out,boxes=process(frame)
        if not cv2.imwrite(str(target),out): raise RuntimeError('Cannot write output')
        print(json.dumps(boxes)); return
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened(): raise ValueError('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30; size=(int(cap.get(3)),int(cap.get(4)))
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    if not writer.isOpened(): raise RuntimeError('Cannot write video')
    idx=0
    try:
        with open(target.with_suffix('.jsonl'),'w') as f:
            while True:
                ok,frame=cap.read()
                if not ok: break
                out,boxes=process(frame); writer.write(out); f.write(json.dumps(dict(frame=idx,boxes=boxes))+'\n'); idx+=1
                if idx%20==0: print(f'Processed {idx} frames',flush=True)
                if a.max_frames and idx>=a.max_frames: break
    finally: cap.release(); writer.release()
    print(f'Saved {target}: {idx} frames')

if __name__=='__main__': main()
