"""Full-frame lightweight scratch inference, preserving native image resolution."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from detect_scratches import overlay

class LitePredictor:
    def __init__(self,path,device='auto'):
        path=Path(path)
        if path.suffix.lower()=='.pt':
            import torch
            from scratch_lite_model import ScratchLite
            torch.set_num_threads(4); torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False; self.torch=torch
            self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
            ck=torch.load(path,weights_only=True,map_location='cpu'); self.threshold=ck['threshold']
            self.model=ScratchLite(); self.model.load_state_dict(ck['state_dict']); self.model.to(self.device).eval(); self.session=None
        else:
            import onnxruntime as ort
            options=ort.SessionOptions(); options.intra_op_num_threads=4
            providers=['CPUExecutionProvider']
            if device=='cuda':
                if 'CUDAExecutionProvider' not in ort.get_available_providers(): raise RuntimeError('CUDA ONNX Runtime is unavailable; use scratch_lite.pt for GPU inference.')
                providers=['CUDAExecutionProvider','CPUExecutionProvider']
            self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
            self.threshold=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['threshold']; self.device=providers[0]
    def __call__(self,rgb):
        x=np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255
        if self.session is not None: return self.session.run(None,{'rgb':x})[0][0,0]
        with self.torch.inference_mode(): return self.model(self.torch.from_numpy(x).to(self.device)).sigmoid()[0,0].cpu().numpy()

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(); p.add_argument('--model',default=str(root/'scratch_lite.onnx')); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--threshold',type=float); p.add_argument('--min-pixels',type=int,default=3); p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args(); source=Path(a.input); target=Path(a.output)
    if source.resolve()==target.resolve(): raise ValueError('Input and output must be different files')
    model=LitePredictor(a.model,a.device); threshold=model.threshold if a.threshold is None else a.threshold
    if not 0<threshold<1 or a.min_pixels<1: raise ValueError('Threshold must be between 0 and 1; min-pixels must be positive')
    target.parent.mkdir(parents=True,exist_ok=True); print(f'Backend: {model.device}; threshold={threshold}',flush=True)
    def process(bgr): return overlay(bgr,model(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),threshold,a.min_pixels)
    if source.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']:
        im=cv2.imread(str(source))
        if im is None: raise ValueError('Cannot read input image')
        out,boxes=process(im)
        if not cv2.imwrite(str(target),out): raise RuntimeError('Cannot write image')
        print(json.dumps(boxes)); return
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise ValueError('Cannot read input video')
    size=(int(cap.get(3)),int(cap.get(4))); fps=cap.get(5) or 30
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    if not writer.isOpened(): cap.release(); raise RuntimeError('Cannot open video writer')
    count=0; started=time.perf_counter()
    try:
        with open(target.with_suffix('.jsonl'),'w') as log:
            while not a.max_frames or count<a.max_frames:
                ok,frame=cap.read()
                if not ok: break
                out,boxes=process(frame); writer.write(out); log.write(json.dumps(dict(frame=count,boxes=boxes))+'\n'); count+=1
                if count%60==0: print(f'Processed {count} frames',flush=True)
    finally: cap.release(); writer.release()
    print(f'Saved {count} frames in {time.perf_counter()-started:.2f}s: {target}',flush=True)

if __name__=='__main__': main()
