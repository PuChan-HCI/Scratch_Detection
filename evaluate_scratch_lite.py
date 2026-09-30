import sys,json,time,argparse,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from train_scratch import load_data,split_data,measure
from train_scratch_lite import load_checkpoint
from evaluate_export import baseline
from detect_scratches_lite import LitePredictor

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,default=Path(r'D:\InnCoreTech\Scratch_Detection')); args=p.parse_args()
    sys.path.insert(0,str(ROOT.parent/'work/python_deps'))
    import onnx,onnxruntime as ort
    torch.set_num_threads(4); model,ck=load_checkpoint(ROOT/'scratch_lite.pt'); model.eval()
    wrapped=torch.nn.Sequential(model,torch.nn.Sigmoid()); dummy=torch.rand(1,3,384,640)
    torch.onnx.export(wrapped,dummy,str(ROOT/'scratch_lite.onnx'),input_names=['rgb'],output_names=['scratch_probability'],dynamic_axes={'rgb':{2:'height',3:'width'},'scratch_probability':{2:'height',3:'width'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(ROOT/'scratch_lite.onnx'))
    config=dict(architecture='ScratchLite',threshold=ck['threshold'],input='RGB float32 NCHW [1,3,H,W], range 0..1',output='Scratch probability float32 [1,1,H,W]',parameters=ck['parameters'],trainable_parameters=ck['trainable_parameters'],selected_epoch=ck['epoch'],preprocessing='Whole image is resized internally for the frozen backbone; refinement uses original-resolution RGB. Do not tile this model.',min_pixels=3)
    json.dump(config,open(ROOT/'scratch_lite.json','w'),indent=2)
    options=ort.SessionOptions(); options.intra_op_num_threads=4; session=ort.InferenceSession(str(ROOT/'scratch_lite.onnx'),sess_options=options)
    data=load_data(args.source/'coco_instance_dataset'); splits=split_data(data); ids=splits['test']
    errors=[]
    for shape in [(96,144),(384,640),(640,1264),(808,1288)]:
        x=np.random.default_rng(83).random((1,3,*shape),dtype=np.float32)
        with torch.inference_mode(): reference=wrapped(torch.from_numpy(x)).numpy()
        actual=session.run(None,{'rgb':x})[0]; error=float(np.abs(reference-actual).max()); errors.append(dict(shape=shape,max_absolute_error=error))
        assert error<2e-4,(shape,error)
    original_raw,original_post,labels=baseline(data,ids,args.source/'one_ai_model.onnx')
    predictions=[]; times=[]
    for i in ids:
        rgb=data[i]['rgb']; x=np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255
        started=time.perf_counter(); pred=session.run(None,{'rgb':x})[0][0,0]; times.append(time.perf_counter()-started); predictions.append(pred)
    metrics=measure(predictions,data,ids,ck['threshold'])
    result=dict(test_images=len(ids),threshold=ck['threshold'],selected_epoch=ck['epoch'],parameters=ck['parameters'],trainable_parameters=ck['trainable_parameters'],original=measure(original_post,data,ids,.5),lite=metrics,onnx_bytes=(ROOT/'scratch_lite.onnx').stat().st_size,original_bytes=(args.source/'one_ai_model.onnx').stat().st_size,previous_large_bytes=(ROOT/'scratch_detail.onnx').stat().st_size,onnx_cpu_seconds_per_image=float(np.mean(times)),onnx_parity=errors,limitations=['The 33-image development benchmark has been inspected in previous iterations.','Original backbone weights are reused; their training provenance is unknown and may include all these images. No claim of an independent unseen test.','143 images train the refinement layers, 33 choose the threshold/checkpoint, 33 are used for development comparison, 24 temporal guard frames are unused.','All images show the same objects/capture sessions; external data is needed to establish generalization.','Small: annotated area <=1024 native pixels. Thin: 2*area/perimeter <=6 pixels. Only 12 small and 3 thin objects in this benchmark.','Object hit means >=25% annotated pixel coverage, not instance AP.'])
    # Report subgroup overlap fairly: only pixel recall inside annotations, not subgroup precision.
    for j in [j for j,i in enumerate(ids) if data[i]['objects']][:6]:
        d=data[ids[j]]; im=cv2.cvtColor(d['rgb'],cv2.COLOR_RGB2BGR); views=[]
        for name,mask in [('Input',np.zeros(im.shape[:2],bool)),('Annotation',d['mask']>0),('Original',original_post[j]>.5),('Lightweight',predictions[j]>=ck['threshold'])]:
            view=im.copy(); view[mask]=(view[mask]*.5+np.array([0,0,255])*.5).astype(np.uint8)
            view=cv2.resize(view,(632,round(im.shape[0]*632/im.shape[1]))); cv2.putText(view,name,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2); views.append(view)
        cv2.imwrite(str(ROOT/f'lite_comparison_{ids[j]:03d}.jpg'),np.vstack([np.hstack(views[:2]),np.hstack(views[2:])]))
    gpu=LitePredictor(ROOT/'scratch_lite.pt','cuda'); rgb=data[ids[0]]['rgb']; gpu(rgb)
    started=time.perf_counter(); gp=gpu(rgb); result['gpu_seconds_one_image']=time.perf_counter()-started
    result['gpu_onnx_max_error']=float(np.abs(gp-predictions[0]).max()); assert result['gpu_onnx_max_error']<.001
    config['onnx_sha256']=hashlib.sha256((ROOT/'scratch_lite.onnx').read_bytes()).hexdigest()
    json.dump(config,open(ROOT/'scratch_lite.json','w'),indent=2); json.dump(result,open(ROOT/'lite_evaluation.json','w'),indent=2); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
