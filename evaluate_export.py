import sys,json,argparse,time
from pathlib import Path
import cv2,numpy as np,torch,onnxruntime as ort
from scratch_model import ScratchNet
from train_scratch import load_data,split_data,predict,measure,ROOT

def baseline(data,ids,path):
    options=ort.SessionOptions(); options.log_severity_level=3; options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']); raw=[]; post=[]; values=set()
    for i in ids:
        d=data[i]; h,w=d['mask'].shape
        x=cv2.resize(d['rgb'],(505,256)).transpose(2,0,1)[None].astype(np.float32)/255
        y=session.run(None,{session.get_inputs()[0].name:x})[0][0,0]; values.update(np.unique(y).tolist())
        b=cv2.resize((y>0).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        raw.append(b); b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)); post.append(b)
    return raw,post,sorted(values)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline-only',action='store_true'); p.add_argument('--source',type=Path,default=Path(r'D:\InnCoreTech\Scratch_Detection')); a=p.parse_args()
    torch.set_num_threads(4)
    source=a.source; data=load_data(source/'coco_instance_dataset'); splits=split_data(data); ids=splits['test']
    raw,post,values=baseline(data,ids,source/'one_ai_model.onnx')
    result=dict(test_images=len(ids),baseline_labels=values,baseline_raw=measure(raw,data,ids,.5),baseline_original_postprocess=measure(post,data,ids,.5),limitations=['Original model training provenance unknown; it may have trained on these test images.','Temporal blocks with 3-frame guards, same objects/capture sessions; not an independent product test.','COCO polygon rasterization uses rounded coordinates with OpenCV.','Object hit: at least 25% of annotated pixels recovered; this is not instance AP.','Small: mask area <=1024 pixels; thin: 2*area/perimeter <=6 pixels.'])
    if a.baseline_only:
        json.dump(result,open(ROOT/'baseline_metrics.json','w'),indent=2); print(json.dumps(result)); return
    ck=torch.load(ROOT/'scratch_detail_best.pt',weights_only=True,map_location='cpu'); model=ScratchNet(); model.load_state_dict(ck['state_dict']); model.cuda().eval()
    size=ck.get('tile_size',320); stride=ck.get('stride',256)
    val_predictions=[predict(model,data[i]['rgb'],size,stride) for i in splits['validation']]
    sensitivity_options=[(t,measure(val_predictions,data,splits['validation'],t)) for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]]
    def f2(option):
        metrics=option[1]; precision=metrics['precision']; recall=metrics['recall']
        return 5*precision*recall/max(4*precision+recall,1e-12)
    sensitive_threshold,sensitive_validation=max(sensitivity_options,key=f2)
    preds=[predict(model,data[i]['rgb'],size,stride) for i in ids]; result['new_model']=measure(preds,data,ids,ck['threshold']); result['threshold']=ck['threshold']; result['selected_epoch']=ck['epoch']
    result['selection']='Best validation pixel Dice; threshold chosen on validation only.'
    result['limitations'].append('This holdout was inspected during development before low-learning-rate training and the change to 512-pixel tiles; final results are a development benchmark, not a pristine independent test.')
    json.dump(result,open(ROOT/'evaluation.json','w'),indent=2)
    for j in [i for i in range(len(ids)) if data[ids[i]]['objects']][:6]:
        d=data[ids[j]]; im=cv2.cvtColor(d['rgb'],cv2.COLOR_RGB2BGR); views=[]
        for label,mask in [('Image',np.zeros(im.shape[:2],bool)),('Annotation',d['mask']>0),('Original',post[j]>.5),('New',preds[j]>=ck['threshold'])]:
            v=im.copy(); v[mask]=(v[mask]*.5+np.array([0,0,255])*.5).astype(np.uint8); v=cv2.resize(v,(632,320)); cv2.putText(v,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2); views.append(v)
        cv2.imwrite(str(ROOT/f'comparison_{ids[j]:03d}.jpg'),np.vstack([np.hstack(views[:2]),np.hstack(views[2:])]))
    sys.path.insert(0,str(ROOT.parent/'work/python_deps'))
    import onnx
    wrapped=torch.nn.Sequential(model.cpu(),torch.nn.Sigmoid()).eval(); dummy=torch.rand(1,3,size,size)
    torch.onnx.export(wrapped,dummy,str(ROOT/'scratch_detail.onnx'),input_names=['rgb'],output_names=['scratch_probability'],opset_version=17,dynamo=False)
    onnx.checker.check_model(str(ROOT/'scratch_detail.onnx'))
    options=ort.SessionOptions(); options.intra_op_num_threads=4
    s=ort.InferenceSession(str(ROOT/'scratch_detail.onnx'),sess_options=options)
    with torch.no_grad(): expected=wrapped(dummy).numpy()
    actual=s.run(None,{'rgb':dummy.numpy()})[0]; err=float(np.abs(actual-expected).max()); assert np.allclose(actual,expected,atol=1e-4,rtol=1e-3),err
    config=dict(threshold=ck['threshold'],input=f'RGB float32 NCHW [1,3,{size},{size}], range 0..1; normalization inside model',output=f'scratch probability float32 [1,1,{size},{size}]',tile_size=size,stride=stride,min_pixels=3,onnx_max_absolute_error=err)
    config['sensitive_threshold']=sensitive_threshold
    json.dump(config,open(ROOT/'scratch_detail.json','w'),indent=2)
    from detect_scratches import probability
    started=time.time(); onnx_preds=[probability(s,data[i]['rgb']) for i in ids]
    result['onnx_cpu_seconds_per_image']=(time.time()-started)/len(ids)
    result['new_model_pytorch']=result['new_model']; result['new_model']=measure(onnx_preds,data,ids,ck['threshold'])
    result['sensitive_threshold']=sensitive_threshold
    result['sensitive_validation']=sensitive_validation
    result['new_model_sensitive']=measure(onnx_preds,data,ids,sensitive_threshold)
    filtered=[]
    for pred in onnx_preds:
        n,labels,stats,_=cv2.connectedComponentsWithStats((pred>=ck['threshold']).astype(np.uint8),8)
        keep=np.where(stats[:,cv2.CC_STAT_AREA]>=3)[0]; keep=keep[keep!=0]; filtered.append(np.isin(labels,keep))
    result['new_model_display_filter']=measure(filtered,data,ids,.5)
    json.dump(result,open(ROOT/'evaluation.json','w'),indent=2)
    print(json.dumps(result),flush=True); print('ONNX VERIFIED',err,flush=True)

if __name__=='__main__': main()
