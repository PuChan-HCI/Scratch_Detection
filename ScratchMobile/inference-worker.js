importScripts('vendor/ort.min.js');
ort.env.wasm.wasmPaths = new URL('vendor/', self.location.href).href;
ort.env.wasm.numThreads = 1;
let session, selected;
onmessage = async ({data: m}) => {
  try {
    if (m.type === 'load') {
      if (session) { await session.release(); session = null; }
      selected = m.model;
      session = await ort.InferenceSession.create(selected === 'lite' ? 'models/scratch_lite.onnx' : 'models/original_probability.onnx', {executionProviders:['wasm']});
      postMessage({id:m.id, loaded:true});
    } else {
      const tensor = new ort.Tensor('float32', m.data, [1,3,m.height,m.width]);
      const result = await session.run({[session.inputNames[0]]:tensor});
      const output = selected === 'original' ? result[session.outputNames[1]] : result[session.outputNames[0]];
      const data = new Float32Array(output.data);
      postMessage({id:m.id,data,width:output.dims[3],height:output.dims[2]}, [data.buffer]);
      tensor.dispose(); Object.values(result).forEach(t=>t.dispose());
    }
  } catch(e) { postMessage({id:m.id,error:String(e.message || e)}); }
};
