"""Rebuild the original model with one additional probability output. Requires onnx."""
from pathlib import Path
import onnx
root=Path(__file__).resolve().parent/'models'
model=onnx.load(root/'one_ai_model.onnx')
name=[n.output[0] for n in model.graph.node if n.op_type=='Sigmoid'][-1]
model.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[1,1,128,252]))
onnx.checker.check_model(model)
onnx.save(model,root/'original_probability.onnx')
print('Saved original_probability.onnx (weights unchanged)')
