import onnx
import onnx_graphsurgeon as gs
import numpy as np

######################### layernorm ######################################

graph = gs.import_onnx(onnx.load('/workspace/encoder.onnx'))
list_ = {}
for idx, node in enumerate(graph.nodes):
    list_.update({node.name: node})


pos_emb_gemm_node = list_["MatMul_141"]

q_gemm_node = list_["MatMul_119"]
q_add_node = list_["Add_120"]

k_gemm_node = list_["MatMul_125"] 
k_add_node = list_["Add_126"]

v_gemm_node = list_["MatMul_131"]
v_add_node = list_["Add_132"]

pos_bias_u_node = list_["Add_146"] # * key

pos_bias_v_node = list_["Add_148"] # * pos

fc_gemm_node = list_["MatMul_178"] # * pos
fc_add_node = list_["Add_179"] # * pos

pos_emb_gemm_node_weights = gs.Constant("pos_emb_gemm_node_weights", values=pos_emb_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
q_gemm_node_weights = gs.Constant("q_gemm_node_weights", values=q_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
k_gemm_node_weights = gs.Constant("k_gemm_node_weights", values=k_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
v_gemm_node_weights = gs.Constant("v_gemm_node_weights", values=v_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
fc_gemm_node_weights = gs.Constant("fc_gemm_node_weights", values=fc_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))

attrs = {}
attrs["plugin_version"] = "1"
attrs["plugin_namespace"] = ""

x = gs.Variable(name="x", dtype=np.float32, shape=(1, 3, 256))
mask = gs.Variable(name="mask", dtype=np.float32, shape=(1, 1, 3))
pos_emb = gs.Variable(name="pos_emb", dtype=np.float32, shape=(1, 3, 256))

mha_out_0 = gs.Variable("mha_out_"+str(0), dtype=np.float32)

def gi(node, index=1):
    return node.inputs[index]

mha_norm = gs.Node(op="MHAPlugin", name= "MHA_" + str(0), inputs=[x, mask, pos_emb,
                                                                  pos_emb_gemm_node_weights,
                                                                  q_gemm_node_weights,
                                                                  gi(q_add_node, 0),
                                                                  k_gemm_node_weights,
                                                                  gi(k_add_node, 0),
                                                                  v_gemm_node_weights,
                                                                  gi(v_add_node, 0),
                                                                  gi(pos_bias_u_node),
                                                                  gi(pos_bias_v_node),
                                                                  fc_gemm_node_weights,
                                                                  gi(fc_add_node, 0)],
                                                                  outputs=[mha_out_0],
                                                                  attrs=attrs)

graph = gs.Graph(nodes=[mha_norm], inputs=[x, mask, pos_emb], outputs=[mha_out_0])
onnx.save(gs.export_onnx(graph), "model.onnx")
