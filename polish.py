import onnx
import onnx_graphsurgeon as gs
import numpy as np

######################### layernorm ######################################

def gi(node, index=1):
    return node.inputs[index]

graph = gs.import_onnx(onnx.load('./encoder_sim.onnx'))
graph.cleanup().toposort()

list_ = {}
for idx, node in enumerate(graph.nodes):
    list_.update({node.name: node})

for idx, node in enumerate(graph.nodes):
    if 1:
        if 'Add' in node.name and 'norm' in node.inputs[1].name:
            # print(list_)
            index = node.name.split('_')[1]
            sub_name = "ReduceMean_" + str(int(index)-10)
            reduce_name = "Sub_" + str(int(index)-9)
            mul_name = "Mul_" + str(int(index)-1)
            sub_node = list_[sub_name]
            reduce_node = list_[reduce_name]
            mul_node = list_[mul_name]
            add_node = list_[node.name]

            attrs = {}
            attrs["num_groups"] = 1
            attrs["eps"] = 1e-5
            attrs["plugin_version"] = "1"
            attrs["plugin_namespace"] = ""

            layer_norm = gs.Node(op="LayerNormPlugin", name= "LayerNorm_" + str(index), inputs=[reduce_node.inputs[0], mul_node.inputs[1], add_node.inputs[1]], outputs=add_node.outputs, attrs=attrs)
            sub_node.inputs.clear()
            reduce_node.inputs.clear()
            
            if "Shape_" + str(int(index)+1) not in list_.keys():
                add_node.outputs.clear()

            graph.nodes.append(layer_norm)

        if 'Softmax' in node.name and 'Log' not in node.name:
            index = node.name.split('_')[1]
            input_Add = "Add_" + str(int(index)-53)
            input_MatMul = "MatMul_" + str(int(index)-27)
            input_UnSqueeze = "Unsqueeze_" + str(int(index)-7)

            add_node = list_[input_Add]
            mul_node = list_[input_MatMul]
            unsqueeze_node = list_[input_UnSqueeze]

            q_Mat_node = list_["MatMul_" + str(int(index)-37)]
            k_Mat_node = list_["MatMul_" + str(int(index)-49)]
            v_Mat_mode = list_["MatMul_" + str(int(index)-43)]
            shape_node = list_["Shape_" + str(int(index)-52)]

            pos_emb_gemm_node = list_["MatMul_" + str(int(index)-27)]
            pos_emb_gemm_node_weights = gs.Constant("pos_emb_gemm_node_weights_" + str(index), values=pos_emb_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))

            q_gemm_node = list_["MatMul_"+str(int(index)-49)]
            q_gemm_node_weights = gs.Constant("q_gemm_node_weights_" + str(index), values=q_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
            q_add_node = list_["Add_"+str(int(index)-48)]

            k_gemm_node = list_["MatMul_"+str(int(index)-43)]
            k_gemm_node_weights = gs.Constant("k_gemm_node_weights_" + str(index), values=k_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
            k_add_node = list_["Add_"+str(int(index)-42)]

            v_gemm_node = list_["MatMul_"+str(int(index)-37)]
            v_gemm_node_weights = gs.Constant("v_gemm_node_weights_" + str(index), values=v_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
            v_add_node = list_["Add_"+str(int(index)-36)]

            pos_bias_u_node = list_["Add_"+str(int(index)-22)] # * key

            pos_bias_v_node = list_["Add_"+str(int(index)-20)] # * pos

            fc_gemm_node = list_["MatMul_"+str(int(index)+10)] # * pos
            fc_gemm_node_weights = gs.Constant("fc_gemm_node_weights_" + str(index), values=fc_gemm_node.inputs[1].values.reshape(256, 256).transpose(1,0))
            fc_add_node = list_["Add_"+str(int(index)+11)] # * pos

            attrs_dict = {}
            attrs_dict['to'] = 1
            Not_Cast_output = gs.Variable(name="New_Cast_output" + str(index), dtype=None, shape=None)
            newNode = gs.Node(name="NewCast_" + str(index), op="Cast", inputs=[unsqueeze_node.inputs[0]],
                  outputs=[Not_Cast_output], attrs=attrs_dict)

            attrs = {}
            attrs["plugin_version"] = "1"
            attrs["plugin_namespace"] = ""

            mha_plugin = gs.Node(op="MHAPlugin", name= "MHA_" + str(index), inputs=[add_node.outputs[0],
                                 Not_Cast_output, # mask
                                 mul_node.inputs[0],# position embedding
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
                                 outputs=[fc_add_node.outputs[0]],
                                 attrs=attrs)
            
            unsqueeze_node.inputs.clear()
            mul_node.inputs.clear()
            fc_add_node.outputs.clear()
            q_Mat_node.inputs.clear()
            k_Mat_node.inputs.clear()
            v_Mat_mode.inputs.clear()
            shape_node.inputs.clear()
            add_node.outputs.clear()
            graph.nodes.append(newNode)
            graph.nodes.append(mha_plugin)

    '''
    if 'Sigmoid' in node.name:
        if 'Split' in graph.nodes[idx-1].name:
            split_index = idx - 1
            mul_index = idx + 1
            attrs = {}
            attrs["plugin_version"] = "1"
            attrs["plugin_namespace"] = ""

            glu = gs.Node(op="GluPlugin", name= "Glu_" + str(idx), inputs=graph.nodes[split_index].inputs, outputs=graph.nodes[mul_index].outputs, attrs=attrs)

            graph.nodes[split_index].inputs.clear()
            graph.nodes[mul_index].outputs.clear()
            graph.nodes.append(glu)

        else:
            mul_index = idx + 1
            attrs = {}
            attrs["plugin_version"] = "1"
            attrs["plugin_namespace"] = ""
    
            swish = gs.Node(op="SwishPlugin", name= "Swish_" + str(idx), inputs=node.inputs, outputs=graph.nodes[mul_index].outputs, attrs=attrs)
    
            graph.nodes[mul_index].inputs.clear()
            graph.nodes[mul_index].outputs.clear()
            node.inputs.clear()
            graph.nodes.append(swish)
    '''

Unsqueeze_29 = list_["Unsqueeze_29"]
Not_30 = list_["Not_30"]
Slice_79 = list_["Slice_79"]
Slice_84 = list_["Slice_84"]

start_node = Unsqueeze_29.outputs[0]
Unsqueeze_29_Cast_output = gs.Variable(name="Unsqueeze_29_Cast_output", dtype=None, shape=None)
attrs_dict = {}
attrs_dict['to'] = 6
newNode = gs.Node(name="Slice_84_Cast", op="Cast", inputs=[start_node],
                  outputs=[Unsqueeze_29_Cast_output], attrs=attrs_dict)
graph.nodes.append(newNode)

Slice_79.inputs[0] = Unsqueeze_29_Cast_output
Slice_84_outputs = Not_30.outputs[0]
end_node = Slice_84.outputs[0]
Not_30.outputs[0] = end_node
Slice_84.outputs[0] = Slice_84_outputs
Not_30.inputs[0] = Slice_84.outputs[0]

Slice_84_Cast_output = gs.Variable(name="Slice_84_Cast_output", dtype=np.bool_, shape=None)
attrs_dict = {}
attrs_dict['to'] = 9
newNode = gs.Node(name="Slice_84_Cast_", op="Cast", inputs=[Slice_84_outputs ],
                  outputs=[Slice_84_Cast_output], attrs=attrs_dict)
graph.nodes.append(newNode)
Not_30.inputs[0] = Slice_84_Cast_output

Not_30_Cast_output = gs.Variable(name="Not_30_output", dtype=np.bool_, shape=None)

graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "./ModifyEncoder.onnx")

########################################## Not ########################################################
