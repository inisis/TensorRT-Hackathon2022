import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("/workspace/decoder.onnx"))

gather = [node for node in graph.nodes if node.name == "Gather_154"] [0]

cast_out = gs.Variable("cast_out", dtype=np.int64)

cast = gs.Node(op="Cast", name= "Slice_82_Gather_154", attrs={"to" : 7,}, inputs=[gather.inputs[1]], outputs=[cast_out])
graph.nodes.append(cast)

gather.inputs[1] = cast.outputs[0]

list_ = []
for idx, node in enumerate(graph.nodes):
    if 1:
        list_.append({node.name: idx})
        # print(node.name)
        if 'Add' in node.name and 'norm' in node.inputs[1].name:
            # print(list_)
            sub_index = list(list_[-10].values())[0]
            reduce_index = list(list_[-11].values())[0]

            attrs = {}
            # The 2nd dimension of the Reshape shape is the number of groups
            attrs["num_groups"] = 1
            attrs["eps"] = 1e-5

            # 1 is the default plugin version the parser will search for, and therefore can be omitted,
            # but we include it here for illustrative purposes.
            attrs["plugin_version"] = "1"

            # "" is the default plugin namespace the parser will use, included here for illustrative purposes
            attrs["plugin_namespace"] = ""

            layer_norm_out = gs.Variable("layernorm_out_"+str(idx), dtype=np.float32)
            layer_norm = gs.Node(op="LayerNormPlugin", name= "LayerNorm_" + str(idx), inputs=[graph.nodes[reduce_index].inputs[0], graph.nodes[idx-1].inputs[1], node.inputs[1]], outputs=node.outputs, attrs=attrs)

            graph.nodes[sub_index].inputs.clear()
            graph.nodes[reduce_index].inputs.clear()
            node.outputs.clear()
            graph.nodes.append(layer_norm)


graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "./ModifiedDecoder.onnx")
