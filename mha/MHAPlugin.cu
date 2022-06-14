#include <cub/cub.cuh>
#include <numeric>
#include "MHAPlugin.h"
#include "serialize.hpp"
#include "src/fastertransformer/models/xlnet/Xlnet.h"

using namespace fastertransformer;

namespace nvinfer1
{
MHAPlugin::MHAPlugin(const std::string &name)
{
}

int MHAPlugin::initialize() noexcept
{
    return 0;
}

MHAPlugin::MHAPlugin(const void* data, size_t length)
{
}

const char* MHAPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* MHAPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int MHAPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs MHAPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    auto input_dimsexprs = inputs[0];
    output.d[2] = exprBuilder.constant(int(input_dimsexprs.d[2]->getConstantValue()));
 
    return output;
}

void MHAPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void MHAPlugin::detachFromContext() noexcept
{
}

int MHAPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    size_t batch_size = input_dims.d[0];
    size_t seq_len = input_dims.d[1];
    size_t hidden_units = input_dims.d[2];

    if (input_dims.d[0] == -1)
        batch_size = 1;
    if (input_dims.d[1] == -1)
        seq_len = 1;
    std::cout << sizeof(float) << std::endl; 
    std::cout << sizeof(half) << std::endl; 
    std::cout << sizeof(bool) << std::endl; 
    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, static_cast<const float*>(inputs[0])}, // x

        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{batch_size, 1, seq_len}, static_cast<const float*>(inputs[1])},  // mask
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{1, seq_len, hidden_units}, static_cast<const float*>(inputs[2])}, // pos_emb

        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units, hidden_units}, static_cast<const float*>(inputs[3])}, //  pos_emb weight

        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units, hidden_units}, static_cast<const float*>(inputs[4])},
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units}, static_cast<const float*>(inputs[5])},
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units, hidden_units}, static_cast<const float*>(inputs[6])},
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units}, static_cast<const float*>(inputs[7])},
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units, hidden_units}, static_cast<const float*>(inputs[8])},
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units}, static_cast<const float*>(inputs[9])},

	Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units}, static_cast<const float*>(inputs[10])}, // pos_bias_u
	Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units}, static_cast<const float*>(inputs[11])}, // pos_bias_v
	
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units, hidden_units}, static_cast<const float*>(inputs[12])}, // fc weight
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{hidden_units}, static_cast<const float*>(inputs[13])}, // fc bias
        };

    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, static_cast<const float*>(outputs[0])}};

    xlnet_->forward(&output_tensors, &input_tensors, static_cast<const float*>(weight_ptr));

    return 0;
}

size_t MHAPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void MHAPlugin::serialize(void* buffer) const noexcept
{
}

bool MHAPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == nvinfer1::PluginFormat::kLINEAR));
}

void MHAPlugin::terminate() noexcept
{
}

void MHAPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    cublas_wrapper_ = nullptr;
    delete this;
}

IPluginV2DynamicExt* MHAPlugin::clone() const noexcept
{
    auto* plugin = new MHAPlugin(mPluginNamespace);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void MHAPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{

    for (int i = 0; i < nbInputs; i++)
    {
      for (int j = 0; j < in[0].desc.dims.nbDims; j++)
      {
        // Do not support dynamic dimensions
        assert(in[0].desc.dims.d[j] != -1);
      }
    }

    int batchSize = in[0].desc.dims.d[0];
    int nbChannels = in[0].desc.dims.d[1];

    if(batchSize == -1)
        batchSize = 1;
    if(nbChannels == -1)
        nbChannels = 1;

    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);

    cublas_algo_map_ = new cublasAlgoMap("gemm_config.in", "");
    allocator_ = new Allocator<AllocatorType::CUDA>(getDevice());
    cublas_wrapper_mutex_ = new std::mutex();
    cublas_wrapper_ = new cublasMMWrapper(cublas_handle_, cublaslt_handle_, nullptr, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);
    // cublas_wrapper_->setFP16GemmConfig();
    cublas_wrapper_->setFP32GemmConfig();
    xlnet_ = new Xlnet<float>(batchSize,
                              nbChannels,
                              4,
                              64,
                              4 * 256,
                              1,
                              1.0f,
                              stream_,
                              cublas_wrapper_,
                              allocator_,
                              false);
}

nvinfer1::DataType MHAPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t MHAPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void MHAPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* MHAPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

MHAPluginCreator::MHAPluginCreator()
{
}

const char* MHAPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* MHAPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* MHAPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* MHAPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void MHAPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* MHAPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    MHAPlugin* plugin = new MHAPlugin(name);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* MHAPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    MHAPlugin* plugin = new MHAPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

PluginFieldCollection MHAPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> MHAPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MHAPluginCreator);

} // namespace nvinfer1
