//#include <cub/cub.cuh>
#include "SwishPlugin.h"
#include "cuda_fp16.h"


#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

constexpr int CAFFE_CUDA_NUM_THREADS = 128;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;

inline int CAFFE_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
          CAFFE_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

__global__ void SwishCUDAKernel(const int N, const float* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __ldg(X + i) / (float(1) + exp(-__ldg(X + i)));
  }
}

__global__ void SwishCUDAKernel(const int N, const __half* X, __half* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __ldg(X + i) / (__half(1) + hexp(-__ldg(X + i)));
  }
}

namespace nvinfer1
{

SwishPlugin::SwishPlugin(const std::string &name)
{

}


SwishPlugin::SwishPlugin(const void* data, size_t length)
{

}

int SwishPlugin::initialize() noexcept
{
    return 0;
}

const char* SwishPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* SwishPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int SwishPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs SwishPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

void SwishPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void SwishPlugin::detachFromContext() noexcept
{
}

int SwishPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int batchSize = input_dims.d[0];
    int nbChannels = input_dims.d[1];
    int normDim = input_dims.d[2];
    if (batchSize == -1)
        batchSize == 1;
    if (nbChannels == -1)
        nbChannels == 1;
    // Calculate size of each group
    
    int M = batchSize * nbChannels;
    int N = normDim;

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        SwishCUDAKernel<<<CAFFE_GET_BLOCKS(M*N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(M*N, static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0])); 
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        SwishCUDAKernel<<<CAFFE_GET_BLOCKS(M*N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(M*N, static_cast<const __half*>(inputs[0]), static_cast<__half*>(outputs[0])); 
    }
    else
    {
        printf("Unsupport datatype!\n");
    }    
    return 0;
}

size_t SwishPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void SwishPlugin::serialize(void* buffer) const noexcept
{
}

bool SwishPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

void SwishPlugin::terminate() noexcept
{
}

void SwishPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* SwishPlugin::clone() const noexcept
{
    auto* plugin = new SwishPlugin(mNamespace);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void SwishPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

nvinfer1::DataType SwishPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

size_t SwishPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void SwishPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* SwishPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

SwishPluginCreator::SwishPluginCreator()
{
}

const char* SwishPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* SwishPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* SwishPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* SwishPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void SwishPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* SwishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    SwishPlugin* plugin = new SwishPlugin(name);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* SwishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    SwishPlugin* plugin = new SwishPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

PluginFieldCollection SwishPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SwishPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SwishPluginCreator);

} // namespace nvinfer1
