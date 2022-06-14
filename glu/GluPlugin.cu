#include <cub/cub.cuh>
#include <numeric>
#include "GluPlugin.h"

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

__global__ void glu_kernel(
    const int M,
    const int split_dim_size,
    const int N,
    const float* Xdata,
    float* Ydata) {
  const int xOffset = 2 * split_dim_size * N;
  const int yOffset = split_dim_size * N;
  CUDA_1D_KERNEL_LOOP(index, M * split_dim_size * N) {
    const int i = index / split_dim_size / N;
    const int j = index / N % split_dim_size;
    const int k = index % N;
    const float x1 = Xdata[i * xOffset + j * N + k];
    const float x2 = Xdata[i * xOffset + (j + split_dim_size) * N + k];
    Ydata[i * yOffset + j * N + k] = x1 * (1. / (1. + exp(-x2)));
  }
}

namespace nvinfer1
{

GluPlugin::GluPlugin(const std::string &name)
{

}


GluPlugin::GluPlugin(const void* data, size_t length)
{

}

int GluPlugin::initialize() noexcept
{
    return 0;
}

const char* GluPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* GluPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int GluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs GluPlugin::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output(inputs[0]);
    auto input_dimsexprs = inputs[0];
    output.d[1] = exprBuilder.constant(int(input_dimsexprs.d[1]->getConstantValue() / 2));
    
    return output;
}

void GluPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void GluPlugin::detachFromContext() noexcept
{
}

int GluPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
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
    int M = batchSize;
    int split_dim_size = nbChannels / 2;
    int N = normDim;

    glu_kernel<<<CAFFE_GET_BLOCKS(M * N * split_dim_size), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(M, split_dim_size, N, static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]));

    return 0;
}

size_t GluPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void GluPlugin::serialize(void* buffer) const noexcept
{
}

bool GluPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

void GluPlugin::terminate() noexcept
{
}

void GluPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* GluPlugin::clone() const noexcept
{
    auto* plugin = new GluPlugin(mNamespace);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void GluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

nvinfer1::DataType GluPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t GluPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void GluPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* GluPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

GluPluginCreator::GluPluginCreator()
{
}

const char* GluPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* GluPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* GluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* GluPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GluPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* GluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    GluPlugin* plugin = new GluPlugin(name);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* GluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    GluPlugin* plugin = new GluPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

PluginFieldCollection GluPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GluPluginCreator);

} // namespace nvinfer1
