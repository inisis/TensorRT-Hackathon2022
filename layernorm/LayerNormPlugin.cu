#include <cub/cub.cuh>
#include <numeric>
#include "LayerNormPlugin.h"
#include "serialize.hpp"

#define FINAL_MASK 0xffffffff

template<typename T>
inline __device__ T hmul2(T a, T b) {
    return __hmul2(a, b);
}

template<typename T>
inline __device__ T hsub2(T a, T b) {
    return __hsub2(a, b);
}

template<typename T>
inline __device__ T hadd2(T a, T b) {
    return __hadd2(a, b);
}

template<typename T>
struct TypeConverter {using Type = half2;}; // keep for generality

template<>
struct TypeConverter<half2> {using Type = half;};

template<>
struct TypeConverter<half> {using Type = half2;};

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

template<typename T>
inline __device__ T float2type(float a);

template<>
inline __device__ half float2type(float a) {
    return __float2half_rn(a);
}

template<typename T>
inline __device__ T float2type2(float a);

template<>
inline __device__ half2 float2type2(float a) {
    return __float2half2_rn(a);
}

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    }
    return val;
}

template<typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T)(0.0f);
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T)0.0f;
}

template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt(T* normed_output,
                                                   T* output,
                                                   const T* __restrict bias,
                                                   const T* __restrict residual,
                                                   const T* __restrict gamma,
                                                   const T* __restrict beta,
                                                   int m,
                                                   int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    T local_sum = float2type2<T>(0.0f);
#pragma unroll
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = float2type2<T>(0.0f);

        if (IS_BIAS) {
            val = hadd2(val, ldg(&bias[i]));
        }
        if (IS_RESIDUAL) {
            val = hadd2(val, ldg(&residual[index]));
        }

        if (IS_OUTPUT) {
            val = hadd2(val, output[index]);
        }
        output[index] = val;
        local_sum = hadd2(local_sum, val);
    }

    mean = blockReduceSum((float)(local_sum.x + local_sum.y));

    if (threadIdx.x == 0) {
        s_mean = mean / n / 2;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T val = output[blockIdx.x * n + i];
        float diff_1 = (float)(val.x) - s_mean;
        float diff_2 = (float)(val.y) - s_mean;
        local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n / 2 + 1e-6f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);
#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = blockIdx.x * n + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

// * Note that typename T is half2 or bfloat2 type
template<typename T, bool IS_OUTPUT, bool IS_BIAS, bool IS_RESIDUAL, bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(T* normed_output,
                                                    T* output,
                                                    const T* __restrict bias,
                                                    const T* __restrict residual,
                                                    const T* __restrict gamma,
                                                    const T* __restrict beta,
                                                    int m,
                                                    int n)
{
    __shared__ float s_mean;
    __shared__ float s_variance;
    float x_sum = 0.0f;
    float x2_sum = 0.0f;
    const int b_offset = blockIdx.x * n;
    using T1 = typename TypeConverter<T>::Type;

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        float val_1 = 0.0f;
        float val_2 = 0.0f;
        T tmp;

        if (IS_BIAS) {
            tmp = ldg(&bias[i]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        if (IS_RESIDUAL) {
            tmp = ldg(&residual[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }

        if (IS_OUTPUT) {
            tmp = ldg(&output[index]);
            val_1 += static_cast<float>(tmp.x);
            val_2 += static_cast<float>(tmp.y);
        }
        tmp.x = float2type<T1>(val_1);
        tmp.y = float2type<T1>(val_2);
        output[index] = tmp;
        x_sum += val_1 + val_2;
        x2_sum += val_1 * val_1 + val_2 * val_2;
    }
    float sums[2];
    sums[0] = x_sum;
    sums[1] = x2_sum;
    blockReduceSumV2<float, 2>(sums);

    if (threadIdx.x == 0) {
        s_mean = sums[0] / n / 2;
        s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + 1e-6f);
    }
    __syncthreads();

    T mean_2 = float2type2<T>(s_mean);
    T var_2 = float2type2<T>(s_variance);

#pragma unroll UNROLL_FACTOR
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int index = b_offset + i;
        T val = hmul2(hmul2(hsub2(output[index], mean_2), var_2), ldg(&gamma[i]));
        if (IS_BETA) {
            val = hadd2(val, ldg(&beta[i]));
        }
        normed_output[index] = val;
    }
}

template<typename T>
__global__ void generalLayerNorm(
    const T* __restrict input, const T* __restrict gamma, const T* __restrict beta, T* output, int m, int n)
{
    const int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        output[blockIdx.x * n + i] =
            (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}

#define HALF_LAYERNORM_OPT(UNROLL_FACTOR)                                                                              \
    generalAddBiasResidualLayerNormOpt<T2, false, false, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(       \
        (T2*)out, (T2*)out, nullptr, (const T2*)input, (const T2*)gamma, (const T2*)beta, m, half_n);

#define HALF_LAYERNORM_OPT2(UNROLL_FACTOR)                                                                             \
    generalAddBiasResidualLayerNormOpt2<T2, false, false, true, true, UNROLL_FACTOR><<<grid, block, 0, stream>>>(      \
        (T2*)out, (T2*)out, nullptr, (const T2*)input, (const T2*)gamma, (const T2*)beta, m, half_n);

template<typename T>
void invokeGeneralLayerNorm(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            cudaStream_t stream,
                            int opt_version)
{
    dim3 grid(m);
    if (n % 2 == 0 && std::is_same<T, half>::value && opt_version > 0) {
        int half_n = n / 2;
        int half_n_32 = (half_n + 31) / 32 * 32;
        dim3 block(min(half_n_32, 512));
        int rolls_per_thread = half_n / block.x;
        int unroll_factor = 8;
        while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
            unroll_factor /= 2;
        }
        using T2 = typename TypeConverter<T>::Type;
        if (opt_version == 1) {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT(8);
            }
        }
        else {
            if (unroll_factor == 1) {
                HALF_LAYERNORM_OPT2(1);
            }
            else if (unroll_factor == 2) {
                HALF_LAYERNORM_OPT2(2);
            }
            else if (unroll_factor == 3) {
                HALF_LAYERNORM_OPT2(3);
            }
            else if (unroll_factor == 4) {
                HALF_LAYERNORM_OPT2(4);
            }
            else if (unroll_factor == 8) {
                HALF_LAYERNORM_OPT2(8);
            }
        }
    }
    else {
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
            Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */
        if (n % 32 != 0) {
            block.x = 1024;
        }

        /* should pay attention to the rsqrt precision*/
        generalLayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, beta, out, m, n);  // For gpt-3
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

namespace nvinfer1
{
LayerNormPlugin::LayerNormPlugin(float epsilon, int nbGroups)
    : mEpsilon(epsilon),
      mNbGroups(nbGroups)
{
    // Number of groups should be positive
    assert(nbGroups > 0);
}

int LayerNormPlugin::initialize() noexcept
{
    mInitialized = true;
    return 0;
}

LayerNormPlugin::LayerNormPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mEpsilon);
    deserialize_value(&data, &length, &mNbGroups);
}

const char* LayerNormPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* LayerNormPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int LayerNormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs LayerNormPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Input (from previous layer), scale and bias are the three inputs to the plugin.
    assert(nbInputs == 3);
    assert(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

void LayerNormPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    _cudnn_handle = cudnnContext;
    cudnnCreateTensorDescriptor(&desc);
    cudnnCreateTensorDescriptor(&bnDesc);
}

// Detach the plugin object from its execution context.
void LayerNormPlugin::detachFromContext() noexcept
{
    //cudnnDestroyTensorDescriptor(desc);
    //cudnnDestroyTensorDescriptor(bnDesc);
}

int LayerNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
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
 
    // LayerNormForwardGpu<float, true, true>(stream, M, N, mEpsilon, static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]), static_cast<const float*>(inputs[2]), static_cast<float*>(normalized), static_cast<float*>(outputs[0]), static_cast<float*>(bnScale), static_cast<float*>(bnBias));
    invokeGeneralLayerNorm<half>(static_cast<half*>(outputs[0]), static_cast<const half*>(inputs[0]), static_cast<const half*>(inputs[1]), static_cast<const half*>(inputs[2]), M, N, stream, 1);
    return 0;
}

size_t LayerNormPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNbGroups) + sizeof(mEpsilon);
}

void LayerNormPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
}

bool LayerNormPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return ((inOut[pos].type == nvinfer1::DataType::kHALF) && (inOut[pos].format == nvinfer1::PluginFormat::kLINEAR));
}

void LayerNormPlugin::terminate() noexcept
{
    if (mInitialized)
    {
        // cudnnDestroyTensorDescriptor(desc);
        // cudnnDestroyTensorDescriptor(bnDesc);
        // cudnnDestroy(_cudnn_handle);	
	// cudaFree(bnScale);
        // cudaFree(bnBias);
    }
    mInitialized = false;
}

void LayerNormPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* LayerNormPlugin::clone() const noexcept
{
    auto* plugin = new LayerNormPlugin(mEpsilon, mNbGroups);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void LayerNormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
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
    int normDim = in[0].desc.dims.d[2];

    if(batchSize == -1)
        batchSize = 1;
    if(nbChannels == -1)
        nbChannels = 1;

    // Allocate device memory and initialize scale and bias values
    cudaMalloc(&bnScale, batchSize * nbChannels * sizeof(float));
    cudaMalloc(&bnBias, batchSize * nbChannels* sizeof(float));
    cudaMalloc(&normalized, batchSize * nbChannels * normDim * sizeof(float));
}

nvinfer1::DataType LayerNormPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t LayerNormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void LayerNormPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* LayerNormPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

LayerNormPluginCreator::LayerNormPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LayerNormPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* LayerNormPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* LayerNormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* LayerNormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LayerNormPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* LayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // Set default values
    int nbGroups{1};
    float epsilon{0.00001F};
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("eps") == 0)
        {
            epsilon = *static_cast<const float*>(fc->fields[i].data);
        }
        if (field_name.compare("num_groups") == 0)
        {
            nbGroups = *static_cast<const int*>(fc->fields[i].data);
        }
    }

    LayerNormPlugin* plugin = new LayerNormPlugin(epsilon, nbGroups);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* LayerNormPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    LayerNormPlugin* plugin = new LayerNormPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

PluginFieldCollection LayerNormPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LayerNormPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

} // namespace nvinfer1
