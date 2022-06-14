#ifndef TRT_LAYER_NORM_PLUGIN_H
#define TRT_LAYER_NORM_PLUGIN_H

#include <NvInfer.h>
#include <cudnn.h>
#include <vector>
#include <iostream>
#include <string>
#include "src/fastertransformer/models/xlnet/Xlnet.h"

namespace
{
static const char *PLUGIN_NAME {"MHAPlugin"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

namespace nvinfer1
{
class MHAPlugin : public IPluginV2DynamicExt
{
public:
    MHAPlugin(const std::string &name);

    MHAPlugin(const void* data, size_t length);

    // It doesn't make sense to make MHAPlugin without arguments, so we
    // delete default constructor.
    MHAPlugin() = delete;

    int getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

private:
    const char* mPluginNamespace;
    std::string mNamespace;

    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;
    
    fastertransformer::cublasAlgoMap* cublas_algo_map_;
    fastertransformer::Allocator<fastertransformer::AllocatorType::CUDA>* allocator_;
    std::mutex* cublas_wrapper_mutex_;
    fastertransformer::cublasMMWrapper* cublas_wrapper_;
    fastertransformer::Xlnet<float>* xlnet_;
    float * weight_ptr;
    float * input_ptr;
    float * output_ptr;
};

class MHAPluginCreator : public IPluginCreator
{
public:
    MHAPluginCreator();

    ~MHAPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace nvinfer1

#endif // TRT_LAYER_NORM_PLUGIN_H
