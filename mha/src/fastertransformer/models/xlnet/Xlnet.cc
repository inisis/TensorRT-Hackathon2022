/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/xlnet/Xlnet.h"

namespace fastertransformer {

template<typename T>
void Xlnet<T>::initialize()
{
    allocateBuffer();
    attention_layer_ = new XlnetAttentionLayer<T>(max_batch_size_,
                                                  max_seq_len_,
                                                  head_num_,
                                                  size_per_head_,
                                                  q_scaling_,
                                                  stream_,
                                                  cublas_wrapper_,
                                                  allocator_,
                                                  is_free_buffer_after_forward_);
}

template<typename T>
Xlnet<T>::Xlnet(size_t max_batch_size,
                size_t max_seq_len,
                size_t head_num,
                size_t size_per_head,
                size_t inter_size,
                size_t num_layer,
                float q_scaling,
                cudaStream_t stream,
                cublasMMWrapper* cublas_wrapper,
                IAllocator* allocator,
                bool is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    q_scaling_(q_scaling)
{
    initialize();
}

template<typename T>
Xlnet<T>::Xlnet(Xlnet<T> const& xlnet):
    BaseLayer(xlnet),
    max_batch_size_(xlnet.max_batch_size_),
    max_seq_len_(xlnet.max_seq_len_),
    head_num_(xlnet.head_num_),
    size_per_head_(xlnet.size_per_head_),
    inter_size_(xlnet.inter_size_),
    hidden_units_(xlnet.hidden_units_),
    num_layer_(xlnet.num_layer_),
    q_scaling_(xlnet.q_scaling_)
{
    initialize();
}

template<typename T>
Xlnet<T>::~Xlnet()
{
    delete attention_layer_;
    freeBuffer();
}

template<typename T>
void Xlnet<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        attn_mask_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);
        seg_mat_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * 2, false);
        attr_k_head_r_ = (T*)allocator_->malloc(sizeof(T) * max_seq_len_ * hidden_units_ * 2, false);
        attn_out_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        output_fc2_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void Xlnet<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(attn_mask_);
        allocator_->free(seg_mat_);
        allocator_->free(attr_k_head_r_);
        allocator_->free(output_fc2_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void Xlnet<T>::forward(std::vector<Tensor>* output_tensors,
                       const std::vector<Tensor>* input_tensors,
                       const T* xlnet_layer_weights)
{
    // input_tensors:
    // input_query [batch_size_, seq_len_, hidden_units_],
    // input_mask [batch,seq_len],
    // seg_id [batch,seq_len]

    // output_tensors:
    // out_tensor [batch_size, seq_len, hidden_units]

    attention_layer_->forward(
        output_tensors, input_tensors, xlnet_layer_weights);

}

template<typename T>
bool Xlnet<T>::isValidBatchSize(size_t batch_size)
{
    if (max_batch_size_ == 0) {
        max_batch_size_ = batch_size;
        return true;
    }
    else {
        return batch_size <= max_batch_size_;
    }
}

template<typename T>
bool Xlnet<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template class Xlnet<float>;
template class Xlnet<half>;

}  // namespace fastertransformer
