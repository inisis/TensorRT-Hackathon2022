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

#include "src/fastertransformer/layers/xlnet_attention_layers/XlnetAttentionLayer.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/xlnet_preprocess_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

template<typename T>
void XlnetAttentionLayer<T>::forward(std::vector<fastertransformer::Tensor>* output_tensors,
                                     const std::vector<fastertransformer::Tensor>* input_tensors,
                                     const T* attention_weights)
{
    const size_t request_batch_size = input_tensors->at(0).shape[0];
    const size_t request_seq_len = input_tensors->at(0).shape[1];

    FT_CHECK(input_tensors->at(0).shape.size() == 3);

    T* out_tensor = (T*)output_tensors->at(0).data;
    T* in_tensor = (T*)input_tensors->at(0).data;
    
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          hidden_units_,
                          max_batch_size_ * max_seq_len_,
                          hidden_units_,
                          input_tensors->at(4).data, // q weight
                          hidden_units_,
                          in_tensor,
                          hidden_units_,
                          input_tensors->at(5).data, // q bias
                          query_buf_,
                          hidden_units_);

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          hidden_units_,
                          max_batch_size_ * max_seq_len_,
                          hidden_units_,
                          input_tensors->at(6).data, // k weight
                          hidden_units_,
                          in_tensor,
                          hidden_units_,
                          input_tensors->at(7).data, // k bias
                          key_buf_,
                          hidden_units_);

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          hidden_units_,
                          max_batch_size_ * max_seq_len_,
                          hidden_units_,
                          input_tensors->at(8).data,  // v weight
                          hidden_units_,
                          in_tensor,
                          hidden_units_,
                          input_tensors->at(9).data,  // v bias
                          value_buf_,
                          hidden_units_);
    
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          hidden_units_,
                          max_seq_len_,
                          hidden_units_,
                          input_tensors->at(3).data,  // pos_emb weight
                          hidden_units_,
                          input_tensors->at(2).data,  // pos_emb input
                          hidden_units_,
			  pos_emb_bias_,
                          pos_emb_buff_,
                          hidden_units_);
   
    cudaD2Dcpy(query_buf_v_, query_buf_, request_batch_size * request_seq_len * hidden_units_);

    invokeAddBias((float *)query_buf_,
                 (const float *)input_tensors->at(10).data,
                 request_batch_size * request_seq_len,
                 hidden_units_,
                 stream_);

    invokeAddBias((float *)query_buf_v_,
                 (const float *)input_tensors->at(11).data,
                 request_batch_size * request_seq_len,
                 hidden_units_,
                 stream_);

    invokeTranspose102v2(request_batch_size, request_seq_len, 4, 64, (float *)q_buf_t_, (float *)query_buf_, stream_); // u batch, head, time, d_model
    invokeTranspose102v2(request_batch_size, request_seq_len, 4, 64, (float *)q_buf_v_t_, (float *)query_buf_v_, stream_); // v batch, head, time, d_model
    invokeTranspose102v2(request_batch_size, request_seq_len, 4, 64, (float *)k_buf_t_, (float *)key_buf_, stream_); // v batch, head, time, d_model
    invokeTranspose102v2(1, request_seq_len, 4, 64, (float *)pos_emb_buff_t_, (float *)pos_emb_buff_, stream_); // v 1, head, time, d_model

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        request_seq_len,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_t_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        q_buf_t_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        request_batch_size * head_num_); // qk matmul pass
    
    T * temp_ptr = pos_emb_buff_expand_;
    for(int i = 0; i < request_batch_size; ++i)
    {
        cudaD2Dcpy(temp_ptr, pos_emb_buff_t_, 1 * request_seq_len * hidden_units_);
	temp_ptr += request_seq_len * hidden_units_;
    }

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        request_seq_len,
                                        request_seq_len,
                                        size_per_head_,
                                        pos_emb_buff_expand_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        q_buf_v_t_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qp_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        request_batch_size * head_num_); //  qp matmul
    
    invokeCalAttnScore(request_batch_size,
                       head_num_,
                       request_seq_len,
                       size_per_head_,
                       q_scaling_,
                       (float *)attn_score_,
                       (float *)qk_buf_,
                       (float *)qp_buf_,
                       (float *)input_tensors->at(1).data,
                       (float *)value_buf_trans_,
                       (float *)value_buf_,
                       stream_);    

    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        request_seq_len,
                                        request_seq_len,
                                        value_buf_trans_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        attn_score_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        attn_vec_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        request_batch_size * head_num_);

    invokeTranspose102(request_batch_size, request_seq_len, 4, 64, (float *)attn_vec_trans_, (float *)attn_vec_, stream_); // v batch, time, head, d_model

    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          hidden_units_,
                          request_batch_size * request_seq_len,
                          hidden_units_,
                          input_tensors->at(12).data, // fc weight
                          hidden_units_,
                          attn_vec_trans_,
                          hidden_units_,
                          input_tensors->at(13).data, // fc bias
                          attn_out_,
                          hidden_units_);

    cudaD2Dcpy((float *)out_tensor, (float *)attn_out_, request_batch_size * request_seq_len * hidden_units_);
}

template<typename T>
XlnetAttentionLayer<T>::XlnetAttentionLayer(size_t max_batch_size,
                                            size_t max_seq_len,
                                            size_t head_num,
                                            size_t size_per_head,
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
    q_scaling_(q_scaling)
{
    hidden_units_ = head_num_ * size_per_head_;
    allocateBuffer();
}

template<typename T>
void XlnetAttentionLayer<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        pos_emb_buff_ = (T*)allocator_->malloc(sizeof(T) * 1 * max_seq_len_ * hidden_units_, false);
	pos_emb_bias_ = (T*)allocator_->malloc(sizeof(T) * hidden_units_, true);

        query_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_ * 3, false);
        key_buf_ = query_buf_ + max_batch_size_ * max_seq_len_ * hidden_units_;
        value_buf_ = query_buf_ + 2 * max_batch_size_ * max_seq_len_ * hidden_units_;

	query_buf_v_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        
	q_buf_t_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
	q_buf_v_t_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);

        k_buf_t_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        pos_emb_buff_t_ = (T*)allocator_->malloc(sizeof(T) * 1 * max_seq_len_ * hidden_units_, false);


        qk_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        qp_buf_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        value_buf_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);

	attn_mask_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_, false);
	pos_emb_buff_expand_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, true);
        
        v_buf_t_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);



	q_buf_bd_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_bd_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * 2 * hidden_units_, false);
        qk_buf_bd_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * max_seq_len_ * 2, false);
        qk_buf_bd_shift_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * max_seq_len_, false);
        q_buf_ef_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        k_buf_ef_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * hidden_units_ * 2, false);
        qk_buf_ef_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * 2, false);
        qk_buf_ef_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * head_num_ * 2, false);
        qk_buf_ef_seg_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        qk_buf_ef_seg_trans_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        attn_score_ =
            (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * max_seq_len_ * head_num_, false);
        value_buf_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_vec_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_vec_trans_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);
        attn_out_ = (T*)allocator_->malloc(sizeof(T) * max_batch_size_ * max_seq_len_ * hidden_units_, false);

        is_allocate_buffer_ = true;
    }
}

template<typename T>
bool XlnetAttentionLayer<T>::isValidBatchSize(size_t batch_size)
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
bool XlnetAttentionLayer<T>::isValidSeqLen(size_t seq_len)
{
    if (max_seq_len_ == 0) {
        max_seq_len_ = seq_len;
        return true;
    }
    else {
        return seq_len <= max_seq_len_;
    }
}

template<typename T>
void XlnetAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_ == true) {
        allocator_->free(pos_emb_buff_);
        allocator_->free(pos_emb_bias_);
        allocator_->free(query_buf_);
        allocator_->free(q_buf_t_);
        allocator_->free(k_buf_t_);
        allocator_->free(qk_buf_);
        allocator_->free(q_buf_bd_);
        allocator_->free(k_buf_bd_);
        allocator_->free(qk_buf_bd_);
        allocator_->free(qk_buf_bd_shift_);
        allocator_->free(q_buf_ef_);
        allocator_->free(k_buf_ef_);
        allocator_->free(qk_buf_ef_);
        allocator_->free(qk_buf_ef_trans_);
        allocator_->free(qk_buf_ef_seg_);
        allocator_->free(qk_buf_ef_seg_trans_);
        allocator_->free(attn_score_);
        allocator_->free(value_buf_trans_);
        allocator_->free(attn_vec_);
        allocator_->free(attn_vec_trans_);
        allocator_->free(attn_out_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
XlnetAttentionLayer<T>::~XlnetAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template class XlnetAttentionLayer<float>;
template class XlnetAttentionLayer<half>;

}  // namespace fastertransformer
