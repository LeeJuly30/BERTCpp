#include "batchgemm.h"
#include "mkl.h"

namespace lh{
    template<>
    void attn_qk<float>(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float* query, float* key, float* output) {
        const float** q_array = new const float *[batch_size * num_heads];
        const float** k_array = new const float *[batch_size * num_heads];
        float** out_array = new float *[batch_size * num_heads];
        for (std::size_t idx = 0; idx < batch_size; idx++)
        {
            for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++)
            {
                q_array[idx * num_heads + head_idx] = query + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                k_array[idx * num_heads + head_idx] = key + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                out_array[idx * num_heads + head_idx] = output + idx * num_heads * seq_len * seq_len + head_idx * seq_len;
            }
        }
        CBLAS_TRANSPOSE tranA = CblasNoTrans;
        CBLAS_TRANSPOSE tranB = CblasTrans;
        const long long int m = seq_len, n = seq_len, k = hidden_size, lda_array = hidden_size * num_heads, ldb_array = hidden_size * num_heads, ldc_array = seq_len * num_heads, group_size = batch_size * num_heads;
        const float alpha = 1.0, beta = 0.0;
        cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, q_array, &lda_array, k_array, &ldb_array, &beta, out_array, &ldc_array, 1, &group_size);
        
        delete [] q_array;
        delete [] k_array;
        delete [] out_array;
    }

    template<>
    void attn_sv<float>(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, float* sim, float* value, float* output){
        const float** sim_array = new const float *[batch_size * num_heads];
        const float** value_array = new const float *[batch_size * num_heads];
        float** out_array = new float *[batch_size * num_heads];
        for (std::size_t idx = 0; idx < batch_size; idx++)
        {
            for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++)
            {
                sim_array[idx * num_heads + head_idx] = sim + idx * num_heads * seq_len * seq_len + head_idx * seq_len;
                value_array[idx * num_heads + head_idx] = value + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
                out_array[idx * num_heads + head_idx] = output + idx * num_heads * seq_len * hidden_size + head_idx * hidden_size;
            }
        }
        CBLAS_TRANSPOSE tranA = CblasNoTrans;
        CBLAS_TRANSPOSE tranB = CblasNoTrans;
        const long long int m = seq_len, n = hidden_size, k = seq_len, lda_array = seq_len * num_heads, ldb_array = hidden_size * num_heads, ldc_array = hidden_size * num_heads, group_size = batch_size * num_heads;
        const float alpha = 1.0, beta = 0.0;
        cblas_sgemm_batch(CblasRowMajor, &tranA, &tranB, &m, &n, &k, &alpha, sim_array, &lda_array, value_array, &ldb_array, &beta, out_array, &ldc_array, 1, &group_size);
        
        delete [] sim_array;
        delete [] value_array;
        delete [] out_array;
    }
}