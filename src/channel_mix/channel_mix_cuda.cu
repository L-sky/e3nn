#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ constexpr uint32_t threads_per_block_forward_child_cuda_kernel()    { return 256; }
__device__ constexpr uint32_t threads_per_block_backward_F_child_cuda_kernel() { return 256; }
__device__ constexpr uint32_t threads_per_block_backward_W_child_cuda_kernel() { return 32; }


// Declarations of child kernels
template<typename T>
__global__ void forward_child_cuda_kernel(
              T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T,
        const uint32_t,
        const uint32_t,
        const uint32_t,
        const uint32_t
);

template<typename T>
__global__ void backward_F_child_cuda_kernel(
              T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T,
        const uint32_t,
        const uint32_t,
        const uint32_t,
        const uint32_t
);

template<typename T>
__global__ void backward_W_child_cuda_kernel(
              T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T,
        const uint32_t,
        const uint32_t,
        const uint32_t,
        const uint32_t
);


// Implementations
template<typename T>
__global__ void forward_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ W,
        const T*        const __restrict__ F,
        const T*        const __restrict__ norm_list,
        const uint32_t* const __restrict__ L_list,
        const uint32_t* const __restrict__ v_sizes,
        const uint32_t* const __restrict__ w_sizes,
        const uint32_t                     a_size
) {
    const uint32_t L_id = blockIdx.x;
    const uint32_t L = L_list[L_id];
    const uint32_t m_size = 2 * L + 1;
    const uint32_t v_size = v_sizes[L_id];
    const uint32_t w_size = w_sizes[L_id];

    // blocks in grid, threads in block
    const uint32_t threads_per_block = threads_per_block_forward_child_cuda_kernel();
    const uint32_t a_blocks = (a_size + threads_per_block - 1) / threads_per_block;
    dim3 blocks(a_blocks, w_size, m_size);

    // calculate offsets
    uint32_t output_offset = 0, W_offset = 0, F_offset = 0;
    for (uint32_t idx = 0; idx < L_id; idx++){
        output_offset += w_sizes[idx] * (2 * L_list[idx] + 1) * a_size;
        W_offset += w_sizes[idx] * v_sizes[idx];
        F_offset += v_sizes[idx] * (2 * L_list[idx] + 1) * a_size;
    }

    // shift pointers by offsets
          T* const __restrict__ output_l = output + output_offset;
    const T* const __restrict__ W_l      = W + W_offset;
    const T* const __restrict__ F_l      = F + F_offset;

    const T norm_l = norm_list[L_id];

    forward_child_cuda_kernel<T><<<blocks, threads_per_block>>>(output_l, W_l, F_l, norm_l,
                                                                m_size, v_size, w_size, a_size);
}


template<typename T>
__global__ void forward_child_cuda_kernel(
              T*        const __restrict__ output_l,
        const T*        const __restrict__ W_l,
        const T*        const __restrict__ F_l,
        const T                            norm_l,
        const uint32_t                     m_size,
        const uint32_t                     v_size,
        const uint32_t                     w_size,
        const uint32_t                     a_size
) {
    const uint32_t w = blockIdx.y;
    const uint32_t m = blockIdx.z;

    const uint32_t a = threadIdx.x + blockIdx.x * blockDim.x;

    // last block can be incompletely filled - terminate early
    if (a >= a_size) return;

    // shift pointer by offset
    const T* const __restrict__ W_lw = W_l + (w * v_size);

    // function main body
    T output_tmp = 0;
    for (uint32_t v = 0; v < v_size; v++){
       output_tmp += W_lw[v] * F_l[v * m_size * a_size + m * a_size + a];
    }

    // write result to the global memory
    output_l[w * m_size * a_size + m * a_size + a] = norm_l * output_tmp;
}


template<typename T>
__global__ void backward_F_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ W,
        const T*        const __restrict__ G,
        const T*        const __restrict__ norm_list,
        const uint32_t* const __restrict__ L_list,
        const uint32_t* const __restrict__ v_sizes,
        const uint32_t* const __restrict__ w_sizes,
        const uint32_t                     a_size
) {
    const uint32_t L_id = blockIdx.x;
    const uint32_t L = L_list[L_id];
    const uint32_t m_size = 2 * L + 1;
    const uint32_t v_size = v_sizes[L_id];
    const uint32_t w_size = w_sizes[L_id];

    // blocks in grid, threads in block
    const uint32_t threads_per_block = threads_per_block_backward_F_child_cuda_kernel();
    const uint32_t a_blocks = (a_size + threads_per_block - 1) / threads_per_block;
    dim3 blocks(a_blocks, v_size, m_size);

    // calculate offsets
    uint32_t output_offset = 0, W_offset = 0, G_offset = 0;
    for (uint32_t idx = 0; idx < L_id; idx++){
        output_offset += v_sizes[idx] * (2 * L_list[idx] + 1) * a_size;
        W_offset += w_sizes[idx] * v_sizes[idx];
        G_offset += w_sizes[idx] * (2 * L_list[idx] + 1) * a_size;
    }

    // shift pointers by offsets
          T* const __restrict__ output_l = output + output_offset;
    const T* const __restrict__ W_l      = W + W_offset;
    const T* const __restrict__ G_l      = G + G_offset;

    const T norm_l = norm_list[L_id];

    backward_F_child_cuda_kernel<T><<<blocks, threads_per_block>>>(output_l, W_l, G_l, norm_l,
                                                                   m_size, v_size, w_size, a_size);
}


template<typename T>
__global__ void backward_F_child_cuda_kernel(
              T*        const __restrict__ output_l,
        const T*        const __restrict__ W_l,
        const T*        const __restrict__ G_l,
        const T                            norm_l,
        const uint32_t                     m_size,
        const uint32_t                     v_size,
        const uint32_t                     w_size,
        const uint32_t                     a_size
) {
    const uint32_t v = blockIdx.y;
    const uint32_t m = blockIdx.z;

    const uint32_t a = threadIdx.x + blockIdx.x * blockDim.x;

    // last block can be incompletely filled - terminate early
    if (a >= a_size) return;

    // function main body
    T output_tmp = 0;
    for (uint32_t w = 0; w < w_size; w++){
       output_tmp += W_l[w * v_size + v] * G_l[w * m_size * a_size + m * a_size + a];
    }

    // write result to the global memory
    output_l[v * m_size * a_size + m * a_size + a] = norm_l * output_tmp;
}


template<typename T>
__global__ void backward_W_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ G,
        const T*        const __restrict__ F,
        const T*        const __restrict__ norm_list,
        const uint32_t* const __restrict__ L_list,
        const uint32_t* const __restrict__ v_sizes,
        const uint32_t* const __restrict__ w_sizes,
        const uint32_t                     a_size
) {
    const uint32_t L_id = blockIdx.x;
    const uint32_t L = L_list[L_id];
    const uint32_t m_size = 2 * L + 1;
    const uint32_t v_size = v_sizes[L_id];
    const uint32_t w_size = w_sizes[L_id];

    // blocks in grid, threads in block
    const uint32_t threads_per_dim = threads_per_block_backward_W_child_cuda_kernel();
    dim3 threads_per_block(threads_per_dim, threads_per_dim);

    const uint32_t w_blocks = (w_size + threads_per_dim - 1) / threads_per_dim;
    const uint32_t v_blocks = (v_size + threads_per_dim - 1) / threads_per_dim;
    dim3 blocks(w_blocks, v_blocks);

    // calculate offsets
    uint32_t output_offset = 0, G_offset = 0, F_offset = 0;
    for (uint32_t idx = 0; idx < L_id; idx++){
        output_offset += w_sizes[idx] * v_sizes[idx];
        G_offset += w_sizes[idx] * (2 * L_list[idx] + 1) * a_size;
        F_offset += v_sizes[idx] * (2 * L_list[idx] + 1) * a_size;
    }

    // shift pointers by offsets
          T* const __restrict__ output_l = output + output_offset;
    const T* const __restrict__ G_l      = G + G_offset;
    const T* const __restrict__ F_l      = F + F_offset;

    const T norm_l = norm_list[L_id];

    backward_W_child_cuda_kernel<T><<<blocks, threads_per_block>>>(output_l, G_l, F_l, norm_l,
                                                                   m_size, v_size, w_size, a_size);

}


template<typename T>
__global__ void backward_W_child_cuda_kernel(
              T*        const __restrict__ output_l,
        const T*        const __restrict__ G_l,
        const T*        const __restrict__ F_l,
        const T                            norm_l,
        const uint32_t                     m_size,
        const uint32_t                     v_size,
        const uint32_t                     w_size,
        const uint32_t                     a_size
) {
    const uint32_t w = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t v = threadIdx.y + blockIdx.y * blockDim.y;

    // blocks on the grid edge can be partially filled - terminate early
    if (w >= w_size || v >= v_size) return;

    // function main body
    T output_tmp = 0;
    for (uint32_t a = 0; a < a_size; a++){
        for (uint32_t m = 0; m < m_size; m++){
            output_tmp += G_l[w * m_size * a_size + m * a_size + a] * F_l[v * m_size * a_size + m * a_size + a];
        }
    }

    // write result to the global memory
    output_l[w * v_size + v] = norm_l * output_tmp;
}


void forward_cuda(
        torch::Tensor output,
        torch::Tensor W,            // [k(L) (w, v)]
        torch::Tensor F,            // [k(L) (vm), a]
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes,
        torch::Tensor norm
) {
    const uint32_t a_size = F.size(1);
    const uint32_t blocks = L_list.size(0);

    const uint32_t* const __restrict__ L_list_ptr   = (uint32_t*) L_list.data_ptr();
    const uint32_t* const __restrict__ v_sizes_ptr  = (uint32_t*) v_sizes.data_ptr();
    const uint32_t* const __restrict__ w_sizes_ptr  = (uint32_t*) w_sizes.data_ptr();

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr = (double*) output.data_ptr();
        const double* const __restrict__ W_ptr      = (double*) W.data_ptr();
        const double* const __restrict__ F_ptr      = (double*) F.data_ptr();
        const double* const __restrict__ norm_ptr   = (double*) norm.data_ptr();

        forward_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, W_ptr, F_ptr, norm_ptr, L_list_ptr, v_sizes_ptr, w_sizes_ptr, a_size);
    }
    else if (output.dtype() == torch::kFloat32) {
              float* const __restrict__ output_ptr = (float*) output.data_ptr();
        const float* const __restrict__ W_ptr      = (float*) W.data_ptr();
        const float* const __restrict__ F_ptr      = (float*) F.data_ptr();
        const float* const __restrict__ norm_ptr   = (float*) norm.data_ptr();

        forward_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, W_ptr, F_ptr, norm_ptr, L_list_ptr, v_sizes_ptr, w_sizes_ptr, a_size);
    }
}


void backward_F_cuda(
        torch::Tensor output,
        torch::Tensor W,            // [k(L) (w, v)]
        torch::Tensor G,            // [k(L) (wm), a]
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes,
        torch::Tensor norm
) {
    const uint32_t a_size = G.size(1);
    const uint32_t blocks = L_list.size(0);

    const uint32_t* const __restrict__ L_list_ptr   = (uint32_t*) L_list.data_ptr();
    const uint32_t* const __restrict__ v_sizes_ptr  = (uint32_t*) v_sizes.data_ptr();
    const uint32_t* const __restrict__ w_sizes_ptr  = (uint32_t*) w_sizes.data_ptr();

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr = (double*) output.data_ptr();
        const double* const __restrict__ W_ptr      = (double*) W.data_ptr();
        const double* const __restrict__ G_ptr      = (double*) G.data_ptr();
        const double* const __restrict__ norm_ptr   = (double*) norm.data_ptr();

        backward_F_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, W_ptr, G_ptr, norm_ptr, L_list_ptr, v_sizes_ptr, w_sizes_ptr, a_size);
    }
    else if (output.dtype() == torch::kFloat32) {
              float* const __restrict__ output_ptr = (float*) output.data_ptr();
        const float* const __restrict__ W_ptr      = (float*) W.data_ptr();
        const float* const __restrict__ G_ptr      = (float*) G.data_ptr();
        const float* const __restrict__ norm_ptr   = (float*) norm.data_ptr();

        backward_F_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, W_ptr, G_ptr, norm_ptr, L_list_ptr, v_sizes_ptr, w_sizes_ptr, a_size);
    }
}


void backward_W_cuda(
        torch::Tensor output,
        torch::Tensor G,            // [k(L) (wm), a]
        torch::Tensor F,            // [k(L) (vm), a]
        torch::Tensor L_list,
        torch::Tensor v_sizes,
        torch::Tensor w_sizes,
        torch::Tensor norm
) {
    const uint32_t a_size = F.size(1);
    const uint32_t blocks = L_list.size(0);

    const uint32_t* const __restrict__ L_list_ptr   = (uint32_t*) L_list.data_ptr();
    const uint32_t* const __restrict__ v_sizes_ptr  = (uint32_t*) v_sizes.data_ptr();
    const uint32_t* const __restrict__ w_sizes_ptr  = (uint32_t*) w_sizes.data_ptr();

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr = (double*) output.data_ptr();
        const double* const __restrict__ G_ptr      = (double*) G.data_ptr();
        const double* const __restrict__ F_ptr      = (double*) F.data_ptr();
        const double* const __restrict__ norm_ptr   = (double*) norm.data_ptr();

        backward_W_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, G_ptr, F_ptr, norm_ptr, L_list_ptr, v_sizes_ptr, w_sizes_ptr, a_size);
    }
    else if (output.dtype() == torch::kFloat32) {
              float* const __restrict__ output_ptr = (float*) output.data_ptr();
        const float* const __restrict__ G_ptr      = (float*) G.data_ptr();
        const float* const __restrict__ F_ptr      = (float*) F.data_ptr();
        const float* const __restrict__ norm_ptr   = (float*) norm.data_ptr();

        backward_W_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, G_ptr, F_ptr, norm_ptr, L_list_ptr, v_sizes_ptr, w_sizes_ptr, a_size);
    }
}