#include <vector>
#include <musa_fp16.h>

#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(T *d_input, T* d_output, int cols, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ char smem[];
  T *sum_s = reinterpret_cast<T*>(smem);

  // grid-stride loop to compute partial sum of diagonal elements
  T sum = 0;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    sum += d_input[i * cols + i];
  }

  const int warpNum = blockDim.x / warpSize;
  const int warpId = threadIdx.x / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // In-warp reduction
  #pragma unroll
  for (int s = warpSize >> 1; s > 0; s >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, s);
  }

  if (laneId == 0) {
    sum_s[warpId] = sum;
  }

  __syncthreads();

  // block-level reduction
  if (warpId == 0) {
    sum = (laneId < warpNum) ? sum_s[laneId] : 0;
    
    #pragma unroll
    for (int s = warpNum >> 1; s > 0; s >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, s);
    }

    if (laneId == 0) {
      atomicAdd(d_output, sum);
    }
  }
}

const int MAX_HEAD_DIM = 64;  // Max supported head dimension for register arrays
const int MAX_Br = 64;        // Max block size for score array

// XOR-based swizzling to reduce bank conflicts in shared memory access
__device__ __forceinline__ int swizzle(int row, int col, int width) {
  return row * width + ((col ^ row) & (width - 1));
}

// Q, O:  [batch_size, N, query_heads, head_dim]
// K, V:  [batch_size, M, kv_heads,    head_dim]
// [batch_id, seq_id, head_id, val_id]
template <typename T>
__global__ void flashAttention_kernel(T* Q, T* K, T* V, T* O, int N, int M, int q_head, int kv_head, int head_dim,
                                      bool is_causal, float scale_fac, int Br, int Bc) {
  const int tid = threadIdx.x;

  const int batch_id = blockIdx.y / q_head;
  const int q_head_id = blockIdx.y % q_head;
  const int kv_head_id = q_head_id / (q_head / kv_head); // GQA support
  const int q_row_id = blockIdx.x * Br + tid;

  // [batch_id, q_row_id, q_head_id, :]
  const int q_offset = batch_id * N * q_head * head_dim + q_row_id * q_head * head_dim + q_head_id * head_dim;

  extern __shared__ char _smem[];
  T *smem = reinterpret_cast<T*>(_smem);

  T *Q_s = smem;
  T *K_s = smem + Br * head_dim;
  T *V_s = smem + Br * head_dim + Bc * head_dim;
  // Shared memory layout: [Q_tile: Br×d | K_tile: Bc×d | V_tile: Bc×d]

  float Oi[MAX_HEAD_DIM] = {0.0f};
  float m_pre = -INFINITY;
  float l = 0.0f;

  for (int d = 0; d < head_dim; ++d) {
    if (q_row_id < N) {
      Q_s[swizzle(tid, d, head_dim)] = Q[q_offset + d];
    } else {
      Q_s[swizzle(tid, d, head_dim)] = T(0.0f);
    }
  }
  __syncthreads();

  // For causal masking: compute max K/V tiles needed based on block's last Q row
  // All threads in block use same tile count to maintain synchronization
  const int block_last_row = blockIdx.x * Br + Br - 1;
  const int kv_tile_num_causal = (min(block_last_row, N - 1) / Bc) + 1;
  const int kv_tile_num_full = (M + Bc - 1) / Bc;
  const int kv_tile_num = is_causal ? min(kv_tile_num_causal, kv_tile_num_full) : kv_tile_num_full;
  
  for (int i = 0; i < kv_tile_num; ++i) {
    const int tile_start = i * Bc;
    const int kv_offset = batch_id * M * kv_head * head_dim + (tile_start + tid) * kv_head * head_dim + kv_head_id * head_dim;

    for (int d = 0; d < head_dim; ++d) {
      if (tile_start + tid < M) {
        K_s[swizzle(tid, d, head_dim)] = K[kv_offset + d];
        V_s[swizzle(tid, d, head_dim)] = V[kv_offset + d];
      } else {
        K_s[swizzle(tid, d, head_dim)] = V_s[swizzle(tid, d, head_dim)] = T(0.0f);
      }
    }
    __syncthreads();
    
    float scores[MAX_Br], m = m_pre;
    
    // Causal mask: only compute attention where key_pos <= query_pos
    const int j_limit_causal = q_row_id - tile_start + 1;  // col_id = i * Bc + j <= q_row_id (for casual)
    const int j_limit_M = M - tile_start;                  // col_id = i * Bc + j < M
    const int j_limit = is_causal ? min(min(j_limit_causal, j_limit_M), Bc) : min(j_limit_M, Bc);

    for (int j = 0; j < j_limit; ++j) {
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        score += float(Q_s[swizzle(tid, d, head_dim)]) * float(K_s[swizzle(j, d, head_dim)]);
      }
      scores[j] = score * scale_fac;
      m = fmaxf(m, scores[j]);
    }
    
    for (int j = j_limit; j < Bc; ++j) {
      scores[j] = -INFINITY;
    }

    // Re-scale l and Oi
    const float alpha = expf(m_pre - m);
    l *= alpha;
    for (int d = 0; d < head_dim; ++d) {
      Oi[d] *= alpha;
    }

    for (int j = 0; j < j_limit; ++j) {
      float p = expf(scores[j] - m);
      l += p;
      for (int d = 0; d < head_dim; ++d) {
        Oi[d] += p * float(V_s[swizzle(j, d, head_dim)]);
      }
    }
    __syncthreads();
    
    m_pre = m;
  }

  if (q_row_id < N) {
    for (int d = 0; d < head_dim; ++d) {
      O[q_offset + d] = T(Oi[d] / l);
    }
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function

  const int BLOCK_SIZE = 512;

  T* d_input = nullptr, *d_output = nullptr;
  
  musaMalloc((void **)&d_input, sizeof(T) * rows * cols);
  musaMalloc((void **)&d_output, sizeof(T));
  musaMemcpy(d_input, h_input.data(), sizeof(T) * rows * cols, musaMemcpyHostToDevice);
  musaMemset(d_output, 0, sizeof(T));

  const int n = std::min(rows, cols);
  const int block = BLOCK_SIZE, grid = (n + block - 1) / block;
  const int smem_size = warpSize * sizeof(T);
  trace_kernel<<<grid, block, smem_size>>>(d_input, d_output, cols, n);

  T* h_output = (T*)malloc(sizeof(T));
  musaMemcpy(h_output, d_output, sizeof(T), musaMemcpyDeviceToHost);

  musaFree(d_input), musaFree(d_output);

  return *h_output;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {     

  int Br = 64, Bc = 64;
  
  // dynamic shared memory size
  const int smem_size = (Br * head_dim + 2 * Bc * head_dim) * sizeof(T);

  const int tgt_size = batch_size * target_seq_len * query_heads * head_dim;
  const int src_size = batch_size * src_seq_len * kv_heads * head_dim;

  const float scale_factor = 1.0f / sqrtf((float)head_dim);

  dim3 block(Br);
  dim3 grid((target_seq_len + Br - 1) / Br, batch_size * query_heads);

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;

  musaMalloc((void **)&d_q, sizeof(T) * tgt_size);
  musaMalloc((void **)&d_k, sizeof(T) * src_size);
  musaMalloc((void **)&d_v, sizeof(T) * src_size);
  musaMalloc((void **)&d_o, sizeof(T) * tgt_size);

  musaMemcpy(d_q, h_q.data(), sizeof(T) * tgt_size, musaMemcpyHostToDevice);
  musaMemcpy(d_k, h_k.data(), sizeof(T) * src_size, musaMemcpyHostToDevice);
  musaMemcpy(d_v, h_v.data(), sizeof(T) * src_size, musaMemcpyHostToDevice);
  
  flashAttention_kernel<<<grid, block, smem_size>>>(d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim,
                                                    is_causal, scale_factor, Br, Bc);

  musaMemcpy(h_o.data(), d_o, sizeof(T) * tgt_size, musaMemcpyDeviceToHost);

  musaFree(d_q), musaFree(d_k),
  musaFree(d_v), musaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
