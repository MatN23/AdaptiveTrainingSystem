// fused_grad_clip.cu
#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8
#define NORM_BUF_DOUBLES 1                 // 1 double  = 8 bytes
#define NORM_BUF_FLOATS                                                        \
  4 // 4 floats  = 16 bytes  → total buf = 24 bytes, allocate 32

// ---------------------------------------------------------------------------
__device__ __forceinline__ double warp_reduce_sum_double(double val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  return val;
}

// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_grad_clip_kernel(
    scalar_t **grad_ptrs, const int *grad_sizes, const int num_tensors,
    const float max_norm,
    double *norm_sq_acc,  // 1 double  – squared norm accumulator
    float *clip_coef_out, // 1 float   – clipping coefficient
    float *final_norm_out // 1 float   – final norm (for logging)
) {
  cg::grid_group grid = cg::this_grid();

  // ------------------------------------------------------------------
  // Phase 1: thread-local squared-norm accumulation
  // ------------------------------------------------------------------
  double thread_sum = 0.0;
  for (int t = 0; t < num_tensors; ++t) {
    scalar_t *g = grad_ptrs[t];
    int sz = grad_sizes[t];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sz;
         i += gridDim.x * blockDim.x) {
      double v = static_cast<double>(g[i]);
      thread_sum += v * v;
    }
  }

  // ------------------------------------------------------------------
  // Phase 2: block reduction  (warp shuffle → shared mem → atomicAdd)
  // ------------------------------------------------------------------
  __shared__ double smem[NUM_WARPS]; // 8 doubles — correctly sized

  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  thread_sum = warp_reduce_sum_double(thread_sum);
  if (lane == 0)
    smem[wid] = thread_sum;
  __syncthreads();

  if (wid == 0) {
    double val = (lane < NUM_WARPS) ? smem[lane] : 0.0;
    val = warp_reduce_sum_double(val);
    if (lane == 0)
      atomicAdd(norm_sq_acc, val);
  }

  // ------------------------------------------------------------------
  // Phase 3: grid sync → compute clip coef (thread 0 only)
  // ------------------------------------------------------------------
  grid.sync();

  if (grid.thread_rank() == 0) {
    float norm = sqrtf((float)(*norm_sq_acc) + 1e-16f);
    float coef = fminf(max_norm / (norm + 1e-6f), 1.0f);
    *clip_coef_out = coef;
    *final_norm_out = norm;
  }

  grid.sync();

  // ------------------------------------------------------------------
  // Phase 4: apply clipping in-place (skip if within threshold)
  // ------------------------------------------------------------------
  float coef = *clip_coef_out;
  if (coef >= 0.999f)
    return;

  for (int t = 0; t < num_tensors; ++t) {
    scalar_t *g = grad_ptrs[t];
    int sz = grad_sizes[t];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sz;
         i += gridDim.x * blockDim.x) {
      g[i] = static_cast<scalar_t>(static_cast<float>(g[i]) * coef);
    }
  }
}

// ---------------------------------------------------------------------------
// Compute safe block count for cooperative launch
// ---------------------------------------------------------------------------
template <typename scalar_t> static int safe_num_blocks() {
  int num_sms = 0, bpsm = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &bpsm, fused_grad_clip_kernel<scalar_t>, BLOCK_SIZE, 0);
  int n = num_sms * bpsm;
  return (n < 1) ? 1 : (n > 512 ? 512 : n); // cap at 512 for safety
}

// ---------------------------------------------------------------------------
// Core launch (shared by both dtype paths)
// ---------------------------------------------------------------------------
template <typename scalar_t>
static cudaError_t
do_launch(scalar_t **ptrs, int *sizes, int num_tensors, float max_norm,
          void *norm_buffer, // must be ≥32 bytes, 8-byte aligned device ptr
          cudaStream_t stream) {
  if (!norm_buffer || !ptrs || !sizes || num_tensors <= 0)
    return cudaErrorInvalidValue;

  // Verify cooperative launch is supported on this device
  int coop = 0;
  cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
  if (!coop) {
    fprintf(stderr, "[fused_grad_clip] ERROR: device does not support "
                    "cooperative launch\n");
    return cudaErrorNotSupported;
  }

  // Zero the entire buffer before use
  cudaMemsetAsync(norm_buffer, 0, 32, stream);

  // Buffer layout (device memory, 8-byte aligned):
  //   offset  0 : double norm_sq_acc    (8 bytes)
  //   offset  8 : float  clip_coef_out  (4 bytes)
  //   offset 12 : float  final_norm_out (4 bytes)
  char *buf = static_cast<char *>(norm_buffer);
  double *norm_sq_acc = reinterpret_cast<double *>(buf);
  float *clip_coef_out = reinterpret_cast<float *>(buf + 8);
  float *final_norm = reinterpret_cast<float *>(buf + 12);

  int num_blocks = safe_num_blocks<scalar_t>();

  void *args[] = {&ptrs,        &sizes,         &num_tensors, &max_norm,
                  &norm_sq_acc, &clip_coef_out, &final_norm};

  return cudaLaunchCooperativeKernel(
      (const void *)fused_grad_clip_kernel<scalar_t>, dim3(num_blocks),
      dim3(BLOCK_SIZE), args, 0, stream);
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------
extern "C" {

// Unified launcher — called from Python
// use_fp16: 0 = float32, 1 = float16
// norm_buffer: device ptr to ≥32 bytes, malloc'd with cudaMalloc (8-byte
// aligned)
int fused_grad_clip_launcher(void **grad_ptrs_device, int *grad_sizes_device,
                             int num_tensors, float max_norm, void *norm_buffer,
                             int use_fp16, cudaStream_t stream) {
  cudaError_t err;
  if (use_fp16)
    err = do_launch<__half>((__half **)grad_ptrs_device, grad_sizes_device,
                            num_tensors, max_norm, norm_buffer, stream);
  else
    err = do_launch<float>((float **)grad_ptrs_device, grad_sizes_device,
                           num_tensors, max_norm, norm_buffer, stream);

  if (err != cudaSuccess)
    fprintf(stderr, "[fused_grad_clip] launch error: %s\n",
            cudaGetErrorString(err));
  return (int)err;
}

// FP32 convenience
int fused_grad_clip_launcher_fp32(float **grad_ptrs_device,
                                  int *grad_sizes_device, int num_tensors,
                                  float max_norm, void *norm_buffer,
                                  cudaStream_t stream) {
  cudaError_t err =
      do_launch<float>(grad_ptrs_device, grad_sizes_device, num_tensors,
                       max_norm, norm_buffer, stream);
  if (err != cudaSuccess)
    fprintf(stderr, "[fused_grad_clip] fp32 error: %s\n",
            cudaGetErrorString(err));
  return (int)err;
}

// FP16 convenience
int fused_grad_clip_launcher_fp16(__half **grad_ptrs_device,
                                  int *grad_sizes_device, int num_tensors,
                                  float max_norm, void *norm_buffer,
                                  cudaStream_t stream) {
  cudaError_t err =
      do_launch<__half>(grad_ptrs_device, grad_sizes_device, num_tensors,
                        max_norm, norm_buffer, stream);
  if (err != cudaSuccess)
    fprintf(stderr, "[fused_grad_clip] fp16 error: %s\n",
            cudaGetErrorString(err));
  return (int)err;
}

// Read results back to host (call AFTER cudaStreamSynchronize)
void fused_grad_clip_get_results(void *norm_buffer, float *out_norm,
                                 float *out_clip_coef) {
  if (!norm_buffer)
    return;
  char *buf = static_cast<char *>(norm_buffer);
  if (out_clip_coef)
    cudaMemcpy(out_clip_coef, buf + 8, sizeof(float), cudaMemcpyDeviceToHost);
  if (out_norm)
    cudaMemcpy(out_norm, buf + 12, sizeof(float), cudaMemcpyDeviceToHost);
}

} // extern "C"