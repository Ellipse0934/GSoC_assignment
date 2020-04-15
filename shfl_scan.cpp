#include <iostream>
#include <iomanip>

#define ERROR struct Error
#define TESTCASES 10
using namespace std;

/*
 *  template <typename C, typename T>
    ERROR   awkward_listarray_compact_offsets3(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length){
        tooffsets[0] = 0;
        for (int64_t i = 0;  i < length;  i++) {
            C start = fromstarts[startsoffset + i];
            C stop = fromstops[stopsoffset + i];
            if (stop < start) {
                return failure("stops[i] < starts[i]", i, (1UL << 63) - 1);
            }
            tooffsets[i + 1] = tooffsets[i] + (stop - start);
        }
        return success();
    }

*/
struct Error{
    const char* str;
    int64_t identity;
    int64_t attempt;
    int64_t extra;
};

struct Error success();
struct Error failure(const char* str, int64_t identity, int64_t attempt);
template <typename C, typename T> int test(int l);

const int elements_per_thread = 1;
const int threads_per_block = 256; //DO NOT MODIFY
const int elements_per_block = elements_per_thread * threads_per_block;



template <typename C, typename T>
__global__
void scan_diff(T *tooffsets, T  *res_sums,const C* fromstarts,const C* fromstops, int64_t length, bool *err) { 
    extern __shared__ T sums[];
    __shared__ bool error;

    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if(id > length) return;
    int lane_id = id % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if(threadIdx.x == 0){
        error = false;
    }
    int width = 256;

    __syncthreads();
    if(fromstops[id] < fromstarts[id]) error = true;
    T value = fromstops[id] - fromstarts[id];

    #pragma unroll
    for (int i = 1; i <= width; i *= 2) {
      unsigned int mask = 0xffffffff;
      T n = __shfl_up_sync(mask, value, i, width);
  
      if (lane_id >= i) value += n;
    }
  
    if (threadIdx.x % warpSize == warpSize - 1) sums[warp_id] = value;


    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
      T warp_sum = sums[lane_id];
  
      int mask = (1 << (blockDim.x / warpSize)) - 1;
      for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
        T n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));
  
        if (lane_id >= i) warp_sum += n;
      }
  
      sums[lane_id] = warp_sum;
    }
  
    __syncthreads();

    int blockSum = 0;
    if (warp_id > 0) blockSum = sums[warp_id - 1];
    value += blockSum;

    if (res_sums != NULL && threadIdx.x == blockDim.x - 1) {
      res_sums[blockIdx.x] = value;
      err[blockIdx.x] = error;
    }
    value = __shfl_up_sync(0x7fffffff, value, 1);
    value = lane_id == 0 ? blockSum:value;     // TODO: Something better probably exists here, 
    tooffsets[id] = value;
}

template <typename T>
__global__ 
void shfl_scan(T *data, T *partial_sums, int width) {
    extern __shared__ T sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    int warp_id = threadIdx.x / warpSize;
  
    T value = data[id];
  
  #pragma unroll
    for (int i = 1; i <= width; i *= 2) {
      unsigned int mask = 0xffffffff;
      T n = __shfl_up_sync(mask, value, i, width);
  
      if (lane_id >= i) value += n;
    }
  
    if (threadIdx.x % warpSize == warpSize - 1) {
      sums[warp_id] = value;
    }

    __syncthreads();
  
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
      T warp_sum = sums[lane_id];
  
      int mask = (1 << (blockDim.x / warpSize)) - 1;
      for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
        T n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));
  
        if (lane_id >= i) warp_sum += n;
      }
  
      sums[lane_id] = warp_sum;
    }
  
    __syncthreads();
    int blockSum = 0;
    if (warp_id > 0) blockSum = sums[warp_id - 1];
    value += blockSum;

    if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
      partial_sums[blockIdx.x] = value;
    }
    value = __shfl_up_sync(0x7fffffff, value, 1);
    value = lane_id == 0 ? blockSum:value; //TODO: Something better probably exists here, 
    data[id] = value;
  }

template <typename T>
__global__
void broadcast_scan(T* array, T* sums, int64_t length){
    __shared__ T val;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= length) return;
    if(threadIdx.x == 0){
        val = sums[blockIdx.x];
    }
    __syncthreads();
    array[idx] += val;
}

template <typename T>
int scan_and_broadcast(T* array, int64_t length){
    cudaError_t err;
    T *sums;

    int gridSize = length / threads_per_block;
    int smem_sz = 8 * sizeof(T);
    int sums_capacity = max(length / threads_per_block, 256L);

    cudaMalloc(&sums, sizeof(T) * sums_capacity);
    cudaMemset(sums, 0, sizeof(T) * sums_capacity);
    shfl_scan<T><<<gridSize, threads_per_block, smem_sz>>>(array, sums, 32);
    err = cudaGetLastError();
    if (err != cudaSuccess){
      cudaFree(sums);
      return 1;
    }

    if(length > threads_per_block){
      scan_and_broadcast(sums, sums_capacity);
      int broadcast_blocks = length / threads_per_block;
      int broadcast_threads = threads_per_block; 
      broadcast_scan<T><<<broadcast_blocks, broadcast_threads>>>(array, sums, length);
      err = cudaGetLastError();
      if (err != cudaSuccess){
        cudaFree(sums);
        return 1;
      }
    }
    cudaFree(sums);
    return 0;
}


/* A CPU function which calls the relevant GPU Kernels to calculate the compact offsets

    First we do a scan on the difference between fromstops - fromstarts.
    Then we call the scan and broadcast function which recursively applies the scan
*/

template <typename C, typename T>
ERROR awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length){
  cudaError_t err;
  int64_t blocks = length/elements_per_block;

  bool *error_block;
  bool error_block_h[blocks];
  cudaMalloc(&error_block, blocks);

  T *sums;
  int64_t sums_capacity = ((blocks - 1)/threads_per_block + 1)*threads_per_block;
  cudaMalloc(&sums, sizeof(T) * sums_capacity); 
  scan_diff<C, T><<<blocks, threads_per_block, sizeof(T) * (elements_per_block + 1)>>>(tooffsets, sums, fromstarts + startsoffset, fromstops + stopsoffset, length, error_block);
  err = cudaGetLastError();
  if (err != cudaSuccess){
    cudaFree(error_block);
    cudaFree(sums);
    return failure("GPU Kernel failure: scan_diff kernel failed", 0, 0);
  }

  cudaMemcpy(error_block_h, error_block, blocks, cudaMemcpyDeviceToHost);
  for(int i = 0; i < blocks; i++){
      if(error_block_h[i] == true){
        std::cout << "Error: stop < start in block " << i + 1 << '\n';
        cudaFree(error_block);
        cudaFree(sums);
        return failure("GPU Kernel failure: stop < start in block", i, (1UL << 63) - 1);
      }
  }

  if(length > 256){
        if(scan_and_broadcast(sums, sums_capacity) != 0){
          cudaFree(error_block);
          cudaFree(sums);
          return failure("GPU Kernel failure: scan_and_broadcast has failed", 0, 0);
        }
        int broadcast_blocks = length / threads_per_block;
        int broadcast_threads = threads_per_block; 
        broadcast_scan<T><<<broadcast_blocks, broadcast_threads>>>(tooffsets, sums, length);
        err = cudaGetLastError();
        if (err != cudaSuccess){
          cudaFree(error_block);
          cudaFree(sums);
          return failure("GPU Kernel failure: Broadcast to tooffsets failed", 0, 0);
        }
    }
    cudaFree(error_block);
    cudaFree(sums);
    return success();      
}


int main(){
  std::cout << "Length\t\t\tAvg time\tStd deviation\n";
  for(int l = 20; l <= 27; l += 1){
    test<int64_t, int64_t>(l);
  }
}

template <typename C, typename T>
int test(int l){
  int length = 1 << l;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaError_t err;
  T *tooffsets_h = (T *) malloc(sizeof(T) * length);
  C *fromstarts_h = (C *) malloc(sizeof(C) * length);
  C *fromstops_h = (C *) malloc(sizeof(C) * length);

  T *tooffsets_d;
  C *fromstarts_d;
  C *fromstops_d;
  
  err = cudaMalloc(&tooffsets_d, sizeof(T) * length);
  if (err != cudaSuccess) printf("Allocation failure!\n");
  err = cudaMalloc(&fromstarts_d, sizeof(C) * length);
  if (err != cudaSuccess) printf("Allocation failure!\n");
  err = cudaMalloc(&fromstops_d, sizeof(C) * length);
  if (err != cudaSuccess) printf("Allocation failure!\n");

  for(int i = 0; i < length; ++i){
      fromstarts_h[i] = 0;
      fromstops_h[i] = 1;
  }
  
  err = cudaMemcpy(fromstarts_d, fromstarts_h, sizeof(C) * length, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) printf("Copy Failure! fromstarts\n");
  err = cudaMemcpy(fromstops_d, fromstops_h, sizeof(C) * length, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) printf("Copy Failure! fromstops\n");
  
  float time[TESTCASES];

  for(int testcase = 0; testcase < TESTCASES; testcase++){

  cudaEventRecord(start);
  awkward_listarray_compact_offsets(tooffsets_d, fromstarts_d, fromstops_d, 0, 0, length);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time[testcase], start, stop);

  T result;
  cudaMemcpy(&result, tooffsets_d + length - 1, sizeof(T), cudaMemcpyDeviceToHost);
  if(result != length - 1){
      std::cout << "FAILURE\n";
      return 0;
  }
  }

  err = cudaMemcpy(tooffsets_h, tooffsets_d, sizeof(T) * length, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) printf("%s \n", cudaGetErrorString(err));

  for(T i = 0; i < length; i++){
      if(i != tooffsets_h[i]){
          std::cout << "ERROR: " << i << ' ' << tooffsets_h[i] << std::endl;
          return 0;
      }
  }

  float mean = 0.0;
  for(float val : time) mean += val;
  mean /= TESTCASES;

  float std_dev = 0.0;
  for(float val : time) std_dev += (val - mean) * (val - mean);
  std_dev = sqrt(std_dev / TESTCASES);

  std::cout << std::left << std::setw(23);
  std::cout << "2^" + std::to_string(l) + " = " + std::to_string(length);
  std::cout << ' ' << mean << "ms\t" << std_dev << std::endl;

  cudaFree(tooffsets_d);
  cudaFree(fromstarts_d);
  cudaFree(fromstops_d);

  free(tooffsets_h);
  free(fromstarts_h);
  free(fromstops_h);
  
  return 0;
}


struct Error success() {
    struct Error out;
    out.str = nullptr;
    out.identity = (1UL << 63) - 1;
    out.attempt = (1UL << 63) - 1;
    out.extra = 0;
    return out;
  }
  
  struct Error failure(const char* str, int64_t identity, int64_t attempt) {
    struct Error out;
    out.str = str;
    out.identity = identity;
    out.attempt = attempt;
    out.extra = 0;
    return out;
  }
  
