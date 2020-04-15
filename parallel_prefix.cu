#include <iostream>

/*  In this program we compute the exclusive parallel prefix addition of an array. This operation on an array computes
 *  the intermediate sums. Hence [a, b, c, d, e] will become [0, a, a + b, a + b + c, a + b + c + d]
 *  
 *  We divide our array up into blocks, with each block being of size 256. Our shfl_scan kernel computes the
 *  exclusive parallel prefix addition for each block independently and store its cumulative sum. Then we 
 *  perform a scan on the resultant sums and add them to the original array
 * 
 *  Expressing this recursively seems natural but I want to reduce to overhead of a function call by telling the compiler
 *  that we won't ever have to make more than 4 recursive calls to scan_and_broadcast. Something like a loop unroll but
 *  not exactly. I can't find the exact word. Should I even worry about the overhead of the extre stackframe ? I can 
 *  share the benchmarks if required.                                                                                           */

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
    value = lane_id == 0 ? blockSum:value;
    data[id] = value;
  }

template <typename T>
__global__
void broadcast_sums(T* array, T* sums, int64_t length){
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
int scan(T* array, int64_t length){
    cudaError_t err;
    T *sums = NULL;

    int gridSize = length / 256;
    int blockSize = 256;
    int smem_sz = gridSize * sizeof(T);
    int sums_capacity = max(length / blockSize, (int64_t)blockSize);

    if(gridSize > 1){
    cudaMalloc(&sums, sizeof(T) * sums_capacity);
    cudaMemset(sums, 0, sizeof(T) * sums_capacity);
    }
    shfl_scan<T><<<gridSize, blockSize, smem_sz>>>(array, sums, 32);
    err = cudaGetLastError();
    if (err != cudaSuccess){
      cudaFree(sums);
      return 1;
    }

    if(gridSize > 1){
      scan(sums, sums_capacity);
      broadcast_sums<T><<<gridSize, blockSize>>>(array, sums, length);
      err = cudaGetLastError();
      if (err != cudaSuccess){
        cudaFree(sums);
        return 1;
      }
    }
    cudaFree(sums);
    return 0;
}

int main(){
    int length = 1 << 15; // Needs to be multiple of 256
    
    int * array_h = new int[length];
    int * array_d;
    cudaMalloc(&array_d, sizeof(int) * length);
    
    for(int i = 0; i < length; i++) array_h[i] = 1; 
    // Kernel computes the exclusive parallel prefix sum of this array.
    
    cudaMemcpy(array_d, array_h, sizeof(int) * length, cudaMemcpyHostToDevice);
    
    scan(array_d, length);
    
    int result;
    cudaMemcpy(&result, array_d + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << result << std::endl;
    return 0;
}
