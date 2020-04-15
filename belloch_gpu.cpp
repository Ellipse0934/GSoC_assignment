#include <iostream>
#include <iomanip>

#define ERROR struct Error
#define TESTCASES 10

struct Error{
    const char* str;
    int64_t identity;
    int64_t attempt;
    int64_t extra;
};

struct Error success();
struct Error failure(const char* str, int64_t identity, int64_t attempt);
template <typename C, typename T> int test(int length);

const int elements_per_thread = 2; //DO NOT MODIFY
const int threads_per_block = 512; //Can Modify: Don't set over 512, broadcast kernel needs modification for that
// ^ Cannot remove const and modify between kernel launches
const int elements_per_block = elements_per_thread * threads_per_block;

/*
template <typename C, typename T>
ERROR awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length){
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

template <typename C, typename T>
__global__
void scan_diff(T *tooffsets, T *res_sums, const C* fromstarts,const C* fromstops, int64_t n, bool *err) { 
    extern __shared__ __align__(sizeof(T)) unsigned char cache[];
    T *smem = reinterpret_cast<T *>(cache);
    __shared__ bool error;

    bool local_error = false;
    int thid = threadIdx.x; 
    int offset = 1;

    int ai = thid; 
    int bi = thid + (n/2); 

    int64_t gai = ai + blockIdx.x * blockDim.x * 2;
    int64_t gbi = bi + blockIdx.x * blockDim.x * 2;

    smem[ai] = fromstops[gai] - fromstarts[gai];
    smem[bi] = fromstops[gbi] - fromstarts[gbi];

    if(fromstops[gai] < fromstarts[gai]) local_error = true;
    if(fromstops[gbi] < fromstarts[gbi]) local_error = true;

    for (int d = n >> 1; d > 0; d >>= 1) { 
        __syncthreads();    
        if (thid < d){ 
            int ai = offset*(2*thid+1)-1; 
            int bi = offset*(2*thid+2)-1; 

            smem[bi] += smem[ai];    
        }    
        offset *= 2; 
    }
    
     if (thid == 0){ 
        smem[n] = smem[n - 1];
        smem[n - 1] = 0;  
        error = false;    // Maybe cheaper to do if(global_id == 0) error = false to avoid write conflict
     }

     for (int d = 1; d < n; d *= 2)  {   
         offset >>= 1;      
         __syncthreads();      
         if (thid < d){
            int ai = offset*(2*thid+1)-1; 
            int bi = offset*(2*thid+2)-1; 

            T t = smem[ai]; 
            smem[ai] = smem[bi]; 
            smem[bi] += t;       
        } 
    }  
    if(local_error) error = true;
    __syncthreads(); 
             
    tooffsets[gai] = smem[ai]; 
    tooffsets[gbi] = smem[bi]; 
    if(thid == 0){
        res_sums[blockIdx.x] = smem[n];
        err[blockIdx.x] = error;
    }
}

template <typename T>
__global__
void scan_kernel(T* array, T* sums, int n){
    extern __shared__ __align__(sizeof(T)) unsigned char cache[];
    T *smem = reinterpret_cast<T *>(cache);

    int thid = threadIdx.x; 
    int offset = 1;
    int ai = thid; 
    int bi = thid + (n/2); 
    int64_t gai = ai + blockIdx.x * blockDim.x * 2;
    int64_t gbi = bi + blockIdx.x * blockDim.x * 2;
    smem[ai] = array[gai];
    smem[bi] = array[gbi];
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();    
        if (thid < d){ 
            int ai = offset*(2*thid+1)-1; 
            int bi = offset*(2*thid+2)-1; 

            smem[bi] += smem[ai];    
        }    
        offset *= 2; 
    }
     if (thid == 0){
        smem[n] = smem[n - 1];
        smem[n - 1] = 0;
     }
     for (int d = 1; d < n; d *= 2){
         offset >>= 1;      
         __syncthreads();      
         if (thid < d){
             int ai = offset*(2*thid+1)-1; 
             int bi = offset*(2*thid+2)-1; 

            T t = smem[ai]; 
            smem[ai] = smem[bi]; 
            smem[bi] += t;   
        }
    }
    __syncthreads();
    
    array[gai] = smem[ai]; 
    array[gbi] = smem[bi]; 
    if(thid == 0) sums[blockIdx.x] = smem[n];
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
    int64_t blocks = (length - 1) / elements_per_block + 1;

    T *sums;
    int64_t sums_length = length/elements_per_block;
    int64_t sums_capacity = ((length - 1) / elements_per_block + 1)*elements_per_block;
    cudaMalloc(&sums, sizeof(T) * sums_capacity); 
    scan_kernel<T><<<blocks, threads_per_block, sizeof(T) * (elements_per_block + 1)>>>(array, sums, elements_per_block);
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "kernel failure: rscan" << std::endl;

    if(length > elements_per_block){
      scan_and_broadcast(sums, sums_length);
      int broadcast_blocks = length / elements_per_block;
      int broadcast_threads = threads_per_block * 2; 
      broadcast_scan<T><<<broadcast_blocks, broadcast_threads>>>(array, sums, sums_capacity);
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
    int blockDim = elements_per_block;
    int64_t blocks = (length - 1) / elements_per_block + 1;

    bool *error_block;
    cudaMalloc(&error_block, blocks);
    bool error_block_h[blocks];

    T *sums;
    int64_t sums_length = length/elements_per_block;
    int64_t sums_capacity = ((length - 1)/elements_per_block + 1)*elements_per_block;
    cudaMalloc(&sums, sizeof(T) * sums_capacity);

    scan_diff<C, T><<<blocks, threads_per_block, sizeof(T) * (elements_per_block + 1)>>>(tooffsets, sums, fromstarts + startsoffset, fromstops + stopsoffset, blockDim, error_block);
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "kernel failure" << std::endl;

    // TODO: Use a reduction kernel to find the minimum index where error occurs
    cudaMemcpy(error_block_h, error_block, blocks, cudaMemcpyDeviceToHost);
    for(int i = 0; i < blocks; i++){ // TODO: memcpy fromstarts and fromstops to get exact index
        if(error_block_h[i] == true){
            std::cout << "Error: stop < start in block " << i + 1 << '\n';
            return failure("GPU Kernel failure", 0, 0);
        }
    }

    if(length > elements_per_block){
        scan_and_broadcast(sums, sums_length);
        int broadcast_blocks = length / elements_per_block;
        int broadcast_threads = threads_per_block * 2;  
        broadcast_scan<T><<<broadcast_blocks, broadcast_threads>>>(tooffsets, sums, sums_capacity);
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
  