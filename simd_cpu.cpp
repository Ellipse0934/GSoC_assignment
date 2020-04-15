#include <iostream>
#include <ctime>
#include <cmath>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>
#include <avxintrin.h>
#include <iomanip>

#define TESTCASES 10
#define ERROR struct Error

using namespace std::chrono;
using namespace std;

struct Error{
    const char* str;
    int64_t identity;
    int64_t attempt;
    int64_t extra;
};

struct Error success();
struct Error failure(const char* str, int64_t identity, int64_t attempt);


//Helper function to print the value held in a __m256i register
inline void print(__m256i x, string s = ""){
    cout << s << ' ';
    int *arr = (int *)aligned_alloc(32, sizeof(int) * 8);
    _mm256_store_si256((__m256i *)&arr[0], x);
    for(int i = 0; i < 8; ++i) cout << arr[i] << ' ';
    cout << '\n';
}

ERROR awkward_listarray_compact_offsets(int* tooffsets, const int* fromstarts, const int* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length){
    tooffsets[0] = 0;
    for (int64_t i = 0;  i < length;  i++) {
    int start = fromstarts[startsoffset + i];
    int stop = fromstops[stopsoffset + i];
    if (stop < start) {
        return failure("stops[i] < starts[i]", i, (1UL << 63) - 1);
    }
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
    }
    return success();
}


/*  Best to see the stack-overflow thread and the intel intrinsics docs to understand what's going on here
 *  https://stackoverflow.com/questions/19494114/parallel-prefix-cumulative-sum-with-sse
 *  https://software.intel.com/sites/landingpage/IntrinsicsGuide/#                                          */

__m256i scan(__m256i x) { //a b c d      e f g h
    __m256i t0, t1;
    t0 =  _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 1, 0, 3)); // d a b c      h e f g
    t1 = _mm256_permute2x128_si256(t0, t0, _MM_SHUFFLE(0, 0, 2, 1)); // 0 0 0 0 d a b c
    x = _mm256_add_epi32(x, _mm256_blend_epi32(t0, t1, 0x11));
    t0 = _mm256_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
    t1 = _mm256_permute2x128_si256(t0, t0, 41);
    x = _mm256_add_epi32(x, _mm256_blend_epi32(t0, t1, 0x33));
    x = _mm256_add_epi32(x,_mm256_permute2x128_si256(x, x, 41));
    return x;
}

ERROR awkward_listarray_compact_offsets_AVX(int* tooffsets, const int* fromstarts, const int* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length){
   // int carry = 0;
  //  tooffsets[0] = 0;
    int64_t i = 1;
   __m256i carry_vec = _mm256_setzero_si256(); 
    for (;  i  < length - 7;  i += 8) {
        //int start = fromstarts[startsoffset + i];
        //int stop = fromstops[stopsoffset + i];
        __m256i start = _mm256_loadu_si256((__m256i * )(fromstarts + startsoffset + i - 1));
        __m256i stop = _mm256_loadu_si256((__m256i *)(fromstops + stopsoffset + i - 1));
        __m256i res = stop - start;
        int mask = _mm256_movemask_epi8(stop < start);
        if (mask != 0) {    // TODO: Print exact index
            return failure("stops[i] < starts[i]", i, (1UL << 63) - 1);
        }
        // tooffsets[i + 1] = tooffsets[i] + (stop - start);

        /*
            Originally we put `res = scan(res)` here put the compiler chose 
            to not inline the function call, hence we forced it to by
            pasting the `scan(m256i x)` code here. But, there was no measurable
            difference in performance.
        */
        //shift1_AVX + add
        __m256i t0 =  _mm256_shuffle_epi32(res, _MM_SHUFFLE(2, 1, 0, 3)); // d a b c      h e f g
        __m256i t1 = _mm256_permute2x128_si256(t0, t0, _MM_SHUFFLE(0, 0, 2, 1)); // 0 0 0 0 d a b c
        res = _mm256_add_epi32(res, _mm256_blend_epi32(t0, t1, 0x11));
        //shift2_AVX + add
        t0 = _mm256_shuffle_epi32(res, _MM_SHUFFLE(1, 0, 3, 2));
        t1 = _mm256_permute2x128_si256(t0, t0, 41);
        res = _mm256_add_epi32(res, _mm256_blend_epi32(t0, t1, 0x33));
        //shift3_AVX + add
        res = _mm256_add_epi32(res,_mm256_permute2x128_si256(res, res, 41));


        res += carry_vec;//_mm256_add_epi32(carry_vec, diff);
        _mm256_storeu_si256((__m256i *)( tooffsets + i),res);
        carry_vec = _mm256_set1_epi32(_mm256_extract_epi32(res, 7));

    }

    for (;  i < length;  i++) { //Do remaining elements
        int start = fromstarts[startsoffset + i];
        int stop = fromstops[stopsoffset + i];
        if (stop < start) {
            return failure("stops[i] < starts[i]", i, (1UL << 63) - 1);
        }
        tooffsets[i] = tooffsets[i - 1] +  (stop - start);
    }
    return success();
}

int main(){
    std::cout << "Length\t\t\tAVX\tAvg time\tStd deviation\n";
    for(int l = 20; l < 28; l++) test(l);  
}

class setup{
    public:
    int* tooffsets;
    int* fromstarts;
    int* fromstops;
    int64_t startsoffset;
    int64_t stopsoffset;
    int64_t length;

    setup(int64_t len){
       this->length = len;
       startsoffset = 0;
       stopsoffset = 0;
       tooffsets = (int *)aligned_alloc(32, sizeof(int) * (length + 1));
       fromstarts = (int *)aligned_alloc(32, sizeof(int) * length);
       fromstops = (int *)aligned_alloc(32, sizeof(int) * length);

       if(tooffsets == NULL) std::cout << "Allocation Failed for tooffsets\n";
       if(fromstarts == NULL) std::cout << "Allocation Failed for tooffsets\n";
       if(fromstops == NULL) std::cout << "Allocation Failed for tooffsets\n";

       for (size_t i = 0; i < length; ++i) tooffsets[i] = 1;
       for (size_t i = 0; i < length; ++i) fromstarts[i] = 0;
       for (size_t i = 0; i < length; ++i) fromstops[i] = 1;
    }
    
    ~setup(){
       free(tooffsets);
       free(fromstarts);
       free(fromstops);
    }
};

void test(int64_t l){
    int length = 1 << l;
    setup a(length);
    double time[TESTCASES];

    std::cout << std::left << std::setw(24);
    std::cout << "2^" + std::to_string(l) + " = " + std::to_string(length);

    for(int testcase = 0; testcase < TESTCASES; testcase++){
        time_point<high_resolution_clock> t1 = high_resolution_clock::now();
        auto r = awkward_listarray_compact_offsets_AVX(a.tooffsets, a.fromstarts, a.fromstops, a.startsoffset, a.stopsoffset, a.length);
        time_point<high_resolution_clock> t2 = high_resolution_clock::now();
        time[testcase] = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
        if(a.tooffsets[length - 1] != length - 1)std::cout << "compact_offsets_AVX has failed\n";
        if(r.str != nullptr) std::cout << "FAILURE" << std::endl;
    }
    
    double mean = 0.0;
    for(double val : time) mean += val;
    mean /= TESTCASES;

    double std_dev = 0.0;
    for(double val : time) std_dev += (val - mean) * (val - mean);
    std_dev = sqrt(std_dev / TESTCASES);

   std::cout << "true\t" << mean << "ms \t" << std_dev << '\n';


    for(int testcase = 0; testcase < TESTCASES; testcase++){
        time_point<high_resolution_clock> t1 = high_resolution_clock::now();
        auto r = awkward_listarray_compact_offsets(a.tooffsets, a.fromstarts, a.fromstops, a.startsoffset, a.stopsoffset, a.length);
        time_point<high_resolution_clock> t2 = high_resolution_clock::now();
        time[testcase] = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
        if(a.tooffsets[length - 1] != length - 1)std::cout << "compact_offsets_AVX has failed\n";
        if(r.str != nullptr) std::cout << "FAILURE" << std::endl;
    }

    std::cout << std::left << std::setw(24);
    std::cout << "2^" + std::to_string(l) + " = " + std::to_string(length);

    mean = 0.0;
    for(double val : time) mean += val;
    mean /= TESTCASES;

    std_dev = 0.0;
    for(double val : time) std_dev += (val - mean) * (val - mean);
    std_dev = sqrt(std_dev / TESTCASES);

    std::cout << "false   " << mean <<  "ms \t" << std_dev << '\n';
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
