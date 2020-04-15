#include <iostream>
#include <random>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <typeinfo>
#include <algorithm>
#include <iomanip>
#include <string>

#define ERROR struct Error

using namespace std::chrono;

struct Error{
    const char* str;
    int64_t identity;
    int64_t attempt;
    int64_t extra;
};

struct Error success();
struct Error failure(const char* str, int64_t identity, int64_t attempt);

// Generate a single testcase for a particular length.
template<typename C, typename T>
class setup{
    public:
    T* tooffsets;
    C* fromstarts;
    C* fromstops;
    int64_t startsoffset;
    int64_t stopsoffset;
    int64_t length;

    setup(int64_t len){
       this->length = len;
       startsoffset = 0;
       stopsoffset = 0;
       tooffsets = new T[length + 1];
       fromstarts = new C[length];
       fromstops = new C[length];
       for (size_t i = 0; i < length; ++i) fromstarts[i] = 0;
       for (size_t i = 0; i < length; ++i) fromstops[i] = 1;
    }
    
    ~setup(){
        delete[] tooffsets;
        delete[] fromstarts;
        delete[] fromstops;
    }
};

template <typename C, typename T>
ERROR  __attribute__ ((optimize("-O2"))) awkward_listarray_compact_offsets2(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length){
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

template <typename C, typename T>
ERROR  __attribute__ ((optimize("-O3"))) awkward_listarray_compact_offsets3(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length){
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

template <typename C, typename T>
double test(int64_t length, int opt){
    setup <C, T>a(length);
    
    time_point<high_resolution_clock> t1 = high_resolution_clock::now();
    if(opt == 2) awkward_listarray_compact_offsets2<C, T>(a.tooffsets, a.fromstarts, a.fromstops, a.startsoffset, a.stopsoffset, a.length);
    else awkward_listarray_compact_offsets3<C, T>(a.tooffsets, a.fromstarts, a.fromstops, a.startsoffset, a.stopsoffset, a.length);
    time_point<high_resolution_clock> t2 = high_resolution_clock::now();

    return duration_cast<microseconds>(t2 - t1).count() / 1000.0;
}

int main(){

    int testcases = 10;

    std::cout << "Length\t\t\tOpt\tType\t\t\tAvg time\tStd deviation\n";
    for(int l = 20; l < 28; l += 1){
        for(int opt : {2, 3}){
            int length = 1 << l;
            
            double res[3][testcases];

            for(int testcase = 0; testcase < testcases; testcase++){
                res[0][testcase] = test<int32_t, int64_t>(length, opt);
                res[1][testcase] = test<uint32_t, int64_t>(length, opt);
                res[2][testcase] = test<int64_t, int64_t>(length, opt);
            }
            //Print results
            for(int i : {0, 1, 2}){
                double mean = 0.0;
                for(double val : res[i]) mean += val;
                mean /= testcases;

                double std_dev = 0.0;
                for(double val: res[i]) std_dev += (val - mean) * (val - mean);
                std_dev = sqrt(std_dev / testcases);

                std::cout << std::left << std::setw(24);
                std::cout << "2^" + std::to_string(l) + " = " + std::to_string(length);
                std::cout << opt << "\t";
                switch(i){
                    case 0: std::cout << "<int32_t, int64_t>\t";
                        break;
                    case 1: std::cout << "<uint32_t, int64_t>\t";
                        break;
                    case 2: std::cout << "<int64_t, int64_t>\t";
                        break;
                }
                std::cout << mean << "ms  \t" << std_dev << '\n';
            }

        }
    }

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
