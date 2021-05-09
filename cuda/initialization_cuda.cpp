#include <curand.h>
#include <cuda_runtime.h>
#include <cuda.h>

void weight_init_cuda(float * W, int n){
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, W, n);
}