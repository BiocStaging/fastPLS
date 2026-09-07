#ifndef FASTPLS_CUDA_RESIDENT_LABELS_CUH
#define FASTPLS_CUDA_RESIDENT_LABELS_CUH

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

namespace fastpls_device {

// Sorted class membership is reusable for X'Y and every score'Y product.
template<class T>
__global__ void label_offsets(const int* keys, int n, int classes,
                              int* offsets, T* priors, int* invalid) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c > classes) return;
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (keys[mid] < c + 1) lo = mid + 1; else hi = mid;
    }
    offsets[c] = lo;
    if (c == 0 && (keys[0] < 1 || keys[n-1] > classes)) atomicExch(invalid, 1);
}

template<class T>
__global__ void label_priors(const int* offsets, int n, int classes,
                             T* priors, int* invalid) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= classes) return;
    const int count = offsets[c+1] - offsets[c];
    if (!count) atomicExch(invalid, 1);
    priors[c] = T(count) / T(n);
}

template<class T>
cudaError_t prepare_labels(const int* labels, int n, int classes,
                           int* keys, int* rows, int* offsets, T* priors,
                           int* invalid, cudaStream_t stream) {
    if (n < 1 || classes < 2) return cudaErrorInvalidValue;
    cudaError_t status = cudaMemsetAsync(invalid, 0, sizeof(int), stream);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(keys, labels, size_t(n)*sizeof(int), cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) return status;
    auto policy = thrust::cuda::par.on(stream);
    auto r = thrust::device_pointer_cast(rows);
    auto k = thrust::device_pointer_cast(keys);
    thrust::sequence(policy, r, r+n);
    thrust::stable_sort_by_key(policy, k, k+n, r);
    label_offsets<T><<<(classes+256)/256,256,0,stream>>>(keys,n,classes,offsets,priors,invalid);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    label_priors<<<(classes+255)/256,256,0,stream>>>(offsets,n,classes,priors,invalid);
    return cudaGetLastError();
}

template<class T>
__global__ void class_sums(const T* x, int n, int p, const int* rows,
                           const int* offsets, T* sums) {
    const int c = blockIdx.x, j = blockIdx.y, tid = threadIdx.x;
    __shared__ T partial[256];
    T value = 0;
    for (int i = offsets[c]+tid; i < offsets[c+1]; i += blockDim.x)
        value += x[size_t(j)*n + rows[i]];
    partial[tid] = value;
    __syncthreads();
    for (int step=128; step; step/=2) {
        if (tid<step) partial[tid] += partial[tid+step];
        __syncthreads();
    }
    if (tid==0) sums[size_t(c)*p+j] = partial[0];
}

template<class T>
__global__ void center_class_sums(T* sums, int p, int classes, const T* priors) {
    const int j=blockIdx.x, tid=threadIdx.x;
    __shared__ T partial[256];
    T total=0;
    for (int c=tid;c<classes;c+=blockDim.x) total+=sums[size_t(c)*p+j];
    partial[tid]=total;
    __syncthreads();
    for (int step=128;step;step/=2) {
        if(tid<step) partial[tid]+=partial[tid+step];
        __syncthreads();
    }
    for(int c=tid;c<classes;c+=blockDim.x)
        sums[size_t(c)*p+j]-=partial[0]*priors[c];
}

template<class T>
cudaError_t class_product(const T* x, int n, int p, int classes,
                          const int* rows, const int* offsets,
                          const T* priors, T* out, cudaStream_t stream) {
    if(n<1 || p<1 || p>65535 || classes<2) return cudaErrorInvalidValue;
    class_sums<<<dim3(classes,p),256,0,stream>>>(x,n,p,rows,offsets,out);
    cudaError_t status=cudaGetLastError();
    if(status!=cudaSuccess) return status;
    center_class_sums<<<p,256,0,stream>>>(out,p,classes,priors);
    return cudaGetLastError();
}
} // namespace fastpls_device
#endif
