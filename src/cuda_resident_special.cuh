#ifndef FASTPLS_CUDA_RESIDENT_SPECIAL_CUH
#define FASTPLS_CUDA_RESIDENT_SPECIAL_CUH

#include "cuda_resident_simpls.cuh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace fastpls_device {

template<class T>
__global__ void opls_difference(T* out,const T* loading,const T* weight,
                                const T* ratio,int size) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;
        i+=blockDim.x*gridDim.x)
        out[i]=loading[i]-weight[i]*ratio[0];
}

template<class T>
__global__ void scalar_ratio(const T* numerator,const T* denominator,T* out,
                             int* invalid) {
    if(!(denominator[0]>T(0))||!isfinite(denominator[0])||
       !isfinite(numerator[0])) {
        atomicExch(invalid,1);out[0]=T(0);
    } else out[0]=numerator[0]/denominator[0];
}

template<class T>
__global__ void reciprocal_device(const T* value,T* out,int* invalid,
                                  bool square_root) {
    if(!(value[0]>T(0))||!isfinite(value[0])) {
        atomicExch(invalid,1);out[0]=T(0);
    } else out[0]=square_root?rsqrt(value[0]):T(1)/value[0];
}

template<class T>
__global__ void row_squared_norms(const T* x,int rows,int columns,T* norms) {
    const int row=blockIdx.x;
    if(row>=rows)return;
    __shared__ T partial[256];
    T sum=0;
    for(int column=threadIdx.x;column<columns;column+=blockDim.x) {
        const T value=x[row+size_t(column)*rows];
        sum+=value*value;
    }
    partial[threadIdx.x]=sum;
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0)norms[row]=partial[0];
}

template<class T>
__global__ void transform_kernel(T* kernel,int rows,int columns,int kind,
                                 T gamma,int degree,T coefficient,
                                 const T* left_norms,const T* right_norms) {
    for(size_t index=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
        index<size_t(rows)*columns;index+=size_t(blockDim.x)*gridDim.x) {
        const int row=index%rows;
        const int column=index/rows;
        const T dot=kernel[index];
        if(kind==2) {
            T distance=left_norms[row]+right_norms[column]-T(2)*dot;
            if(distance<T(0))distance=T(0);
            kernel[index]=exp(-gamma*distance);
        } else {
            T base=gamma*dot+coefficient;
            T value=T(1);
            for(int power=0;power<degree;++power)value*=base;
            kernel[index]=value;
        }
    }
}

template<class T>
__global__ void column_means(const T* matrix,int rows,int columns,T* means) {
    const int column=blockIdx.x;
    if(column>=columns)return;
    __shared__ T partial[256];
    T sum=0;
    for(int row=threadIdx.x;row<rows;row+=blockDim.x)
        sum+=matrix[row+size_t(column)*rows];
    partial[threadIdx.x]=sum;
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0)means[column]=partial[0]/T(rows);
}

template<class T>
__global__ void vector_mean(const T* values,int size,T* result) {
    __shared__ T partial[256];
    T sum=0;
    for(int i=threadIdx.x;i<size;i+=blockDim.x)sum+=values[i];
    partial[threadIdx.x]=sum;
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0)result[0]=partial[0]/T(size);
}

template<class T>
__global__ void center_training_kernel(T* matrix,int size,const T* means,
                                       const T* grand) {
    for(size_t index=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
        index<size_t(size)*size;index+=size_t(blockDim.x)*gridDim.x) {
        const int row=index%size;
        const int column=index/size;
        matrix[index]-=means[row]+means[column]-grand[0];
    }
}

template<class T>
__global__ void row_means(const T* matrix,int rows,int columns,T* means) {
    const int row=blockIdx.x;
    if(row>=rows)return;
    __shared__ T partial[256];
    T sum=0;
    for(int column=threadIdx.x;column<columns;column+=blockDim.x)
        sum+=matrix[row+size_t(column)*rows];
    partial[threadIdx.x]=sum;
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(threadIdx.x<step)partial[threadIdx.x]+=partial[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0)means[row]=partial[0]/T(columns);
}

template<class T>
__global__ void center_test_kernel(T* matrix,int rows,int columns,
                                   const T* row_average,
                                   const T* training_average,
                                   const T* grand) {
    for(size_t index=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
        index<size_t(rows)*columns;index+=size_t(blockDim.x)*gridDim.x) {
        const int row=index%rows;
        const int column=index/rows;
        matrix[index]-=row_average[row]+training_average[column]-grand[0];
    }
}

template<class T> class ResidentOpls {
    int n,p,q,components,north,oversample,power;
    cudaStream_t stream;
    cublasHandle_t host_blas=nullptr,device_blas=nullptr;
    cusolverDnHandle_t solver_handle=nullptr;
    curandGenerator_t rng_handle=nullptr;
    T *filtered=nullptr,*response=nullptr,*crosscov=nullptr,*meanX=nullptr,
      *scaleX=nullptr,*meanY=nullptr,*scaleY=nullptr,*W=nullptr,*P=nullptr,
      *candidate=nullptr,*score=nullptr,*loading=nullptr,*orthogonal=nullptr,
      *scalars=nullptr,*effectiveR=nullptr;
    int *labels=nullptr,*keys=nullptr,*rows=nullptr,*offsets=nullptr,
      *invalid=nullptr;
    int prediction_capacity=0;
    T* prediction_score=nullptr;
    bool classification=false;
    std::unique_ptr<RsvdWorkspace<T>> direction;
    std::unique_ptr<ResidentSimpls<T>> inner;
    template<class U> void allocate(U*& value,size_t count) {
        require_cuda(cudaMalloc(&value,count*sizeof(U)));
    }
    void release() noexcept {
        inner.reset();direction.reset();
        for(T* value:{filtered,response,crosscov,meanX,scaleX,meanY,scaleY,W,P,
                      candidate,score,loading,orthogonal,scalars,effectiveR,
                      prediction_score})cudaFree(value);
        for(int* value:{labels,keys,rows,offsets,invalid})cudaFree(value);
        if(rng_handle)curandDestroyGenerator(rng_handle);
        if(solver_handle)cusolverDnDestroy(solver_handle);
        if(device_blas)cublasDestroy(device_blas);
        if(host_blas)cublasDestroy(host_blas);
    }
    void gemm(cublasOperation_t left,cublasOperation_t right,int output_rows,
              int output_columns,int shared,const T* a,int lda,const T* b,
              int ldb,T* out,int ldc) {
        const T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(host_blas,left,right,output_rows,
            output_columns,shared,&one,a,lda,b,ldb,&zero,out,ldc));
    }
    void gemv(cublasOperation_t operation,int matrix_rows,int matrix_columns,
              const T* matrix,const T* vector,T* out) {
        require_blas(Blas<T>::gemv(device_blas,operation,matrix_rows,
            matrix_columns,scalars,matrix,matrix_rows,vector,scalars+2,out));
    }
    void normalize(T* vector,int size) {
        require_blas(Blas<T>::norm(device_blas,size,vector,scalars+3));
        reciprocal_device<<<1,1,0,stream>>>(scalars+3,scalars+4,invalid,true);
        require_blas(Blas<T>::scale(device_blas,size,scalars+4,vector));
    }
    void reserve_prediction(int rows_count) {
        if(rows_count<=prediction_capacity)return;
        T* next=nullptr;allocate(next,rows_count);
        cudaFree(prediction_score);prediction_score=next;
        prediction_capacity=rows_count;
    }
    void apply_filters(T* x,int rows_count) {
        reserve_prediction(rows_count);
        for(int component=0;component<north;++component) {
            require_blas(Blas<T>::gemv(device_blas,CUBLAS_OP_N,rows_count,p,
                scalars,x,rows_count,W+size_t(component)*p,scalars+2,
                prediction_score));
            require_blas(Blas<T>::ger(device_blas,rows_count,p,scalars+1,
                prediction_score,P+size_t(component)*p,x));
        }
    }
public:
    ResidentOpls(int n_,int p_,int q_,int a,int oversample_,int power_,
                 cudaStream_t stream_,bool retain_scores=true,
                 bool classification_=false,int north_=1,int=0,T=T(0),
                 int=0,T=T(0))
      :n(n_),p(p_),q(q_),components(a),north(std::max(0,north_)),
       oversample(oversample_),power(power_),stream(stream_),
       classification(classification_) {
        if(n<2||p<1||q<1||components<1||components>std::min(n-1,p))
            throw std::invalid_argument("invalid resident CUDA OPLS dimensions");
        try {
            require_blas(cublasCreate(&host_blas));
            require_blas(cublasSetStream(host_blas,stream));
            configure_blas_math<T>(host_blas);
            require_blas(cublasCreate(&device_blas));
            require_blas(cublasSetStream(device_blas,stream));
            require_blas(cublasSetPointerMode(device_blas,CUBLAS_POINTER_MODE_DEVICE));
            configure_blas_math<T>(device_blas);
            require_solver(cusolverDnCreate(&solver_handle));
            require_solver(cusolverDnSetStream(solver_handle,stream));
            require_random(curandCreateGenerator(&rng_handle,CURAND_RNG_PSEUDO_DEFAULT));
            require_random(curandSetStream(rng_handle,stream));
            allocate(filtered,size_t(n)*p);allocate(crosscov,size_t(p)*q);
            allocate(meanX,p);allocate(scaleX,p);allocate(meanY,q);allocate(scaleY,q);
            allocate(W,size_t(p)*std::max(1,north));allocate(P,size_t(p)*std::max(1,north));
            allocate(candidate,p);allocate(score,n);allocate(loading,p);
            allocate(orthogonal,p);allocate(scalars,5);allocate(invalid,1);
            allocate(effectiveR,size_t(p)*components);
            component_constants<<<1,1,0,stream>>>(scalars);
            direction.reset(new RsvdWorkspace<T>(p,q,1,oversample,power,
                stream,0,0,host_blas,solver_handle,rng_handle));
            inner.reset(new ResidentSimpls<T>(n,p,q,components,oversample,
                power,stream,retain_scores,classification,0,0,T(0),0,T(0),
                false));
        } catch(...) {release();throw;}
    }
    ~ResidentOpls(){release();}
    void fit(const T* hostX,const T* hostY,const int* hostLabels,int scaling,
             unsigned long long seed) {
        if((hostY==nullptr)==(hostLabels==nullptr))
            throw std::invalid_argument("provide responses or labels, not both");
        require_cuda(cudaMemsetAsync(invalid,0,sizeof(int),stream));
        require_cuda(cudaMemcpyAsync(filtered,hostX,size_t(n)*p*sizeof(T),
                                     cudaMemcpyHostToDevice,stream));
        require_cuda(preprocess(filtered,n,p,scaling,meanX,scaleX,stream));
        if(hostLabels) {
            allocate(labels,n);allocate(keys,n);allocate(rows,n);allocate(offsets,q+1);
            require_cuda(cudaMemcpyAsync(labels,hostLabels,n*sizeof(int),
                                         cudaMemcpyHostToDevice,stream));
            require_cuda(prepare_labels(labels,n,q,keys,rows,offsets,meanY,
                                        invalid,stream));
        } else {
            allocate(response,size_t(n)*q);
            require_cuda(cudaMemcpyAsync(response,hostY,size_t(n)*q*sizeof(T),
                                         cudaMemcpyHostToDevice,stream));
            require_cuda(preprocess(response,n,q,1,meanY,scaleY,stream));
        }
        for(int component=0;component<north;++component) {
            if(hostLabels) {
                require_cuda(class_product(filtered,n,p,q,rows,offsets,meanY,
                                           crosscov,stream));
            } else {
                gemm(CUBLAS_OP_T,CUBLAS_OP_N,p,q,n,filtered,n,response,n,
                     crosscov,p);
            }
            direction->solve(crosscov,seed+component,candidate,nullptr,nullptr,
                             invalid);
            normalize(candidate,p);
            gemv(CUBLAS_OP_N,n,p,filtered,candidate,score);
            require_blas(Blas<T>::dot(device_blas,n,score,score,scalars+3));
            reciprocal_device<<<1,1,0,stream>>>(scalars+3,scalars+4,invalid,false);
            gemv(CUBLAS_OP_T,n,p,filtered,score,loading);
            require_blas(Blas<T>::scale(device_blas,p,scalars+4,loading));
            require_blas(Blas<T>::dot(device_blas,p,candidate,loading,scalars+3));
            require_blas(Blas<T>::dot(device_blas,p,candidate,candidate,scalars+4));
            scalar_ratio<<<1,1,0,stream>>>(scalars+3,scalars+4,scalars+3,invalid);
            opls_difference<<<std::min(256,(p+255)/256),256,0,stream>>>(
                orthogonal,loading,candidate,scalars+3,p);
            normalize(orthogonal,p);
            gemv(CUBLAS_OP_N,n,p,filtered,orthogonal,score);
            require_blas(Blas<T>::dot(device_blas,n,score,score,scalars+3));
            reciprocal_device<<<1,1,0,stream>>>(scalars+3,scalars+4,invalid,false);
            gemv(CUBLAS_OP_T,n,p,filtered,score,loading);
            require_blas(Blas<T>::scale(device_blas,p,scalars+4,loading));
            require_cuda(cudaMemcpyAsync(W+size_t(component)*p,orthogonal,
                p*sizeof(T),cudaMemcpyDeviceToDevice,stream));
            require_cuda(cudaMemcpyAsync(P+size_t(component)*p,loading,
                p*sizeof(T),cudaMemcpyDeviceToDevice,stream));
            require_blas(Blas<T>::ger(device_blas,n,p,scalars+1,score,loading,
                                      filtered));
        }
        // The filtering operator is no longer needed. Release it before the
        // predictive SIMPLS core chooses its explicit/implicit device route.
        cudaFree(response);response=nullptr;
        cudaFree(crosscov);crosscov=nullptr;
        direction.reset();
        inner->adopt_device_predictors(filtered,hostY,hostLabels,3,seed,true);
        require_cuda(cudaMemcpyAsync(effectiveR,inner->weights(),
            size_t(p)*components*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        for(int component=north-1;component>=0;--component) {
            require_blas(Blas<T>::gemv(device_blas,CUBLAS_OP_T,p,components,
                scalars,effectiveR,p,P+size_t(component)*p,scalars+2,
                loading));
            require_blas(Blas<T>::ger(device_blas,p,components,scalars+1,
                W+size_t(component)*p,loading,effectiveR));
        }
        int bad=0;require_cuda(cudaMemcpyAsync(&bad,invalid,sizeof(int),
                                               cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
        if(bad)throw std::runtime_error("resident CUDA OPLS filtering failed");
    }
    void standardize_device(T* test,int rows_count) {
        standardize<<<256,256,0,stream>>>(test,size_t(rows_count)*p,rows_count,
                                         meanX,scaleX);
        require_cuda(cudaGetLastError());apply_filters(test,rows_count);
    }
    void project_standardized_device(const T* test,int rows_count,int prefix,
                                     T* scores) {
        inner->project_standardized_device(test,rows_count,prefix,scores);
    }
    void project_device(T* test,int rows_count,int prefix,T* scores) {
        standardize_device(test,rows_count);
        project_standardized_device(test,rows_count,prefix,scores);
    }
    void predict_increment_device(const T* scores,int rows_count,int first,
                                  int prefix,T* predictions) {
        inner->predict_increment_device(scores,rows_count,first,prefix,predictions);
    }
    void predict_projected_device(const T* scores,int rows_count,int prefix,
                                  T* predictions) {
        inner->predict_projected_device(scores,rows_count,prefix,predictions);
    }
    void predict_device(T* test,int rows_count,int prefix,T* scores,T* predictions) {
        project_device(test,rows_count,prefix,scores);
        predict_projected_device(scores,rows_count,prefix,predictions);
    }
    const T* weights()const{return effectiveR;}
    const T* loadings()const{return inner->loadings();}
    const T* training_scores()const{return inner->training_scores();}
    const T* standardized_predictors()const{return inner->standardized_predictors();}
    const int* label_rows()const{return inner->label_rows();}
    const int* label_offsets()const{return inner->label_offsets();}
    const T* class_priors()const{return inner->class_priors();}
    int solver_oversample()const{return inner->solver_oversample();}
    int solver_power()const{return inner->solver_power();}
    int solver_block()const{return inner->solver_block();}
    int solver_block_limit()const{return inner->solver_block_limit();}
    bool implicit_operator()const{return inner->implicit_operator();}
    bool predictor_crossprod_cache()const{return inner->predictor_crossprod_cache();}
    bool has_lda_moments()const{return inner->has_lda_moments();}
    void prepare_lda_moments(){inner->prepare_lda_moments();}
    const T* lda_gram()const{return inner->lda_gram();}
    const T* lda_sums()const{return inner->lda_sums();}
    void compact_training(){inner->compact_training();}
    const T* exported_field(int field)const{
        if(field==0)return effectiveR;
        if(field==3)return meanX;
        if(field==4)return scaleX;
        return inner->exported_field(field);
    }
    int input_columns()const{return p;}
    int feature_columns()const{return p;}
};

template<class T> class ResidentKernelPls {
    int n,p,q,components,kind,degree,oversample,power;
    T gamma,coefficient;
    cudaStream_t stream;
    cublasHandle_t blas=nullptr;
    T *reference=nullptr,*meanX=nullptr,*scaleX=nullptr,*reference_norms=nullptr,
      *training_means=nullptr,*grand=nullptr,*test_kernel=nullptr,
      *test_norms=nullptr,*test_means=nullptr;
    int test_capacity=0;
    std::unique_ptr<ResidentSimpls<T>> inner;
    template<class U> void allocate(U*& value,size_t count) {
        require_cuda(cudaMalloc(&value,count*sizeof(U)));
    }
    void release() noexcept {
        inner.reset();
        for(T* value:{reference,meanX,scaleX,reference_norms,training_means,
                      grand,test_kernel,test_norms,test_means})cudaFree(value);
        if(blas)cublasDestroy(blas);
    }
    void product(const T* left,int rows,const T* right,int right_rows,T* out) {
        const T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,
            rows,right_rows,p,&one,left,rows,right,right_rows,&zero,out,rows));
    }
    void transform(T* matrix,int rows,const T* left_norms) {
        const size_t count=size_t(rows)*n;
        transform_kernel<<<std::min<size_t>(65535,(count+255)/256),256,0,stream>>>(
            matrix,rows,n,kind,gamma,degree,coefficient,left_norms,
            reference_norms);
        require_cuda(cudaGetLastError());
    }
    void reserve_test(int rows) {
        if(rows<=test_capacity)return;
        size_t free_bytes=0,total_bytes=0;
        require_cuda(cudaMemGetInfo(&free_bytes,&total_bytes));
        const size_t required=size_t(rows)*n*sizeof(T)+size_t(rows)*2*sizeof(T);
        if(required>free_bytes*3/5)
            throw std::runtime_error("nonlinear kernel prediction exceeds the guarded CUDA workspace; reduce the prediction block size or training sample count");
        T *next_kernel=nullptr,*next_norms=nullptr,*next_means=nullptr;
        try {allocate(next_kernel,size_t(rows)*n);allocate(next_norms,rows);
             allocate(next_means,rows);} catch(...) {
            cudaFree(next_kernel);cudaFree(next_norms);cudaFree(next_means);throw;
        }
        cudaFree(test_kernel);cudaFree(test_norms);cudaFree(test_means);
        test_kernel=next_kernel;test_norms=next_norms;test_means=next_means;
        test_capacity=rows;
    }
    void build_test_kernel(const T* test,int rows) {
        reserve_test(rows);
        row_squared_norms<<<rows,256,0,stream>>>(test,rows,p,test_norms);
        require_cuda(cudaGetLastError());
        product(test,rows,reference,n,test_kernel);
        transform(test_kernel,rows,test_norms);
        row_means<<<rows,256,0,stream>>>(test_kernel,rows,n,test_means);
        require_cuda(cudaGetLastError());
        const size_t count=size_t(rows)*n;
        center_test_kernel<<<std::min<size_t>(65535,(count+255)/256),256,0,
                             stream>>>(test_kernel,rows,n,test_means,
                                       training_means,grand);
        require_cuda(cudaGetLastError());
    }
public:
    ResidentKernelPls(int n_,int p_,int q_,int a,int oversample_,int power_,
                      cudaStream_t stream_,bool retain_scores=true,
                      bool classification=false,int=0,int kind_=2,
                      T gamma_=T(1),int degree_=3,T coefficient_=T(1))
      :n(n_),p(p_),q(q_),components(a),kind(kind_),degree(degree_),
       oversample(oversample_),power(power_),gamma(gamma_),
       coefficient(coefficient_),stream(stream_) {
        if(n<2||p<1||q<1||components<1||components>n-1||
           (kind!=2&&kind!=3)||degree<1||!(gamma>T(0)))
            throw std::invalid_argument("invalid resident CUDA nonlinear kernel PLS dimensions or controls");
        size_t free_bytes=0,total_bytes=0;
        require_cuda(cudaMemGetInfo(&free_bytes,&total_bytes));
        const size_t gram_bytes=size_t(n)*n*sizeof(T);
        if(n>0&&gram_bytes/sizeof(T)/size_t(n)!=size_t(n))
            throw std::overflow_error("nonlinear kernel Gram matrix size overflow");
        if(gram_bytes+size_t(n)*p*sizeof(T)>free_bytes*3/5)
            throw std::runtime_error("nonlinear kernel PLS requires an n-by-n Gram matrix that exceeds the guarded CUDA memory budget");
        try {
            require_blas(cublasCreate(&blas));require_blas(cublasSetStream(blas,stream));
            configure_blas_math<T>(blas);
            allocate(reference,size_t(n)*p);allocate(meanX,p);allocate(scaleX,p);
            allocate(reference_norms,n);allocate(training_means,n);allocate(grand,1);
            inner.reset(new ResidentSimpls<T>(n,n,q,components,oversample,
                power,stream,retain_scores,classification,0,0,T(0),0,T(0),
                false));
        } catch(...) {release();throw;}
    }
    ~ResidentKernelPls(){release();}
    void fit(const T* hostX,const T* hostY,const int* hostLabels,int scaling,
             unsigned long long seed) {
        if((hostY==nullptr)==(hostLabels==nullptr))
            throw std::invalid_argument("provide responses or labels, not both");
        require_cuda(cudaMemcpyAsync(reference,hostX,size_t(n)*p*sizeof(T),
                                     cudaMemcpyHostToDevice,stream));
        require_cuda(preprocess(reference,n,p,scaling,meanX,scaleX,stream));
        row_squared_norms<<<n,256,0,stream>>>(reference,n,p,reference_norms);
        require_cuda(cudaGetLastError());
        T* kernel=nullptr;allocate(kernel,size_t(n)*n);
        try {
            product(reference,n,reference,n,kernel);
            transform(kernel,n,reference_norms);
            column_means<<<n,256,0,stream>>>(kernel,n,n,training_means);
            vector_mean<<<1,256,0,stream>>>(training_means,n,grand);
            const size_t count=size_t(n)*n;
            center_training_kernel<<<std::min<size_t>(65535,(count+255)/256),
                                     256,0,stream>>>(kernel,n,training_means,grand);
            require_cuda(cudaGetLastError());
            inner->adopt_device_predictors(kernel,hostY,hostLabels,3,seed,true);
        } catch(...) {cudaFree(kernel);throw;}
    }
    void standardize_device(T* test,int rows) {
        standardize<<<256,256,0,stream>>>(test,size_t(rows)*p,rows,meanX,scaleX);
        require_cuda(cudaGetLastError());
    }
    void project_standardized_device(const T* test,int rows,int prefix,T* scores) {
        build_test_kernel(test,rows);
        inner->project_standardized_device(test_kernel,rows,prefix,scores);
    }
    void project_device(T* test,int rows,int prefix,T* scores) {
        standardize_device(test,rows);project_standardized_device(test,rows,prefix,scores);
    }
    void predict_increment_device(const T* scores,int rows,int first,int prefix,
                                  T* predictions) {
        inner->predict_increment_device(scores,rows,first,prefix,predictions);
    }
    void predict_projected_device(const T* scores,int rows,int prefix,
                                  T* predictions) {
        inner->predict_projected_device(scores,rows,prefix,predictions);
    }
    void predict_device(T* test,int rows,int prefix,T* scores,T* predictions) {
        project_device(test,rows,prefix,scores);
        predict_projected_device(scores,rows,prefix,predictions);
    }
    const T* weights()const{return inner->weights();}
    const T* loadings()const{return inner->loadings();}
    const T* training_scores()const{return inner->training_scores();}
    const T* standardized_predictors()const{return inner->standardized_predictors();}
    const int* label_rows()const{return inner->label_rows();}
    const int* label_offsets()const{return inner->label_offsets();}
    const T* class_priors()const{return inner->class_priors();}
    int solver_oversample()const{return inner->solver_oversample();}
    int solver_power()const{return inner->solver_power();}
    int solver_block()const{return inner->solver_block();}
    int solver_block_limit()const{return inner->solver_block_limit();}
    bool implicit_operator()const{return inner->implicit_operator();}
    bool predictor_crossprod_cache()const{return inner->predictor_crossprod_cache();}
    bool has_lda_moments()const{return inner->has_lda_moments();}
    void prepare_lda_moments(){inner->prepare_lda_moments();}
    const T* lda_gram()const{return inner->lda_gram();}
    const T* lda_sums()const{return inner->lda_sums();}
    void compact_training(){inner->compact_training();}
    const T* exported_field(int field)const{
        if(field==3)return meanX;
        if(field==4)return scaleX;
        return inner->exported_field(field);
    }
    int input_columns()const{return p;}
    int feature_columns()const{return n;}
};

} // namespace fastpls_device
#endif
