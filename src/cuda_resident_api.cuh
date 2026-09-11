// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#ifndef FASTPLS_CUDA_RESIDENT_API_CUH
#define FASTPLS_CUDA_RESIDENT_API_CUH
#include "cuda_resident_api.h"
#include "cuda_resident_simpls.cuh"
#include "cuda_resident_plssvd.cuh"
#include "cuda_resident_special.cuh"
#include "cuda_resident_lda.cuh"
#include "cuda_resident_variance.cuh"
#include "cuda_resident_metrics.cuh"
#include <cstdio>
#include <memory>

namespace fastpls_device {
template<class T> __global__ void resident_topk(const T* values,int rows,int classes,int top,int* result){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows)return;
    for(int rank=0;rank<top;++rank){
        int best=-1;T value=0;
        for(int c=0;c<classes;++c){
            bool used=false;for(int k=0;k<rank;++k)if(result[size_t(k)*rows+i]==c+1)used=true;
            T candidate=values[size_t(c)*rows+i];
            if(!used&&(best<0||candidate>value)){best=c;value=candidate;}
        }
        result[size_t(rank)*rows+i]=best+1;
    }
}
template<class T,int MaximumTop> __global__ void resident_topk_one_pass(
    const T* values,int rows,int classes,int top,int* result) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)return;
    T best_values[MaximumTop];
    int best_classes[MaximumTop];
    for(int rank=0;rank<top;++rank) {
        best_values[rank]=-std::numeric_limits<T>::infinity();
        best_classes[rank]=-1;
    }
    for(int c=0;c<classes;++c) {
        const T value=values[size_t(c)*rows+row];
        if(value<=best_values[top-1])continue;
        int position=top-1;
        while(position>0&&value>best_values[position-1]) {
            best_values[position]=best_values[position-1];
            best_classes[position]=best_classes[position-1];
            --position;
        }
        best_values[position]=value;
        best_classes[position]=c+1;
    }
    for(int rank=0;rank<top;++rank)
        result[size_t(rank)*rows+row]=best_classes[rank];
}
class ResidentHandle {
public:
    virtual ~ResidentHandle()=default;
    virtual void predict(const void*,int,int,void*,bool)=0;
    virtual void predict_path(const void*,int,const int*,int,bool,void*)=0;
    virtual void export_field(int,void*,size_t)=0;
    virtual void classify(const void*,int,int,bool,int,int*)=0;
    virtual void classify_path(const void*,int,const int*,int,bool,int,int*)=0;
    virtual void classify_response_path(
        const void*,int,const int*,int,bool,int,int*,void*)=0;
    virtual void response_sums(const void*,const void*,const int*,int,int,void*)=0;
    virtual void project(const void*,int,int,void*)=0;
    virtual void controls(int*,int*,int*,int*,int*,int*)=0;
    virtual void compact(bool)=0;
};
template<class T,template<class> class Fit=ResidentSimpls> class TypedResidentHandle final:public ResidentHandle {
    cudaStream_t stream=nullptr;
    std::unique_ptr<Fit<T>> model;
    std::unique_ptr<ResidentLda<T>> lda;
    std::unique_ptr<VarianceWorkspace<T>> variance;
    T *x=nullptr,*scores=nullptr,*predictions=nullptr;
    T *observed=nullptr,*metric_sums=nullptr;
    int *observed_labels=nullptr,*metric_invalid=nullptr;
    size_t observed_capacity=0,label_capacity=0;
    int* top_indices=nullptr;size_t top_capacity=0;
    int p,feature_p,q,a,training_n,capacity=0;
    bool classification;
    void prepare_lda_workspace() {
        if(lda)return;
        if(!classification)
            throw std::invalid_argument("LDA requires class labels");
        if(model->has_lda_moments()) {
            model->prepare_lda_moments();
            lda.reset(new ResidentLda<T>(
                model->lda_gram(),model->lda_sums(),training_n,a,q,
                model->label_offsets(),model->class_priors(),stream));
        } else {
            lda.reset(new ResidentLda<T>(
                model->training_scores(),training_n,a,q,
                model->label_rows(),model->label_offsets(),
                model->class_priors(),stream));
        }
    }
    void release() noexcept {
        variance.reset();lda.reset();model.reset();cudaFree(x);cudaFree(scores);cudaFree(predictions);
        cudaFree(top_indices);
        cudaFree(observed);cudaFree(metric_sums);cudaFree(observed_labels);cudaFree(metric_invalid);
        if(stream)cudaStreamDestroy(stream);
        x=scores=predictions=nullptr;stream=nullptr;
    }
    void reserve(int rows) {
        if(rows<=capacity)return;
        // Allocate replacement buffers before releasing the previous workspace.
        T *nx=nullptr,*ns=nullptr,*np=nullptr;
        try {
            require_cuda(cudaMalloc(&nx,size_t(rows)*p*sizeof(T)));
            require_cuda(cudaMalloc(&ns,size_t(rows)*a*sizeof(T)));
            require_cuda(cudaMalloc(&np,size_t(rows)*q*sizeof(T)));
        } catch(...) {cudaFree(nx);cudaFree(ns);cudaFree(np);throw;}
        cudaFree(x);cudaFree(scores);cudaFree(predictions);
        x=nx;scores=ns;predictions=np;capacity=rows;
    }
    void execute_resident(int rows,int prefix,bool use_lda) {
        if(use_lda){
            prepare_lda_workspace();
            model->project_device(x,rows,prefix,scores);
            lda->predict(scores,rows,prefix,predictions);
        }else model->predict_device(x,rows,prefix,scores,predictions);
    }
    void execute_classification_path(const T* input,int rows,
                                     const int* prefixes,int prefix_count,
                                     bool use_lda,int top,int* out) {
        if(prefix_count<1)throw std::invalid_argument("empty component path");
        int previous=0;
        for(int j=0;j<prefix_count;++j) {
            if(prefixes[j]<=previous||prefixes[j]>a)
                throw std::invalid_argument("component path must be strictly increasing");
            previous=prefixes[j];
        }
        require_cuda(cudaMemcpyAsync(
            x,input,size_t(rows)*p*sizeof(T),cudaMemcpyHostToDevice,stream));
        model->standardize_device(x,rows);
        model->project_standardized_device(
            x,rows,prefixes[prefix_count-1],scores);
        previous=0;
        for(int j=0;j<prefix_count;++j) {
            const int prefix=prefixes[j];
            if(use_lda) {
                prepare_lda_workspace();
                lda->predict(scores,rows,prefix,predictions);
            } else {
                model->predict_increment_device(
                    scores,rows,previous,prefix,predictions);
            }
            launch_topk(rows,top);
            require_cuda(cudaMemcpyAsync(
                out+size_t(j)*rows*top,top_indices,
                size_t(rows)*top*sizeof(int),cudaMemcpyDeviceToHost,stream));
            previous=prefix;
        }
        require_cuda(cudaStreamSynchronize(stream));
    }
    void reserve_top(size_t size) {
        if(size<=top_capacity)return;
        int* next=nullptr;require_cuda(cudaMalloc(&next,size*sizeof(int)));
        cudaFree(top_indices);top_indices=next;top_capacity=size;
    }
    void launch_topk(int rows,int top) {
        if(top<=10) {
            resident_topk_one_pass<T,10><<<(rows+255)/256,256,0,stream>>>(
                predictions,rows,q,top,top_indices);
        } else {
            resident_topk<<<(rows+255)/256,256,0,stream>>>(
                predictions,rows,q,top,top_indices);
        }
        require_cuda(cudaGetLastError());
    }
public:
    TypedResidentHandle(const void* hx,const void* hy,const int* labels,int n,
                         int p_,int q_,int a_,int scaling,int oversample,
                         int power,bool retain_scores,unsigned long long seed,
                         int north=0,int kernel=0,double gamma=0.0,
                         int degree=0,double coefficient=0.0)
      :p(p_),feature_p(p_),q(q_),a(a_),training_n(n),
       classification(labels!=nullptr) {
        try {
            require_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            model.reset(new Fit<T>(n,p,q,a,oversample,power,stream,
                                   retain_scores,classification,north,kernel,
                                   static_cast<T>(gamma),degree,
                                   static_cast<T>(coefficient)));
            feature_p=model->feature_columns();
            model->fit(static_cast<const T*>(hx),static_cast<const T*>(hy),labels,scaling,seed);
        } catch(...) {release();throw;}
    }
    ~TypedResidentHandle(){release();}
    void export_field(int field,void* out,size_t size)override{
        const size_t counts[]={size_t(feature_p)*a,size_t(q)*a,
            size_t(training_n)*a,size_t(p),size_t(p),size_t(q),
            size_t(feature_p)*a,size_t(a)+1};
        if(field<0||field>7||!out||size!=counts[field])throw std::invalid_argument("invalid resident field output size");
        const T* source=nullptr;
        if(field<6)source=model->exported_field(field);
        else {
            if(!variance){
                auto next=std::make_unique<VarianceWorkspace<T>>(
                    training_n,feature_p,a,stream);
                next->compute(model->standardized_predictors(),model->training_scores());
                require_cuda(cudaStreamSynchronize(stream));
                variance=std::move(next);
            }
            source=field==6?variance->predictor_loadings():variance->sums_of_squares();
        }
        require_cuda(cudaMemcpyAsync(out,source,size*sizeof(T),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
    }
    void execute(const void* input,int rows,int prefix,bool use_lda) {
        if(!input||rows<1||prefix<1||prefix>a)throw std::invalid_argument("invalid resident prediction input");
        if(use_lda&&!classification)throw std::invalid_argument("LDA requires class labels");
        reserve(rows);
        require_cuda(cudaMemcpyAsync(x,input,size_t(rows)*p*sizeof(T),cudaMemcpyHostToDevice,stream));
        execute_resident(rows,prefix,use_lda);
    }
    void predict(const void* input,int rows,int prefix,void* out,bool use_lda) override {
        if(!out)throw std::invalid_argument("null prediction output");
        execute(input,rows,prefix,use_lda);
        require_cuda(cudaMemcpyAsync(out,predictions,size_t(rows)*q*sizeof(T),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
    }
    void predict_path(const void* input,int rows,const int* prefixes,
                      int prefix_count,bool use_lda,void* out) override {
        if(!input||!out||rows<1||prefix_count<1)
            throw std::invalid_argument("invalid resident prediction-path request");
        int previous=0;
        for(int j=0;j<prefix_count;++j) {
            if(prefixes[j]<=previous||prefixes[j]>a)
                throw std::invalid_argument("component path must be strictly increasing");
            previous=prefixes[j];
        }
        if(use_lda&&!classification)
            throw std::invalid_argument("LDA requires class labels");
        reserve(rows);
        require_cuda(cudaMemcpyAsync(
            x,input,size_t(rows)*p*sizeof(T),cudaMemcpyHostToDevice,stream));
        model->standardize_device(x,rows);
        model->project_standardized_device(
            x,rows,prefixes[prefix_count-1],scores);
        previous=0;
        T* output=static_cast<T*>(out);
        for(int j=0;j<prefix_count;++j) {
            const int prefix=prefixes[j];
            if(use_lda) {
                prepare_lda_workspace();
                lda->predict(scores,rows,prefix,predictions);
            } else {
                model->predict_projected_device(
                    scores,rows,prefix,predictions);
            }
            require_cuda(cudaMemcpyAsync(
                output+size_t(j)*rows*q,predictions,
                size_t(rows)*q*sizeof(T),cudaMemcpyDeviceToHost,stream));
            previous=prefix;
        }
        require_cuda(cudaStreamSynchronize(stream));
    }
    void classify(const void* input,int rows,int prefix,bool use_lda,int top,int* out)override{
        if(!classification||!out||top<1||top>q)throw std::invalid_argument("invalid resident classification request");
        size_t free_bytes=0,total_bytes=0;
        require_cuda(cudaMemGetInfo(&free_bytes,&total_bytes));
        (void)total_bytes;
        const size_t bytes_per_row=
            size_t(p+a+q)*sizeof(T)+size_t(top)*sizeof(int);
        const size_t full_workspace=size_t(rows)*bytes_per_row;
        const bool stream_blocks=full_workspace>free_bytes*2/5;
        if(!stream_blocks) {
            execute(input,rows,prefix,use_lda);
            const size_t size=size_t(rows)*top;
            reserve_top(size);launch_topk(rows,top);
            require_cuda(cudaMemcpyAsync(out,top_indices,size*sizeof(int),
                                         cudaMemcpyDeviceToHost,stream));
            require_cuda(cudaStreamSynchronize(stream));
            return;
        }
        const size_t target_bytes=std::max<size_t>(
            bytes_per_row*1024,free_bytes*3/20);
        const int block_rows=std::max(1024,std::min(
            rows,std::min(65536,int(target_bytes/bytes_per_row))));
        const T* source=static_cast<const T*>(input);
        reserve(std::min(rows,block_rows));
        reserve_top(size_t(std::min(rows,block_rows))*top);
        for(int start=0;start<rows;start+=block_rows) {
            const int count=std::min(block_rows,rows-start);
            require_cuda(cudaMemcpy2DAsync(
                x,size_t(count)*sizeof(T),source+start,size_t(rows)*sizeof(T),
                size_t(count)*sizeof(T),p,cudaMemcpyHostToDevice,stream));
            execute_resident(count,prefix,use_lda);
            launch_topk(count,top);
            require_cuda(cudaMemcpy2DAsync(
                out+start,size_t(rows)*sizeof(int),top_indices,
                size_t(count)*sizeof(int),size_t(count)*sizeof(int),top,
                cudaMemcpyDeviceToHost,stream));
            require_cuda(cudaStreamSynchronize(stream));
        }
    }
    void classify_path(const void* input,int rows,const int* prefixes,
                       int prefix_count,bool use_lda,int top,int* out)override{
        if(!classification||!input||!out||prefix_count<1||top<1||top>q)
            throw std::invalid_argument("invalid resident classification-path request");
        const int maximum_prefix=prefixes[prefix_count-1];
        if(maximum_prefix<1||maximum_prefix>a)
            throw std::invalid_argument("invalid resident classification-path prefix");
        size_t free_bytes=0,total_bytes=0;
        require_cuda(cudaMemGetInfo(&free_bytes,&total_bytes));
        (void)total_bytes;
        const size_t bytes_per_row=size_t(p+a+q)*sizeof(T)+
            size_t(top)*sizeof(int);
        const size_t full_workspace=size_t(rows)*bytes_per_row;
        const bool stream_blocks=full_workspace>free_bytes*2/5;
        if(!stream_blocks) {
            reserve(rows);reserve_top(size_t(rows)*top);
            execute_classification_path(static_cast<const T*>(input),rows,
                prefixes,prefix_count,use_lda,top,out);
            return;
        }
        const size_t target_bytes=std::max<size_t>(
            bytes_per_row*1024,free_bytes*3/20);
        const int block_rows=std::max(1024,std::min(
            rows,std::min(65536,int(target_bytes/bytes_per_row))));
        const T* source=static_cast<const T*>(input);
        reserve(std::min(rows,block_rows));
        reserve_top(size_t(std::min(rows,block_rows))*top);
        for(int start=0;start<rows;start+=block_rows) {
            const int count=std::min(block_rows,rows-start);
            require_cuda(cudaMemcpy2DAsync(
                x,size_t(count)*sizeof(T),source+start,size_t(rows)*sizeof(T),
                size_t(count)*sizeof(T),p,cudaMemcpyHostToDevice,stream));
            model->standardize_device(x,count);
            model->project_standardized_device(
                x,count,maximum_prefix,scores);
            int previous=0;
            for(int j=0;j<prefix_count;++j) {
                const int prefix=prefixes[j];
                if(prefix<=previous||prefix>maximum_prefix)
                    throw std::invalid_argument("component path must be strictly increasing");
                if(use_lda) {
                    if(!lda)lda.reset(new ResidentLda<T>(
                        model->training_scores(),training_n,a,q,
                        model->label_rows(),model->label_offsets(),
                        model->class_priors(),stream));
                    lda->predict(scores,count,prefix,predictions);
                } else {
                    model->predict_increment_device(
                        scores,count,previous,prefix,predictions);
                }
                launch_topk(count,top);
                require_cuda(cudaMemcpy2DAsync(
                    out+size_t(j)*rows*top+start,size_t(rows)*sizeof(int),
                    top_indices,size_t(count)*sizeof(int),
                    size_t(count)*sizeof(int),top,cudaMemcpyDeviceToHost,stream));
                previous=prefix;
            }
            require_cuda(cudaStreamSynchronize(stream));
        }
    }
    void classify_response_path(
        const void* input,int rows,const int* prefixes,int prefix_count,
        bool use_lda,int top,int* label_out,void* response_out)override{
        if(!classification||!input||!label_out||!response_out||
           prefix_count<1||top<1||top>q)
            throw std::invalid_argument(
                "invalid resident classification-response path request");
        int previous=0;
        for(int j=0;j<prefix_count;++j) {
            if(prefixes[j]<=previous||prefixes[j]>a)
                throw std::invalid_argument(
                    "component path must be strictly increasing");
            previous=prefixes[j];
        }
        const int maximum_prefix=prefixes[prefix_count-1];
        size_t free_bytes=0,total_bytes=0;
        require_cuda(cudaMemGetInfo(&free_bytes,&total_bytes));
        (void)total_bytes;
        const size_t bytes_per_row=size_t(p+a+q)*sizeof(T)+
            size_t(top)*sizeof(int);
        const size_t full_workspace=size_t(rows)*bytes_per_row;
        const bool stream_blocks=full_workspace>free_bytes*2/5;
        const int block_rows=stream_blocks?std::max(1024,std::min(
            rows,std::min(65536,int(std::max<size_t>(
                bytes_per_row*1024,free_bytes*3/20)/bytes_per_row)))):rows;
        const T* source=static_cast<const T*>(input);
        T* response=static_cast<T*>(response_out);
        reserve(block_rows);reserve_top(size_t(block_rows)*top);
        for(int start=0;start<rows;start+=block_rows) {
            const int count=std::min(block_rows,rows-start);
            if(start==0&&count==rows) {
                require_cuda(cudaMemcpyAsync(
                    x,source,size_t(rows)*p*sizeof(T),
                    cudaMemcpyHostToDevice,stream));
            } else {
                require_cuda(cudaMemcpy2DAsync(
                    x,size_t(count)*sizeof(T),source+start,
                    size_t(rows)*sizeof(T),size_t(count)*sizeof(T),p,
                    cudaMemcpyHostToDevice,stream));
            }
            model->standardize_device(x,count);
            model->project_standardized_device(
                x,count,maximum_prefix,scores);
            for(int j=0;j<prefix_count;++j) {
                const int prefix=prefixes[j];
                model->predict_projected_device(
                    scores,count,prefix,predictions);
                require_cuda(cudaMemcpy2DAsync(
                    response+size_t(j)*rows*q+start,
                    size_t(rows)*sizeof(T),predictions,
                    size_t(count)*sizeof(T),size_t(count)*sizeof(T),q,
                    cudaMemcpyDeviceToHost,stream));
                if(use_lda) {
                    prepare_lda_workspace();
                    lda->predict(scores,count,prefix,predictions);
                }
                launch_topk(count,top);
                require_cuda(cudaMemcpy2DAsync(
                    label_out+size_t(j)*rows*top+start,
                    size_t(rows)*sizeof(int),top_indices,
                    size_t(count)*sizeof(int),size_t(count)*sizeof(int),top,
                    cudaMemcpyDeviceToHost,stream));
            }
            require_cuda(cudaStreamSynchronize(stream));
        }
    }
    void project(const void* input,int rows,int prefix,void* out)override{
        if(!input||!out||rows<1||prefix<1||prefix>a)throw std::invalid_argument("invalid resident score request");
        reserve(rows);
        require_cuda(cudaMemcpyAsync(x,input,size_t(rows)*p*sizeof(T),cudaMemcpyHostToDevice,stream));
        model->project_device(x,rows,prefix,scores);
        require_cuda(cudaMemcpyAsync(out,scores,size_t(rows)*prefix*sizeof(T),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
    }
    void response_sums(const void* input,const void* y,const int* labels,int rows,int prefix,void* out)override{
        if(!out||(!y&&!labels)||(y&&labels)||(classification!=(labels!=nullptr)))
            throw std::invalid_argument("invalid resident response representation");
        execute(input,rows,prefix,false);
        if(!metric_sums)require_cuda(cudaMalloc(&metric_sums,size_t(3)*q*sizeof(T)));
        if(!metric_invalid)require_cuda(cudaMalloc(&metric_invalid,sizeof(int)));
        require_cuda(cudaMemsetAsync(metric_invalid,0,sizeof(int),stream));
        if(labels){
            if(size_t(rows)>label_capacity){
                int* next=nullptr;require_cuda(cudaMalloc(&next,size_t(rows)*sizeof(int)));
                cudaFree(observed_labels);observed_labels=next;label_capacity=rows;
            }
            require_cuda(cudaMemcpyAsync(observed_labels,labels,size_t(rows)*sizeof(int),cudaMemcpyHostToDevice,stream));
        }else{
            size_t count=size_t(rows)*q;
            if(count>observed_capacity){
                T* next=nullptr;require_cuda(cudaMalloc(&next,count*sizeof(T)));
                cudaFree(observed);observed=next;observed_capacity=count;
            }
            require_cuda(cudaMemcpyAsync(observed,y,count*sizeof(T),cudaMemcpyHostToDevice,stream));
        }
        fastpls_device::response_sums<<<std::min(q,65535),256,0,stream>>>(predictions,
            labels?nullptr:observed,labels?observed_labels:nullptr,model->exported_field(5),rows,q,metric_sums,metric_invalid);
        require_cuda(cudaGetLastError());
        int invalid=0;
        require_cuda(cudaMemcpyAsync(&invalid,metric_invalid,sizeof(int),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaMemcpyAsync(out,metric_sums,size_t(3)*q*sizeof(T),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
        if(invalid)throw std::runtime_error("nonfinite response/prediction or invalid class label in resident metrics");
    }
    void controls(int* oversample,int* power,int* block,int* block_limit,
                  int* implicit_operator,int* predictor_crossprod_cache)override{
        if(!oversample||!power||!block||!block_limit||!implicit_operator||
           !predictor_crossprod_cache)
            throw std::invalid_argument("null resident control output");
        *oversample=model->solver_oversample();*power=model->solver_power();*block=model->solver_block();
        *block_limit=model->solver_block_limit();
        *implicit_operator=model->implicit_operator()?1:0;
        *predictor_crossprod_cache=model->predictor_crossprod_cache()?1:0;
    }
    void compact(bool prepare_lda) override {
        if(prepare_lda) {
            if(!classification)throw std::invalid_argument("LDA compaction requires classification labels");
            prepare_lda_workspace();
        }
        model->compact_training();
    }
};
inline void resident_error(char* out,size_t size,const char* message) noexcept {
    if(out&&size)std::snprintf(out,size,"%s",message);
}
template<class T> void host_gemm(const T* left,const T* right,int rows,
                                 int inner,int columns,T* output) {
    cudaStream_t stream=nullptr;
    cublasHandle_t blas=nullptr;
    T *device_left=nullptr,*device_right=nullptr,*device_output=nullptr;
    auto release=[&]() noexcept {
        cudaFree(device_left);cudaFree(device_right);cudaFree(device_output);
        if(blas)cublasDestroy(blas);
        if(stream)cudaStreamDestroy(stream);
    };
    try {
        require_cuda(cudaStreamCreate(&stream));
        require_blas(cublasCreate(&blas));
        require_blas(cublasSetStream(blas,stream));
        configure_blas_math<T>(blas);
        const size_t left_size=size_t(rows)*inner;
        const size_t right_size=size_t(inner)*columns;
        const size_t output_size=size_t(rows)*columns;
        require_cuda(cudaMalloc(&device_left,left_size*sizeof(T)));
        require_cuda(cudaMalloc(&device_right,right_size*sizeof(T)));
        require_cuda(cudaMalloc(&device_output,output_size*sizeof(T)));
        require_cuda(cudaMemcpyAsync(device_left,left,left_size*sizeof(T),
            cudaMemcpyHostToDevice,stream));
        require_cuda(cudaMemcpyAsync(device_right,right,right_size*sizeof(T),
            cudaMemcpyHostToDevice,stream));
        const T one=1,zero=0;
        require_blas(Decomposition<T>::gemm(
            blas,CUBLAS_OP_N,CUBLAS_OP_N,rows,columns,inner,&one,
            device_left,rows,device_right,inner,&zero,device_output,rows));
        require_cuda(cudaMemcpyAsync(output,device_output,
            output_size*sizeof(T),cudaMemcpyDeviceToHost,stream));
        require_cuda(cudaStreamSynchronize(stream));
        release();
    } catch(...) {release();throw;}
}

template<class T>
__global__ void gather_matrix_rows(const T* source,int source_rows,int columns,
                                   const int* indices,int output_rows,
                                   T* output) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    const int column=blockIdx.y;
    if(row<output_rows&&column<columns)
        output[size_t(column)*output_rows+row]=
            source[size_t(column)*source_rows+indices[row]];
}

template<class T>
__global__ void subtract_crosscovariance(
    const T* full,T* heldout,size_t size) {
    for(size_t index=blockIdx.x*size_t(blockDim.x)+threadIdx.x;
        index<size;index+=size_t(blockDim.x)*gridDim.x)
        heldout[index]=full[index]-heldout[index];
}

template<class T,template<class> class Model,bool IncrementFoldSeed>
void resident_pls_cv_classification(
    const T* predictors,const int* labels,const int* folds,int n,int p,
    int classes,const int* prefixes,int prefix_count,int scaling,
    bool use_lda,int oversample,int power,unsigned long long seed,
    bool store_predictions,bool store_scores,int* prediction_output,
    T* score_output,int* status,double* metrics) {
    if(!predictors||!labels||!folds||!prefixes||!status||!metrics||n<2||
       p<1||classes<2||prefix_count<1||scaling<1||scaling>3)
        throw std::invalid_argument("invalid resident CUDA CV input");
    int maximum_prefix=0,fold_count=0;
    for(int j=0;j<prefix_count;++j) {
        if(prefixes[j]<=maximum_prefix)
            throw std::invalid_argument(
                "CUDA CV component path must be strictly increasing");
        maximum_prefix=prefixes[j];
    }
    for(int row=0;row<n;++row) {
        if(labels[row]<1||labels[row]>classes||folds[row]<1)
            throw std::invalid_argument("invalid CUDA CV label or fold");
        fold_count=std::max(fold_count,folds[row]);
    }
    if(maximum_prefix>std::min(n-1,p))
        throw std::invalid_argument("CUDA CV component count exceeds rank");
    if(store_predictions&&!prediction_output)
        throw std::invalid_argument("null CUDA CV prediction output");
    if(store_scores&&!score_output)
        throw std::invalid_argument("null CUDA CV score output");

    std::vector<std::vector<int>> test_rows(static_cast<size_t>(fold_count));
    for(int row=0;row<n;++row)
        test_rows[static_cast<size_t>(folds[row]-1)].push_back(row);
    int maximum_test=0,maximum_train=0;
    for(const auto& test:test_rows) {
        maximum_test=std::max(maximum_test,static_cast<int>(test.size()));
        maximum_train=std::max(maximum_train,n-static_cast<int>(test.size()));
    }
    if(maximum_test<1||maximum_train<2)
        throw std::invalid_argument("CUDA CV contains an empty train or test fold");

    cudaStream_t stream=nullptr;
    cublasHandle_t model_blas=nullptr,component_blas=nullptr;
    cusolverDnHandle_t model_solver=nullptr;
    curandGenerator_t model_rng=nullptr;
    T *full=nullptr,*train=nullptr,*test=nullptr,*test_scores=nullptr,
      *response_scores=nullptr,*discriminants=nullptr;
    int *train_indices=nullptr,*test_indices=nullptr,*top_indices=nullptr;
    auto release=[&]() noexcept {
        cudaFree(full);cudaFree(train);cudaFree(test);cudaFree(test_scores);
        cudaFree(response_scores);cudaFree(discriminants);
        cudaFree(train_indices);cudaFree(test_indices);cudaFree(top_indices);
        if(component_blas)cublasDestroy(component_blas);
        if(model_rng)curandDestroyGenerator(model_rng);
        if(model_solver)cusolverDnDestroy(model_solver);
        if(model_blas)cublasDestroy(model_blas);
        if(stream)cudaStreamDestroy(stream);
    };
    try {
        require_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        require_blas(cublasCreate(&model_blas));
        require_blas(cublasSetStream(model_blas,stream));
        configure_blas_math<T>(model_blas);
        require_blas(cublasCreate(&component_blas));
        require_blas(cublasSetStream(component_blas,stream));
        require_blas(cublasSetPointerMode(
            component_blas,CUBLAS_POINTER_MODE_DEVICE));
        require_solver(cusolverDnCreate(&model_solver));
        require_solver(cusolverDnSetStream(model_solver,stream));
        require_random(curandCreateGenerator(
            &model_rng,CURAND_RNG_PSEUDO_DEFAULT));
        require_random(curandSetStream(model_rng,stream));
        require_cuda(cudaMalloc(&full,size_t(n)*p*sizeof(T)));
        require_cuda(cudaMalloc(&train,size_t(maximum_train)*p*sizeof(T)));
        require_cuda(cudaMalloc(&test,size_t(maximum_test)*p*sizeof(T)));
        require_cuda(cudaMalloc(&test_scores,
                                size_t(maximum_test)*maximum_prefix*sizeof(T)));
        require_cuda(cudaMalloc(&response_scores,
                                size_t(maximum_test)*classes*sizeof(T)));
        require_cuda(cudaMalloc(&discriminants,
                                size_t(maximum_test)*classes*sizeof(T)));
        require_cuda(cudaMalloc(&train_indices,size_t(maximum_train)*sizeof(int)));
        require_cuda(cudaMalloc(&test_indices,size_t(maximum_test)*sizeof(int)));
        require_cuda(cudaMalloc(&top_indices,size_t(maximum_test)*sizeof(int)));
        require_cuda(cudaMemcpyAsync(full,predictors,size_t(n)*p*sizeof(T),
                                     cudaMemcpyHostToDevice,stream));
        std::fill(metrics,metrics+prefix_count,0.0);
        std::vector<double> metric_counts(static_cast<size_t>(prefix_count),0.0);
        if(store_scores)
            std::fill(score_output,
                      score_output+size_t(n)*classes*prefix_count,T(0));

        for(int fold=0;fold<fold_count;++fold) {
            const auto& heldout=test_rows[static_cast<size_t>(fold)];
            if(heldout.empty()) {status[fold]=2;continue;}
            std::vector<char> heldout_flag(static_cast<size_t>(n),0);
            for(const int row:heldout)heldout_flag[static_cast<size_t>(row)]=1;
            std::vector<int> training;
            training.reserve(static_cast<size_t>(n)-heldout.size());
            std::vector<int> counts(static_cast<size_t>(classes),0);
            for(int row=0;row<n;++row)if(!heldout_flag[static_cast<size_t>(row)]) {
                training.push_back(row);
                ++counts[static_cast<size_t>(labels[row]-1)];
            }
            std::vector<int> active;
            std::vector<int> active_map(static_cast<size_t>(classes),-1);
            for(int label=0;label<classes;++label)if(counts[static_cast<size_t>(label)]>0) {
                active_map[static_cast<size_t>(label)]=static_cast<int>(active.size());
                active.push_back(label+1);
            }
            if(active.size()<=1) {
                const int fallback=active.empty()?1:active.front();
                for(int prefix=0;prefix<prefix_count;++prefix) {
                    for(const int row:heldout) {
                        if(store_predictions)
                            prediction_output[size_t(prefix)*n+row]=fallback;
                        metrics[prefix]+=labels[row]==fallback?1.0:0.0;
                        metric_counts[static_cast<size_t>(prefix)]+=1.0;
                    }
                }
                status[fold]=4;continue;
            }
            const int train_n=static_cast<int>(training.size());
            const int test_n=static_cast<int>(heldout.size());
            if(maximum_prefix>std::min(train_n-1,p))
                throw std::invalid_argument(
                    "CUDA CV component count exceeds a fold rank");
            std::vector<int> compact_labels(static_cast<size_t>(train_n));
            for(int i=0;i<train_n;++i)
                compact_labels[static_cast<size_t>(i)]=
                    active_map[static_cast<size_t>(labels[training[static_cast<size_t>(i)]]-1)]+1;
            require_cuda(cudaMemcpyAsync(train_indices,training.data(),
                                         size_t(train_n)*sizeof(int),
                                         cudaMemcpyHostToDevice,stream));
            require_cuda(cudaMemcpyAsync(test_indices,heldout.data(),
                                         size_t(test_n)*sizeof(int),
                                         cudaMemcpyHostToDevice,stream));
            gather_matrix_rows<<<dim3((train_n+255)/256,p),256,0,stream>>>(
                full,n,p,train_indices,train_n,train);
            gather_matrix_rows<<<dim3((test_n+255)/256,p),256,0,stream>>>(
                full,n,p,test_indices,test_n,test);
            require_cuda(cudaGetLastError());

            {
                const unsigned long long fold_seed = seed +
                    (IncrementFoldSeed ? static_cast<unsigned long long>(fold) :
                     0ULL);
                Model<T> model(
                    train_n,p,static_cast<int>(active.size()),maximum_prefix,
                    oversample,power,stream,use_lda,true,0,0,T(0),0,T(0),false,
                    model_blas,model_solver,model_rng,component_blas);
                model.fit_borrowed_device_predictors(
                    train,nullptr,compact_labels.data(),scaling,fold_seed,false);
                model.standardize_device(test,test_n);
                model.project_standardized_device(
                    test,test_n,maximum_prefix,test_scores);
                std::unique_ptr<ResidentLda<T>> lda;
                if(use_lda) {
                    if(model.has_lda_moments()) {
                        model.prepare_lda_moments();
                        lda.reset(new ResidentLda<T>(
                            model.lda_gram(),model.lda_sums(),train_n,
                            maximum_prefix,static_cast<int>(active.size()),
                            model.label_offsets(),model.class_priors(),stream));
                    } else {
                        lda.reset(new ResidentLda<T>(
                            model.training_scores(),train_n,maximum_prefix,
                            static_cast<int>(active.size()),model.label_rows(),
                            model.label_offsets(),model.class_priors(),stream));
                    }
                }
                std::vector<int> host_labels(
                    size_t(test_n)*prefix_count);
                std::vector<T> host_scores;
                if(store_scores)host_scores.resize(
                    size_t(test_n)*active.size()*prefix_count);
                for(int prefix_index=0;prefix_index<prefix_count;++prefix_index) {
                    const int prefix=prefixes[prefix_index];
                    model.predict_projected_device(
                        test_scores,test_n,prefix,response_scores);
                    if(store_scores)
                        require_cuda(cudaMemcpyAsync(
                            host_scores.data()+size_t(prefix_index)*test_n*active.size(),
                            response_scores,size_t(test_n)*active.size()*sizeof(T),
                            cudaMemcpyDeviceToHost,stream));
                    T* decoded_scores=response_scores;
                    if(use_lda) {
                        lda->predict(test_scores,test_n,prefix,discriminants);
                        decoded_scores=discriminants;
                    }
                    resident_topk_one_pass<T,10><<<(test_n+255)/256,256,0,stream>>>(
                        decoded_scores,test_n,static_cast<int>(active.size()),1,
                        top_indices);
                    require_cuda(cudaGetLastError());
                    require_cuda(cudaMemcpyAsync(
                        host_labels.data()+size_t(prefix_index)*test_n,
                        top_indices,size_t(test_n)*sizeof(int),
                        cudaMemcpyDeviceToHost,stream));
                }
                require_cuda(cudaStreamSynchronize(stream));
                for(int prefix_index=0;prefix_index<prefix_count;++prefix_index) {
                    for(int i=0;i<test_n;++i) {
                        const int predicted=active[static_cast<size_t>(
                            host_labels[size_t(prefix_index)*test_n+i]-1)];
                        const int row=heldout[static_cast<size_t>(i)];
                        if(store_predictions)
                            prediction_output[size_t(prefix_index)*n+row]=predicted;
                        metrics[prefix_index]+=predicted==labels[row]?1.0:0.0;
                        metric_counts[static_cast<size_t>(prefix_index)]+=1.0;
                    }
                    if(store_scores)for(size_t active_index=0;
                        active_index<active.size();++active_index) {
                        const int destination_class=active[active_index]-1;
                        for(int i=0;i<test_n;++i) {
                            const int row=heldout[static_cast<size_t>(i)];
                            score_output[size_t(prefix_index)*n*classes+
                                size_t(destination_class)*n+row]=
                                host_scores[size_t(prefix_index)*test_n*active.size()+
                                    active_index*test_n+i];
                        }
                    }
                }
            }
            status[fold]=1;
        }
        for(int prefix=0;prefix<prefix_count;++prefix)
            metrics[prefix]=metric_counts[static_cast<size_t>(prefix)]>0.0?
                metrics[prefix]/metric_counts[static_cast<size_t>(prefix)]:
                std::numeric_limits<double>::quiet_NaN();
        release();
    } catch(...) {release();throw;}
}

template<class T,template<class> class Model,bool ReuseCrosscov,
         bool IncrementFoldSeed>
void resident_pls_cv_regression(
    const T* predictors,const T* responses,const int* folds,int n,int p,int q,
    const int* prefixes,int prefix_count,int scaling,int metric,
    int oversample,int power,unsigned long long seed,bool store_predictions,
    double* prediction_output,int* status,double* metrics,double* q2_values,
    double* rmsd_values,double* observed_r2_values) {
    if(!predictors||!responses||!folds||!prefixes||!status||!metrics||n<2||
       p<1||q<1||prefix_count<1||scaling<1||scaling>3||metric<2||metric>4)
        throw std::invalid_argument("invalid resident CUDA regression CV input");
    int maximum_prefix=0,fold_count=0;
    for(int j=0;j<prefix_count;++j) {
        if(prefixes[j]<=maximum_prefix)
            throw std::invalid_argument(
                "CUDA CV component path must be strictly increasing");
        maximum_prefix=prefixes[j];
    }
    for(int row=0;row<n;++row) {
        if(folds[row]<1)
            throw std::invalid_argument("invalid CUDA CV fold");
        fold_count=std::max(fold_count,folds[row]);
    }
    if(maximum_prefix>std::min(n-1,p)||
       (store_predictions&&!prediction_output))
        throw std::invalid_argument("invalid CUDA regression CV output or rank");
    std::vector<std::vector<int>> test_rows(static_cast<size_t>(fold_count));
    for(int row=0;row<n;++row)
        test_rows[static_cast<size_t>(folds[row]-1)].push_back(row);
    int maximum_test=0,maximum_train=0;
    for(const auto& test:test_rows) {
        maximum_test=std::max(maximum_test,static_cast<int>(test.size()));
        maximum_train=std::max(maximum_train,n-static_cast<int>(test.size()));
    }
    if(maximum_test<1||maximum_train<2)
        throw std::invalid_argument("CUDA CV contains an empty train or test fold");

    long double total_ss=0.0L;
    for(int response=0;response<q;++response) {
        long double response_sum=0.0L,response_square=0.0L;
        for(int row=0;row<n;++row) {
            const long double value=responses[size_t(response)*n+row];
            response_sum+=value;
            response_square+=value*value;
        }
        total_ss+=response_square-response_sum*response_sum/n;
    }
    cudaStream_t stream=nullptr;
    cublasHandle_t cv_blas=nullptr;
    cublasHandle_t component_blas=nullptr;
    cusolverDnHandle_t model_solver=nullptr;
    curandGenerator_t model_rng=nullptr;
    T *full_x=nullptr,*full_y=nullptr,*train_x=nullptr,*train_y=nullptr,
      *test_x=nullptr,*test_y=nullptr,*test_scores=nullptr,*predictions=nullptr,
      *metric_sums=nullptr,*full_crosscov=nullptr,*fold_crosscov=nullptr;
    int *train_indices=nullptr,*test_indices=nullptr,*metric_invalid=nullptr;
    auto release=[&]() noexcept {
        cudaFree(full_x);cudaFree(full_y);cudaFree(train_x);cudaFree(train_y);
        cudaFree(test_x);cudaFree(test_y);cudaFree(test_scores);
        cudaFree(predictions);cudaFree(metric_sums);cudaFree(train_indices);
        cudaFree(test_indices);cudaFree(metric_invalid);
        cudaFree(full_crosscov);cudaFree(fold_crosscov);
        if(component_blas)cublasDestroy(component_blas);
        if(model_rng)curandDestroyGenerator(model_rng);
        if(model_solver)cusolverDnDestroy(model_solver);
        if(cv_blas)cublasDestroy(cv_blas);
        if(stream)cudaStreamDestroy(stream);
    };
    try {
        require_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        require_blas(cublasCreate(&cv_blas));
        require_blas(cublasSetStream(cv_blas,stream));
        configure_blas_math<T>(cv_blas);
        require_blas(cublasCreate(&component_blas));
        require_blas(cublasSetStream(component_blas,stream));
        require_blas(cublasSetPointerMode(
            component_blas,CUBLAS_POINTER_MODE_DEVICE));
        require_solver(cusolverDnCreate(&model_solver));
        require_solver(cusolverDnSetStream(model_solver,stream));
        require_random(curandCreateGenerator(
            &model_rng,CURAND_RNG_PSEUDO_DEFAULT));
        require_random(curandSetStream(model_rng,stream));
        require_cuda(cudaMalloc(&full_x,size_t(n)*p*sizeof(T)));
        require_cuda(cudaMalloc(&full_y,size_t(n)*q*sizeof(T)));
        require_cuda(cudaMalloc(&train_x,size_t(maximum_train)*p*sizeof(T)));
        require_cuda(cudaMalloc(&train_y,size_t(maximum_train)*q*sizeof(T)));
        require_cuda(cudaMalloc(&test_x,size_t(maximum_test)*p*sizeof(T)));
        require_cuda(cudaMalloc(&test_y,size_t(maximum_test)*q*sizeof(T)));
        require_cuda(cudaMalloc(&test_scores,
                                size_t(maximum_test)*maximum_prefix*sizeof(T)));
        require_cuda(cudaMalloc(&predictions,size_t(maximum_test)*q*sizeof(T)));
        require_cuda(cudaMalloc(&metric_sums,size_t(3)*q*sizeof(T)));
        require_cuda(cudaMalloc(&train_indices,size_t(maximum_train)*sizeof(int)));
        require_cuda(cudaMalloc(&test_indices,size_t(maximum_test)*sizeof(int)));
        require_cuda(cudaMalloc(&metric_invalid,sizeof(int)));
        require_cuda(cudaMemcpyAsync(full_x,predictors,size_t(n)*p*sizeof(T),
                                     cudaMemcpyHostToDevice,stream));
        require_cuda(cudaMemcpyAsync(full_y,responses,size_t(n)*q*sizeof(T),
                                     cudaMemcpyHostToDevice,stream));
        const size_t crosscov_elements=size_t(p)*q;
        const size_t crosscov_bytes=crosscov_elements*sizeof(T);
        size_t free_bytes=0,total_bytes=0;
        require_cuda(cudaMemGetInfo(&free_bytes,&total_bytes));
        bool share_crosscov=ReuseCrosscov&&crosscov_bytes<=free_bytes/4;
        if(share_crosscov) {
            cudaError_t first=cudaMalloc(&full_crosscov,crosscov_bytes);
            cudaError_t second=first==cudaSuccess?
                cudaMalloc(&fold_crosscov,crosscov_bytes):first;
            if(first!=cudaSuccess||second!=cudaSuccess) {
                cudaFree(full_crosscov);cudaFree(fold_crosscov);
                full_crosscov=fold_crosscov=nullptr;
                cudaGetLastError();
                share_crosscov=false;
            }
        }
        if(share_crosscov) {
            const T one=T(1),zero=T(0);
            require_blas(Decomposition<T>::gemm(
                cv_blas,CUBLAS_OP_T,CUBLAS_OP_N,p,q,n,&one,full_x,n,
                full_y,n,&zero,full_crosscov,p));
        }
        std::fill(metrics,metrics+prefix_count,0.0);
        std::vector<double> counts(static_cast<size_t>(prefix_count),0.0);
        std::vector<double> fold_training_ss(
            static_cast<size_t>(prefix_count),0.0);

        for(int fold=0;fold<fold_count;++fold) {
            const auto& heldout=test_rows[static_cast<size_t>(fold)];
            if(heldout.empty()) {status[fold]=2;continue;}
            std::vector<char> heldout_flag(static_cast<size_t>(n),0);
            for(const int row:heldout)heldout_flag[static_cast<size_t>(row)]=1;
            std::vector<int> training;
            training.reserve(static_cast<size_t>(n)-heldout.size());
            for(int row=0;row<n;++row)
                if(!heldout_flag[static_cast<size_t>(row)])training.push_back(row);
            const int train_n=static_cast<int>(training.size());
            const int test_n=static_cast<int>(heldout.size());
            if(maximum_prefix>std::min(train_n-1,p))
                throw std::invalid_argument(
                    "CUDA CV component count exceeds a fold rank");
            require_cuda(cudaMemcpyAsync(train_indices,training.data(),
                                         size_t(train_n)*sizeof(int),
                                         cudaMemcpyHostToDevice,stream));
            require_cuda(cudaMemcpyAsync(test_indices,heldout.data(),
                                         size_t(test_n)*sizeof(int),
                                         cudaMemcpyHostToDevice,stream));
            gather_matrix_rows<<<dim3((train_n+255)/256,p),256,0,stream>>>(
                full_x,n,p,train_indices,train_n,train_x);
            gather_matrix_rows<<<dim3((test_n+255)/256,p),256,0,stream>>>(
                full_x,n,p,test_indices,test_n,test_x);
            gather_matrix_rows<<<dim3((train_n+255)/256,q),256,0,stream>>>(
                full_y,n,q,train_indices,train_n,train_y);
            gather_matrix_rows<<<dim3((test_n+255)/256,q),256,0,stream>>>(
                full_y,n,q,test_indices,test_n,test_y);
            require_cuda(cudaGetLastError());
            if(share_crosscov) {
                const T one=T(1),zero=T(0);
                require_blas(Decomposition<T>::gemm(
                    cv_blas,CUBLAS_OP_T,CUBLAS_OP_N,p,q,test_n,&one,
                    test_x,test_n,test_y,test_n,&zero,fold_crosscov,p));
                subtract_crosscovariance<<<256,256,0,stream>>>(
                    full_crosscov,fold_crosscov,crosscov_elements);
                require_cuda(cudaGetLastError());
            }
            {
                const unsigned long long fold_seed = seed +
                    (IncrementFoldSeed ? static_cast<unsigned long long>(fold) :
                     0ULL);
                Model<T> model(
                    train_n,p,q,maximum_prefix,oversample,power,stream,false,
                    false,0,0,T(0),0,T(0),false,
                    cv_blas,model_solver,model_rng,component_blas);
                if constexpr(ReuseCrosscov) {
                    if(share_crosscov) {
                        model.fit_borrowed_device_regression_crosscov(
                            train_x,train_y,fold_crosscov,scaling,fold_seed);
                    } else {
                        model.fit_borrowed_device_regression(
                            train_x,train_y,scaling,fold_seed,false,false);
                    }
                } else {
                    model.fit_borrowed_device_regression(
                        train_x,train_y,scaling,fold_seed,false,false);
                }
                model.standardize_device(test_x,test_n);
                model.project_standardized_device(
                    test_x,test_n,maximum_prefix,test_scores);
                std::vector<T> host_predictions;
                if(store_predictions)host_predictions.resize(
                    size_t(test_n)*q*prefix_count);
                std::vector<T> host_metric(size_t(3)*q*prefix_count);
                for(int prefix_index=0;prefix_index<prefix_count;++prefix_index) {
                    model.predict_projected_device(
                        test_scores,test_n,prefixes[prefix_index],predictions);
                    if(store_predictions)
                        require_cuda(cudaMemcpyAsync(
                            host_predictions.data()+size_t(prefix_index)*test_n*q,
                            predictions,size_t(test_n)*q*sizeof(T),
                            cudaMemcpyDeviceToHost,stream));
                    require_cuda(cudaMemsetAsync(
                        metric_invalid,0,sizeof(int),stream));
                    fastpls_device::response_sums<<<std::min(q,65535),256,0,stream>>>(
                        predictions,test_y,nullptr,model.exported_field(5),
                        test_n,q,metric_sums,metric_invalid);
                    require_cuda(cudaGetLastError());
                    require_cuda(cudaMemcpyAsync(
                        host_metric.data()+size_t(prefix_index)*3*q,
                        metric_sums,size_t(3)*q*sizeof(T),
                        cudaMemcpyDeviceToHost,stream));
                }
                require_cuda(cudaStreamSynchronize(stream));
                for(int prefix_index=0;prefix_index<prefix_count;++prefix_index) {
                    long double sse=0.0L,training_ss=0.0L;
                    for(int response=0;response<q;++response) {
                        sse+=host_metric[size_t(prefix_index)*3*q+
                                         size_t(response)*3];
                        training_ss+=host_metric[size_t(prefix_index)*3*q+
                                                 size_t(response)*3+1];
                    }
                    metrics[prefix_index]+=static_cast<double>(sse);
                    fold_training_ss[static_cast<size_t>(prefix_index)]+=
                        static_cast<double>(training_ss);
                    counts[static_cast<size_t>(prefix_index)]+=
                        static_cast<double>(test_n)*q;
                    if(store_predictions)for(int response=0;response<q;++response)
                        for(int i=0;i<test_n;++i) {
                            const int row=heldout[static_cast<size_t>(i)];
                            prediction_output[size_t(prefix_index)*n*q+
                                size_t(response)*n+row]=
                                static_cast<double>(host_predictions[
                                    size_t(prefix_index)*test_n*q+
                                    size_t(response)*test_n+i]);
                        }
                }
            }
            status[fold]=1;
        }
        for(int prefix=0;prefix<prefix_count;++prefix) {
            const double sse=metrics[prefix];
            rmsd_values[prefix]=counts[static_cast<size_t>(prefix)]>0.0?
                std::sqrt(sse/counts[static_cast<size_t>(prefix)]):
                std::numeric_limits<double>::quiet_NaN();
            q2_values[prefix]=fold_training_ss[static_cast<size_t>(prefix)]>0.0?
                1.0-sse/fold_training_ss[static_cast<size_t>(prefix)]:
                std::numeric_limits<double>::quiet_NaN();
            observed_r2_values[prefix]=total_ss>0.0L?
                1.0-sse/static_cast<double>(total_ss):
                std::numeric_limits<double>::quiet_NaN();
            metrics[prefix]=metric==4?rmsd_values[prefix]:
                metric==3?q2_values[prefix]:observed_r2_values[prefix];
        }
        release();
    } catch(...) {release();throw;}
}
} // namespace fastpls_device

extern "C" int fastpls_resident_simpls_cv_classification(
    const void* predictors,const int* labels,const int* folds,int precision,
    int n,int p,int classes,const int* prefixes,int prefix_count,int scaling,
    int lda,int oversample,int power,unsigned long long seed,
    int store_predictions,int store_scores,int* predictions,void* scores,
    int* status,double* metrics,char* error,size_t error_capacity) {
    using namespace fastpls_device;
    resident_error(error,error_capacity,"");
    try {
        if((precision!=32&&precision!=64)||(lda!=0&&lda!=1)||
           (store_predictions!=0&&store_predictions!=1)||
           (store_scores!=0&&store_scores!=1))
            throw std::invalid_argument("invalid resident CUDA CV controls");
        if(precision==32)resident_pls_cv_classification<
            float,ResidentSimpls,false>(
            static_cast<const float*>(predictors),labels,folds,n,p,classes,
            prefixes,prefix_count,scaling,lda==1,oversample,power,seed,
            store_predictions==1,store_scores==1,predictions,
            static_cast<float*>(scores),status,metrics);
        else resident_pls_cv_classification<double,ResidentSimpls,false>(
            static_cast<const double*>(predictors),labels,folds,n,p,classes,
            prefixes,prefix_count,scaling,lda==1,oversample,power,seed,
            store_predictions==1,store_scores==1,predictions,
            static_cast<double*>(scores),status,metrics);
        return 0;
    } catch(const std::exception& exception) {
        resident_error(error,error_capacity,exception.what());return 1;
    } catch(...) {
        resident_error(error,error_capacity,
                       "unknown resident CUDA CV error");return 1;
    }
}

extern "C" int fastpls_resident_simpls_cv_regression(
    const void* predictors,const void* responses,const int* folds,
    int precision,int n,int p,int q,const int* prefixes,int prefix_count,
    int scaling,int metric,int oversample,int power,unsigned long long seed,
    int store_predictions,double* predictions,int* status,double* metrics,
    double* q2,double* rmsd,double* observed_r2,
    char* error,size_t error_capacity) {
    using namespace fastpls_device;
    resident_error(error,error_capacity,"");
    try {
        if((precision!=32&&precision!=64)||!q2||!rmsd||!observed_r2||
           (store_predictions!=0&&store_predictions!=1))
            throw std::invalid_argument(
                "invalid resident CUDA regression CV controls");
        if(precision==32)resident_pls_cv_regression<
            float,ResidentSimpls,true,false>(
            static_cast<const float*>(predictors),
            static_cast<const float*>(responses),folds,n,p,q,prefixes,
            prefix_count,scaling,metric,oversample,power,seed,
            store_predictions==1,predictions,status,metrics,q2,rmsd,
            observed_r2);
        else resident_pls_cv_regression<double,ResidentSimpls,true,false>(
            static_cast<const double*>(predictors),
            static_cast<const double*>(responses),folds,n,p,q,prefixes,
            prefix_count,scaling,metric,oversample,power,seed,
            store_predictions==1,predictions,status,metrics,q2,rmsd,
            observed_r2);
        return 0;
    } catch(const std::exception& exception) {
        resident_error(error,error_capacity,exception.what());return 1;
    } catch(...) {
        resident_error(error,error_capacity,
                       "unknown resident CUDA regression CV error");return 1;
    }
}

extern "C" int fastpls_resident_plssvd_cv_classification(
    const void* predictors,const int* labels,const int* folds,int precision,
    int n,int p,int classes,const int* prefixes,int prefix_count,int scaling,
    int lda,int oversample,int power,unsigned long long seed,
    int store_predictions,int store_scores,int* predictions,void* scores,
    int* status,double* metrics,char* error,size_t error_capacity) {
    using namespace fastpls_device;
    resident_error(error,error_capacity,"");
    try {
        if((precision!=32&&precision!=64)||(lda!=0&&lda!=1)||
           (store_predictions!=0&&store_predictions!=1)||
           (store_scores!=0&&store_scores!=1))
            throw std::invalid_argument(
                "invalid resident CUDA PLS-SVD CV controls");
        if(precision==32)
            resident_pls_cv_classification<float,ResidentPlssvd,true>(
                static_cast<const float*>(predictors),labels,folds,n,p,
                classes,prefixes,prefix_count,scaling,lda==1,oversample,
                power,seed,store_predictions==1,store_scores==1,predictions,
                static_cast<float*>(scores),status,metrics);
        else resident_pls_cv_classification<double,ResidentPlssvd,true>(
                static_cast<const double*>(predictors),labels,folds,n,p,
                classes,prefixes,prefix_count,scaling,lda==1,oversample,
                power,seed,store_predictions==1,store_scores==1,predictions,
                static_cast<double*>(scores),status,metrics);
        return 0;
    } catch(const std::exception& exception) {
        resident_error(error,error_capacity,exception.what());return 1;
    } catch(...) {
        resident_error(error,error_capacity,
                       "unknown resident CUDA PLS-SVD CV error");return 1;
    }
}

extern "C" int fastpls_resident_plssvd_cv_regression(
    const void* predictors,const void* responses,const int* folds,
    int precision,int n,int p,int q,const int* prefixes,int prefix_count,
    int scaling,int metric,int oversample,int power,unsigned long long seed,
    int store_predictions,double* predictions,int* status,double* metrics,
    double* q2,double* rmsd,double* observed_r2,
    char* error,size_t error_capacity) {
    using namespace fastpls_device;
    resident_error(error,error_capacity,"");
    try {
        if((precision!=32&&precision!=64)||!q2||!rmsd||!observed_r2||
           (store_predictions!=0&&store_predictions!=1))
            throw std::invalid_argument(
                "invalid resident CUDA PLS-SVD regression CV controls");
        if(precision==32)
            resident_pls_cv_regression<float,ResidentPlssvd,false,true>(
                static_cast<const float*>(predictors),
                static_cast<const float*>(responses),folds,n,p,q,prefixes,
                prefix_count,scaling,metric,oversample,power,seed,
                store_predictions==1,predictions,status,metrics,q2,rmsd,
                observed_r2);
        else resident_pls_cv_regression<double,ResidentPlssvd,false,true>(
                static_cast<const double*>(predictors),
                static_cast<const double*>(responses),folds,n,p,q,prefixes,
                prefix_count,scaling,metric,oversample,power,seed,
                store_predictions==1,predictions,status,metrics,q2,rmsd,
                observed_r2);
        return 0;
    } catch(const std::exception& exception) {
        resident_error(error,error_capacity,exception.what());return 1;
    } catch(...) {
        resident_error(error,error_capacity,
                       "unknown resident CUDA PLS-SVD regression CV error");
        return 1;
    }
}

extern "C" int fastpls_cuda_gemm(const void* left,const void* right,
    int precision,int rows,int inner,int columns,void* output,char* error,
    size_t size) {
    using namespace fastpls_device;
    resident_error(error,size,"");
    try {
        if(!left||!right||!output||(precision!=32&&precision!=64)||
           rows<1||inner<1||columns<1)
            throw std::invalid_argument("invalid CUDA matrix multiplication input");
        if(precision==32)
            host_gemm(static_cast<const float*>(left),
                static_cast<const float*>(right),rows,inner,columns,
                static_cast<float*>(output));
        else
            host_gemm(static_cast<const double*>(left),
                static_cast<const double*>(right),rows,inner,columns,
                static_cast<double*>(output));
        return 0;
    } catch(const std::exception& exception) {
        resident_error(error,size,exception.what());return 1;
    } catch(...) {
        resident_error(error,size,"unknown CUDA matrix multiplication error");
        return 1;
    }
}

extern "C" void* fastpls_resident_simpls_create(const void* x,const void* y,const int* labels,
    int precision,int n,int p,int q,int components,int scaling,int oversample,
    int power,int retain_scores,unsigned long long seed,char* error,size_t size) {
    using namespace fastpls_device;
    resident_error(error,size,"");
    try {
        if(!x || (precision!=32&&precision!=64))throw std::invalid_argument("invalid resident matrix or precision");
        if(retain_scores!=0&&retain_scores!=1)
            throw std::invalid_argument("invalid resident score-retention request");
        if(precision==32)return new TypedResidentHandle<float>(x,y,labels,n,p,q,components,scaling,oversample,power,retain_scores==1,seed);
        return new TypedResidentHandle<double>(x,y,labels,n,p,q,components,scaling,oversample,power,retain_scores==1,seed);
    } catch(const std::exception& e) {resident_error(error,size,e.what());return nullptr;}
      catch(...) {resident_error(error,size,"unknown resident CUDA fitting error");return nullptr;}
}
extern "C" void* fastpls_resident_plssvd_create(const void* x,const void* y,const int* labels,
    int precision,int n,int p,int q,int components,int scaling,int oversample,
    int power,int retain_scores,unsigned long long seed,char* error,size_t size) {
    using namespace fastpls_device;
    resident_error(error,size,"");
    try {
        if(!x || (precision!=32&&precision!=64))throw std::invalid_argument("invalid resident matrix or precision");
        if(retain_scores!=0&&retain_scores!=1)
            throw std::invalid_argument("invalid resident score-retention request");
        if(precision==32)return new TypedResidentHandle<float,ResidentPlssvd>(x,y,labels,n,p,q,components,scaling,oversample,power,retain_scores==1,seed);
        return new TypedResidentHandle<double,ResidentPlssvd>(x,y,labels,n,p,q,components,scaling,oversample,power,retain_scores==1,seed);
    } catch(const std::exception& e) {resident_error(error,size,e.what());return nullptr;}
      catch(...) {resident_error(error,size,"unknown resident CUDA PLS-SVD fitting error");return nullptr;}
}
extern "C" void* fastpls_resident_opls_create(const void* x,const void* y,
    const int* labels,int precision,int n,int p,int q,int components,
    int scaling,int oversample,int power,int retain_scores,
    unsigned long long seed,int north,char* error,size_t size) {
    using namespace fastpls_device;
    resident_error(error,size,"");
    try {
        if(!x||(precision!=32&&precision!=64)||north<0)
            throw std::invalid_argument("invalid resident CUDA OPLS input");
        if(precision==32)return new TypedResidentHandle<float,ResidentOpls>(
            x,y,labels,n,p,q,components,scaling,oversample,power,
            retain_scores==1,seed,north);
        return new TypedResidentHandle<double,ResidentOpls>(
            x,y,labels,n,p,q,components,scaling,oversample,power,
            retain_scores==1,seed,north);
    } catch(const std::exception& exception) {
        resident_error(error,size,exception.what());return nullptr;
    } catch(...) {
        resident_error(error,size,"unknown resident CUDA OPLS fitting error");
        return nullptr;
    }
}
extern "C" void* fastpls_resident_kernelpls_create(const void* x,const void* y,
    const int* labels,int precision,int n,int p,int q,int components,
    int scaling,int oversample,int power,int retain_scores,
    unsigned long long seed,int kernel,double gamma,int degree,double coef0,
    char* error,size_t size) {
    using namespace fastpls_device;
    resident_error(error,size,"");
    try {
        if(!x||(precision!=32&&precision!=64)||(kernel!=2&&kernel!=3))
            throw std::invalid_argument(
                "invalid resident CUDA nonlinear kernel PLS input");
        if(precision==32)return new TypedResidentHandle<float,ResidentKernelPls>(
            x,y,labels,n,p,q,components,scaling,oversample,power,
            retain_scores==1,seed,0,kernel,gamma,degree,coef0);
        return new TypedResidentHandle<double,ResidentKernelPls>(
            x,y,labels,n,p,q,components,scaling,oversample,power,
            retain_scores==1,seed,0,kernel,gamma,degree,coef0);
    } catch(const std::exception& exception) {
        resident_error(error,size,exception.what());return nullptr;
    } catch(...) {
        resident_error(error,size,
            "unknown resident CUDA nonlinear kernel PLS fitting error");
        return nullptr;
    }
}
extern "C" void fastpls_resident_simpls_destroy(void* model) {
    delete static_cast<fastpls_device::ResidentHandle*>(model);
}
extern "C" int fastpls_resident_export(void* model,int field,void* out,size_t size,char* error,size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model)throw std::invalid_argument("null resident model");
        static_cast<ResidentHandle*>(model)->export_field(field,out,size);return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident export error");return 1;}
}
extern "C" int fastpls_resident_predict_path(
    void* model,const void* x,int rows,const int* prefixes,int prefix_count,
    int lda,void* out,char* error,size_t size){
    using namespace fastpls_device;resident_error(error,size,"");
    try{
        if(!model||(lda!=0&&lda!=1))
            throw std::invalid_argument("invalid resident prediction-path model");
        static_cast<ResidentHandle*>(model)->predict_path(
            x,rows,prefixes,prefix_count,lda==1,out);
        return 0;
    }catch(const std::exception& e){resident_error(error,size,e.what());return 1;}
     catch(...){resident_error(error,size,"unknown resident prediction-path error");return 1;}
}
extern "C" int fastpls_resident_classify_path(
    void* model,const void* x,int rows,const int* prefixes,int prefix_count,
    int lda,int top,int* out,char* error,size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model||(lda!=0&&lda!=1))
            throw std::invalid_argument("invalid resident classification-path model");
        static_cast<ResidentHandle*>(model)->classify_path(
            x,rows,prefixes,prefix_count,lda==1,top,out);
        return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident classification-path error");return 1;}
}
extern "C" int fastpls_resident_classify_response_path(
    void* model,const void* x,int rows,const int* prefixes,int prefix_count,
    int lda,int top,int* labels,void* predictions,char* error,
    size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model)throw std::invalid_argument("null resident model");
        static_cast<ResidentHandle*>(model)->classify_response_path(
            x,rows,prefixes,prefix_count,lda!=0,top,labels,predictions);
        return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident classification-response path error");return 1;}
}
extern "C" int fastpls_resident_response_sums(void* model,const void* x,const void* y,const int* labels,int rows,int prefix,void* out,char* error,size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model)throw std::invalid_argument("null resident model");
        static_cast<ResidentHandle*>(model)->response_sums(x,y,labels,rows,prefix,out);return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident metric error");return 1;}
}
extern "C" int fastpls_resident_project(void* model,const void* x,int rows,int prefix,void* out,char* error,size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model)throw std::invalid_argument("null resident model");
        static_cast<ResidentHandle*>(model)->project(x,rows,prefix,out);return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident projection error");return 1;}
}
extern "C" int fastpls_resident_controls(void* model,int* oversample,int* power,
    int* block,int* block_limit,int* implicit_operator,int* predictor_crossprod_cache,
    char* error,size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model)throw std::invalid_argument("null resident model");
        static_cast<ResidentHandle*>(model)->controls(
            oversample,power,block,block_limit,implicit_operator,
            predictor_crossprod_cache);
        return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident control error");return 1;}
}
extern "C" int fastpls_resident_compact(void* model,int prepare_lda,char* error,size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model||(prepare_lda!=0&&prepare_lda!=1))throw std::invalid_argument("invalid resident compaction request");
        static_cast<ResidentHandle*>(model)->compact(prepare_lda==1);return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident compaction error");return 1;}
}
#endif
