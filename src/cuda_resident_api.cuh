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
} // namespace fastpls_device

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
extern "C" int fastpls_resident_simpls_predict(void* model,const void* x,int rows,int prefix,
    void* out,char* error,size_t size) {
    using namespace fastpls_device;
    resident_error(error,size,"");
    try {
        if(!model)throw std::invalid_argument("null resident model");
        static_cast<ResidentHandle*>(model)->predict(x,rows,prefix,out,false);return 0;
    } catch(const std::exception& e){resident_error(error,size,e.what());return 1;}
      catch(...){resident_error(error,size,"unknown resident CUDA prediction error");return 1;}
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
extern "C" int fastpls_resident_lda_predict(void* model,const void* x,int rows,int prefix,
    void* out,char* error,size_t size){
    using namespace fastpls_device;resident_error(error,size,"");
    try{
        if(!model)throw std::invalid_argument("null resident model");
        static_cast<ResidentHandle*>(model)->predict(x,rows,prefix,out,true);return 0;
    }catch(const std::exception& e){resident_error(error,size,e.what());return 1;}
     catch(...){resident_error(error,size,"unknown resident LDA error");return 1;}
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
extern "C" int fastpls_resident_classify(void* model,const void* x,int rows,int prefix,int lda,int top,int* out,char* error,size_t capacity){
    using namespace fastpls_device;resident_error(error,capacity,"");
    try{
        if(!model||(lda!=0&&lda!=1))throw std::invalid_argument("invalid resident classification model");
        static_cast<ResidentHandle*>(model)->classify(x,rows,prefix,lda==1,top,out);return 0;
    }catch(const std::exception& e){resident_error(error,capacity,e.what());return 1;}
     catch(...){resident_error(error,capacity,"unknown resident classification error");return 1;}
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
