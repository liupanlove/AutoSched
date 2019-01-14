#include <slave.h>
#include <simd.h>
#include <dma.h>
#include <assert.h>
#define SPNUM 64

typedef struct DivPara_st {
  void *src,*dst,*old;
  int *center_num;
  int count,dims;
}DivPara;

inline void mb()
{
    asm volatile("memb");
    asm volatile("":::"memory");
}
__thread_local dma_desc dma_get_src,dma_get_num_src,dma_put_dst;

void sw_slave_div_d(DivPara *para) {
  const int max_len = 58*1024;
  int id = athread_get_id(-1);
  int count = para->count;
  int dims = para->dims;
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int local_count = count/SPNUM + (id<(count%SPNUM));
  if(local_count < 1) return;
  assert(max_len - local_count*sizeof(int) > 0);
  int max_num = (max_len -  local_count*sizeof(int)) / sizeof(double);
  assert(max_num >0);
  max_num = max_num > dims ? dims : max_num;
  int start_addr = start * dims;
  double* src_ptr = &(((double*)para->src)[start_addr]);
  double* dst_ptr = &(((double*)para->dst)[start_addr]);
  double* old_ptr = &(((double*)para->old)[start_addr]);
  int*    num_ptr = &(((int*)para->center_num)[start]);
  double* local_src = (double*)ldm_malloc(max_num*sizeof(double));
  int* local_num = (int*)ldm_malloc(local_count*sizeof(int));
  volatile int replyget=0, replyput=0,replyget_num=0;
  int i,c,k;
  // DMA settings
  //dma_desc dma_get_src,dma_get_num_src,dma_put_dst;
  
  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);
  dma_set_size(&dma_get_src,max_num*sizeof(double));

  dma_set_op(&dma_get_num_src, DMA_GET);
  dma_set_mode(&dma_get_num_src, PE_MODE);
  dma_set_reply(&dma_get_num_src, &replyget_num);
  dma_set_size(&dma_get_num_src,local_count*sizeof(int));

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);
  dma_set_size(&dma_put_dst,max_num*sizeof(double));
  
  doublev4 vsrc,vc;
  int num_val = 0,simd_size = 4,left_dims=0;
  long index = 0;
  dma(dma_get_num_src, (long)(num_ptr), (long)(local_num));
  dma_wait(&replyget_num, 1); replyget_num = 0;

  for(c = 0; c < local_count; c++)
  {
    num_val = local_num[c];
    index = c*dims;
    if(num_val > 0) {
      vc = num_val;
      dma_set_size(&dma_get_src,max_num*sizeof(double));
      dma_set_size(&dma_put_dst,max_num*sizeof(double));
      for(k = 0; k+max_num-1 < dims; k+=max_num)
      {
         // DMA get a block
         dma(dma_get_src, (long)(src_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;
    
         for(i = 0; i + simd_size -1 < max_num; i+=simd_size) {
           simd_load(vsrc,local_src+i);
           vsrc /= vc;
           simd_store(vsrc,local_src+i);
         }
         for(; i < max_num; i++) local_src[i] /= num_val;

         mb();
         // DMA put result
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
      if(k < dims){
         left_dims = dims - k;
         dma_set_size(&dma_get_src,left_dims*sizeof(double));
         dma(dma_get_src, (long)(src_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;
       
         for(i = 0; i + simd_size -1 < max_num; i+=simd_size) {
           simd_load(vsrc,local_src+i);
           vsrc /= vc;
           simd_store(vsrc,local_src+i);
         }
         for(; i < max_num; i++) local_src[i] /= num_val;

         mb();
         // DMA put result
         dma_set_size(&dma_put_dst,left_dims*sizeof(double));
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
    }
    else{
      dma_set_size(&dma_get_src,max_num*sizeof(double));
      dma_set_size(&dma_put_dst,max_num*sizeof(double));
      for(k = 0; k+max_num-1 < dims; k+=max_num)
      {
         // DMA get a block
         mb();
         dma(dma_get_src, (long)(old_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;

         mb();
         // DMA put result
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
      if(k < dims){
         mb();
         left_dims = dims - k;
         dma_set_size(&dma_get_src,left_dims*sizeof(double));
         dma(dma_get_src, (long)(old_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;
       
         mb();
         // DMA put result
         dma_set_size(&dma_put_dst,left_dims*sizeof(double));
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
    }
  }
  
  ldm_free(local_src, max_num*sizeof(double));
  ldm_free(local_num, local_count*sizeof(int));
}
void sw_slave_div_f(DivPara *para) {
  const int max_len = 58*1024;
  int id = athread_get_id(-1);
  int count = para->count;
  int dims = para->dims;
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int local_count = count/SPNUM + (id<(count%SPNUM));
  if(local_count < 1) return;

  int max_num = (max_len -  local_count*sizeof(int)) / sizeof(float);
  assert(max_num >0);
  max_num = max_num > dims ? dims : max_num;
  int start_addr = start * dims;
  float* src_ptr = &(((float*)para->src)[start_addr]);
  float* dst_ptr = &(((float*)para->dst)[start_addr]);
  float* old_ptr = &(((float*)para->old)[start_addr]);
  int* num_ptr = &(((int*)para->center_num)[start]);
  float* local_src = (float*)ldm_malloc(max_num*sizeof(float));
  int* local_num = (int*)ldm_malloc(local_count*sizeof(int));
  volatile int replyget=0, replyput=0,replyget_num=0;
  int i,c,k;
  // DMA settings
  //dma_desc dma_get_src,dma_get_num_src,dma_put_dst;
  
  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);
  dma_set_size(&dma_get_src,max_num*sizeof(float));

  dma_set_op(&dma_get_num_src, DMA_GET);
  dma_set_mode(&dma_get_num_src, PE_MODE);
  dma_set_reply(&dma_get_num_src, &replyget_num);
  dma_set_size(&dma_get_num_src,local_count*sizeof(int));

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);
  dma_set_size(&dma_put_dst,max_num*sizeof(float));
  
  floatv4 vsrc,vc;
  int num_val = 0,simd_size = 4,left_dims=0;
  long index = 0;
  dma(dma_get_num_src, (long)(num_ptr), (long)(local_num));
  dma_wait(&replyget_num, 1); replyget_num = 0;

  for(c = 0; c < local_count; c++)
  {
    num_val = local_num[c];
    index = c*dims;
    if(num_val > 0) {
      vc = num_val;
      dma_set_size(&dma_get_src,max_num*sizeof(float));
      dma_set_size(&dma_put_dst,max_num*sizeof(float));
      for(k = 0; k+max_num-1 < dims; k+=max_num)
      {
         // DMA get a block
         dma(dma_get_src, (long)(src_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;
    
         for(i = 0; i + simd_size -1 < max_num; i+=simd_size) {
           simd_load(vsrc,local_src+i);
           vsrc /= vc;
           simd_store(vsrc,local_src+i);
         }
         for(; i < max_num; i++) local_src[i] /= num_val;

         mb();
         // DMA put result
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
      if(k < dims){
         left_dims = dims - k;
         dma_set_size(&dma_get_src,left_dims*sizeof(float));
         dma(dma_get_src, (long)(src_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;
       
         for(i = 0; i + simd_size -1 < max_num; i+=simd_size) {
           simd_load(vsrc,local_src+i);
           vsrc /= vc;
           simd_store(vsrc,local_src+i);
         }
         for(; i < max_num; i++) local_src[i] /= num_val;

         mb();
         // DMA put result
         dma_set_size(&dma_put_dst,left_dims*sizeof(float));
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
    }
    else{
      dma_set_size(&dma_get_src,max_num*sizeof(float));
      dma_set_size(&dma_put_dst,max_num*sizeof(float));
      for(k = 0; k+max_num-1 < dims; k+=max_num)
      {
         // DMA get a block
         mb();
         dma(dma_get_src, (long)(old_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;

         mb();
         // DMA put result
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
      if(k < dims){
         left_dims = dims - k;
         mb();
         dma_set_size(&dma_get_src,left_dims*sizeof(float));
         dma(dma_get_src, (long)(old_ptr + index + k), (long)(local_src));
         dma_wait(&replyget, 1); replyget = 0;
       
         mb();
         // DMA put result
         dma_set_size(&dma_put_dst,left_dims*sizeof(float));
         dma(dma_put_dst, (long)(dst_ptr + index + k), (long)(local_src));
         dma_wait(&replyput, 1); replyput = 0;
      }
    }
  }
  
  ldm_free(local_src, max_num*sizeof(float));
  ldm_free(local_num, local_count*sizeof(int));
}
