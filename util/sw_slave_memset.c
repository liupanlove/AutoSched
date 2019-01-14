#include <slave.h>
#include <simd.h>
#include <dma.h>
// BUFFSIZE: number of float/double numbers in LDM buffer

typedef struct MemsetPara_st {
  void *src;
  long count;
}MemsetPara;

inline void mb()
{
    asm volatile("memb");
    asm volatile("":::"memory");
}
__thread_local dma_desc dma_put;

void sw_slave_memset_d(MemsetPara *para) {
  const int  SIMDSIZE = 4;
  const int  BUFFSIZE = 4*1024;
  const int  SPNUM = 64; 
  double* local_src = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double * src_ptr = &(((double *)para->src)[start]);
  doublev4 dst = 0.0;
  volatile int replyput=0;
  int off;
  // DMA settings
  //dma_desc  dma_put;

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);
  dma_set_size(&dma_put,BUFFSIZE*sizeof(double));
  for(off=0; off<BUFFSIZE; off+=SIMDSIZE) {
     simd_store(dst,local_src+off);
  }
  mb();
  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA put result
    dma(dma_put, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }
  mb();

  if(off<local_count) {
    dma_set_size(&dma_put,(local_count-off)*sizeof(double));
    dma(dma_put, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
    mb();
  }
  ldm_free(local_src, BUFFSIZE*sizeof(double));
}
void sw_slave_memset_f(MemsetPara *para) {
  const int  SIMDSIZE = 4;
  const int  BUFFSIZE = 4*1024;
  const int  SPNUM = 64; 
  float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  int off;
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr = &(((float *)para->src)[start]);
  floatv4 dst = 0.0;
  volatile int replyput=0;
  // DMA settings
  //dma_desc dma_put;
  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);
  dma_set_size(&dma_put,BUFFSIZE*sizeof(float));

  for(off=0; off<BUFFSIZE; off+=SIMDSIZE) {
     simd_store(dst,local_src+off);
  }
  mb();
  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA put result
    dma(dma_put, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }
  mb();

  if(off<local_count) {
    dma_set_size(&dma_put,(local_count-off)*sizeof(float));
    dma(dma_put, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }
  mb();

  ldm_free(local_src, BUFFSIZE*sizeof(float));
}
void sw_slave_memset_i(MemsetPara *para) {
  const int  SIMDSIZE = 8;
  const int  BUFFSIZE = 4*1024;
  const int  SPNUM = 64; 
  int * local_src = (int *)ldm_malloc(BUFFSIZE*sizeof(int ));
  int off;
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int * src_ptr = (int *)para->src;
  intv8 dst = 0;
  volatile int replyput=0;
  // DMA settings
  //dma_desc dma_put;

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  dma_set_size(&dma_put,BUFFSIZE*sizeof(int));

  for(off=0; off<BUFFSIZE; off+=SIMDSIZE) {
     simd_store(dst,local_src+off);
  }
  mb();
  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA put result
    dma(dma_put, (long)(src_ptr+start+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_put,(local_count-off)*sizeof(int));
    dma(dma_put, (long)(src_ptr+start+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  mb();
  ldm_free(local_src, BUFFSIZE*sizeof(int));
}
