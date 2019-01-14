#include <slave.h>
#include <simd.h>
#include <dma.h>
#include <assert.h>
// BUFFSIZE: number of float/double numbers in LDM buffer


typedef struct MemcpyPara_st {
  void *src;
  void *dst;
  long count;
}MemcpyPara;

inline void mb()
{
    asm volatile("memb");
    asm volatile("":::"memory");
}
__thread_local  dma_desc dma_get, dma_put;
void sw_slave_memcpy_d(MemcpyPara *para) {
  const int BUFFSIZE = 2*1024;
  const int SPNUM = 64;
  double* local_src = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  int id = athread_get_id(-1);
  long count = para->count;
  long local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double * src_ptr = &(((double *)para->src)[start]);
  double * dst_ptr = &(((double *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  long off;
  // DMA settings
  //dma_desc dma_get, dma_put;

  dma_set_op(&dma_get, DMA_GET);
  dma_set_mode(&dma_get, PE_MODE);
  dma_set_reply(&dma_get, &replyget);

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  dma_set_size(&dma_get,BUFFSIZE*sizeof(double));
  dma_set_size(&dma_put,BUFFSIZE*sizeof(double));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    mb();
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
    mb();
  }

  if(off<local_count) {
    int left_count = local_count-off;
    mb();
    dma_set_size(&dma_get,left_count*sizeof(double));
    dma_set_size(&dma_put,left_count*sizeof(double));
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
    mb();
  }

  ldm_free(local_src, BUFFSIZE*sizeof(double));
}

void sw_slave_memcpy_f(MemcpyPara *para) {
  const int BUFFSIZE = 4*1024;
  const int SPNUM = 64;
  float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  int id = athread_get_id(-1);
  long count = para->count;
  long local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr = &(((float *)para->src)[start]);
  float * dst_ptr = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  long off;
  // DMA settings
  //dma_desc dma_get, dma_put;
  dma_set_op(&dma_get, DMA_GET);
  dma_set_mode(&dma_get, PE_MODE);
  dma_set_reply(&dma_get, &replyget);

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  dma_set_size(&dma_get,BUFFSIZE*sizeof(float));
  dma_set_size(&dma_put,BUFFSIZE*sizeof(float));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
    mb();
  }

  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get,left_count*sizeof(float));
    dma_set_size(&dma_put,left_count*sizeof(float));
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
    mb();
  }

  ldm_free(local_src, BUFFSIZE*sizeof(float));
}
void sw_slave_memcpy_i(MemcpyPara *para) {
  const int BUFFSIZE = 8*1024;
  const int SPNUM = 64;
  int* local_src = (int*)ldm_malloc(BUFFSIZE*sizeof(int));
  int id = athread_get_id(-1);
  long count = para->count;
  long local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int * src_ptr = &(((int *)para->src)[start]);
  int * dst_ptr = &(((int *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  long off;
  // DMA settings
  //dma_desc dma_get, dma_put;

  dma_set_op(&dma_get, DMA_GET);
  dma_set_mode(&dma_get, PE_MODE);
  dma_set_reply(&dma_get, &replyget);

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  dma_set_size(&dma_get,BUFFSIZE*sizeof(int));
  dma_set_size(&dma_put,BUFFSIZE*sizeof(int));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
    mb();
  }

  if(off<local_count) {
    dma_set_size(&dma_get,(local_count-off)*sizeof(int));
    dma_set_size(&dma_put,(local_count-off)*sizeof(int));
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
    mb();
  }

  ldm_free(local_src, BUFFSIZE*sizeof(int));
}
void sw_slave_memcpy_c2d(MemcpyPara *para) {
  const int BUFFSIZE = 2*1024;
  const int SPNUM = 64;
  unsigned char* local_src = (unsigned char*)ldm_malloc(BUFFSIZE*sizeof(unsigned char));
  double* local_dst = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  int id = athread_get_id(-1);
  long count = para->count;
  long local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  unsigned char * src_ptr = &(((unsigned char *)para->src)[start]);
  double * dst_ptr = &(((double *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  long off,i;
  // DMA settings
  //dma_desc dma_get, dma_put;

  dma_set_op(&dma_get, DMA_GET);
  dma_set_mode(&dma_get, PE_MODE);
  dma_set_reply(&dma_get, &replyget);

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  dma_set_size(&dma_get,BUFFSIZE*sizeof(unsigned char));
  dma_set_size(&dma_put,BUFFSIZE*sizeof(double));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;

    for(i = 0; i < BUFFSIZE;i++)
      local_dst[i] = local_src[i];
    mb();

    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get,left_count*sizeof(unsigned char));
    dma_set_size(&dma_put,left_count*sizeof(double));
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    
    for(i = 0; i < left_count;i++)
      local_dst[i] = local_src[i];

    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }
  mb();

  ldm_free(local_src, BUFFSIZE*sizeof(unsigned char));
  ldm_free(local_dst, BUFFSIZE*sizeof(double));
}
void sw_slave_memcpy_c2f(MemcpyPara *para) {
  const int BUFFSIZE = 4096;
  const int SPNUM = 64;
  unsigned char* local_src = (unsigned char*)ldm_malloc(BUFFSIZE*sizeof(unsigned char));
  assert(local_src != NULL);
  float* local_dst = (float*)ldm_malloc(BUFFSIZE*sizeof(float));
  assert(local_dst != NULL);
  int id = athread_get_id(-1);
  long count = para->count;
  long local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  unsigned char * src_ptr = &(((unsigned char *)para->src)[start]);
  float * dst_ptr = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  long off,i;
  // DMA settings
  //dma_desc dma_get, dma_put;

  dma_set_op(&dma_get, DMA_GET);
  dma_set_mode(&dma_get, PE_MODE);
  dma_set_reply(&dma_get, &replyget);

  dma_set_op(&dma_put, DMA_PUT);
  dma_set_mode(&dma_put, PE_MODE);
  dma_set_reply(&dma_put, &replyput);

  dma_set_size(&dma_get,BUFFSIZE*sizeof(unsigned char));
  dma_set_size(&dma_put,BUFFSIZE*sizeof(float));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;

    for(i = 0; i < BUFFSIZE;i++)
      local_dst[i] = local_src[i];
    mb();

    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get,left_count*sizeof(unsigned char));
    dma_set_size(&dma_put,left_count*sizeof(float));
    dma(dma_get, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    
    for(i = 0; i < left_count;i++)
      local_dst[i] = local_src[i];

    mb();
    // DMA put result
    dma(dma_put, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }
  mb();

  ldm_free(local_src, BUFFSIZE*sizeof(unsigned char));
  ldm_free(local_dst, BUFFSIZE*sizeof(float));
}
 
 
