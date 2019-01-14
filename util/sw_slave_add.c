#include <slave.h>
#include <simd.h>
#include <dma.h>
#define SPNUM 64


typedef struct addPara_st {
  void *src1;
  void *src2;
  void *src3;
  void *src4;
  void *dst;
  long count;
}addPara;

__thread_local dma_desc dma_get_src, dma_put_dst;
void sw_slave_add_4cg_d(addPara *para) {
  const int buff_size = 1024;
  const int simd_size = 4;
  double* local_src1 = (double*)ldm_malloc(buff_size*sizeof(double));
  double* local_src2 = (double*)ldm_malloc(buff_size*sizeof(double));
  double* local_src3 = (double*)ldm_malloc(buff_size*sizeof(double));
  double* local_src4 = (double*)ldm_malloc(buff_size*sizeof(double));
  double* local_dst  = (double*)ldm_malloc(buff_size*sizeof(double));
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double* src_ptr1 = &(((double*)para->src1)[start]);
  double* src_ptr2 = &(((double*)para->src2)[start]);
  double* src_ptr3 = &(((double*)para->src3)[start]);
  double* src_ptr4 = &(((double*)para->src4)[start]);
  double* dst_ptr  = &(((double *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  doublev4 va1,va2,va3,va4;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,buff_size*sizeof(double));
  dma_set_size(&dma_put_dst,buff_size*sizeof(double));

  for(off = 0; off+buff_size-1<local_count; off+=buff_size)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma(dma_get_src, (long)(src_ptr3+off), (long)(local_src3));
    dma(dma_get_src, (long)(src_ptr4+off), (long)(local_src4));
    dma_wait(&replyget, 4); replyget = 0;

    for(i=0; i<buff_size; i+=simd_size) {
       simd_load(va1,local_src1+i);
       simd_load(va2,local_src2+i);
       simd_load(va3,local_src3+i);
       va1 = va1 + va2;
       simd_load(va4,local_src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,local_dst+i);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }
  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get_src,left_count*sizeof(double));
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma(dma_get_src, (long)(src_ptr3+off), (long)(local_src3));
    dma(dma_get_src, (long)(src_ptr4+off), (long)(local_src4));
    dma_wait(&replyget, 4); replyget = 0;

    for(i=0; i+simd_size-1<left_count; i+=simd_size) {
       simd_load(va1,local_src1+i);
       simd_load(va2,local_src2+i);
       simd_load(va3,local_src3+i);
       va1 = va1 + va2;
       simd_load(va4,local_src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,local_dst+i);
    }
    for(;i<left_count;i++) {
      local_dst[i]=local_src1[i]+local_src2[i]+local_src3[i]+local_src4[i];
    }
    dma_set_size(&dma_put_dst,left_count*sizeof(double));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, buff_size*sizeof(double));
  ldm_free(local_src2, buff_size*sizeof(double));
  ldm_free(local_src3, buff_size*sizeof(double));
  ldm_free(local_src4, buff_size*sizeof(double));
  ldm_free(local_dst,  buff_size*sizeof(double));
}
void sw_slave_add_d(addPara *para) {
  const int buff_size = 2*1024;
  const int simd_size = 4;
  double* local_src1 = (double*)ldm_malloc(buff_size*sizeof(double));
  double* local_src2 = (double*)ldm_malloc(buff_size*sizeof(double));
  double* local_dst  = (double*)ldm_malloc(buff_size*sizeof(double));
  doublev4 vsrc1,vsrc2,vdst;
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double* src_ptr1 = &(((double*)para->src1)[start]);
  double* src_ptr2 = &(((double*)para->src2)[start]);
  double* dst_ptr  = &(((double *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,buff_size*sizeof(double));
  dma_set_size(&dma_put_dst,buff_size*sizeof(double));

  for(off = 0; off+buff_size-1<local_count; off+=buff_size)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i<buff_size; i+=simd_size) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get_src,left_count*sizeof(double));
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i+simd_size-1<left_count; i+=simd_size) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<left_count;i++) {
      local_dst[i]=local_src1[i]+local_src2[i];
    }
    dma_set_size(&dma_put_dst,left_count*sizeof(double));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, buff_size*sizeof(double));
  ldm_free(local_src2, buff_size*sizeof(double));
  ldm_free(local_dst, buff_size*sizeof(double));
}

void sw_slave_add_f(addPara *para) {
  const int buff_size = 4*1024;
  const int simd_size = 4;
  float * local_src1 = (float *)ldm_malloc(buff_size*sizeof(float ));
  float * local_src2 = (float *)ldm_malloc(buff_size*sizeof(float ));
  float * local_dst  = (float *)ldm_malloc(buff_size*sizeof(float ));
  floatv4 vsrc1,vsrc2,vdst;
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr1 = &(((float *)para->src1)[start]);
  float * src_ptr2 = &(((float *)para->src2)[start]);
  float * dst_ptr  = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,buff_size*sizeof(float));
  dma_set_size(&dma_put_dst,buff_size*sizeof(float));

  for(off = 0; off+buff_size-1<local_count; off+=buff_size)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i<buff_size; i+=simd_size) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get_src,left_count*sizeof(float));
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i+simd_size-1<left_count; i+=simd_size) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<left_count;i++) {
      local_dst[i]=local_src1[i]+local_src2[i];
    }
    dma_set_size(&dma_put_dst,left_count*sizeof(float));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, buff_size*sizeof(float));
  ldm_free(local_src2, buff_size*sizeof(float));
  ldm_free(local_dst, buff_size*sizeof(float));
}
void sw_slave_add_4cg_i(addPara *para) {
  const int buff_size = 2*1024;
  const int simd_size = 8;
  int* local_src1 = (int*)ldm_malloc(buff_size*sizeof(int));
  int* local_src2 = (int*)ldm_malloc(buff_size*sizeof(int));
  int* local_src3 = (int*)ldm_malloc(buff_size*sizeof(int));
  int* local_src4 = (int*)ldm_malloc(buff_size*sizeof(int));
  int* local_dst  = (int*)ldm_malloc(buff_size*sizeof(int));
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int* src_ptr1 = &(((int*)para->src1)[start]);
  int* src_ptr2 = &(((int*)para->src2)[start]);
  int* src_ptr3 = &(((int*)para->src3)[start]);
  int* src_ptr4 = &(((int*)para->src4)[start]);
  int* dst_ptr  = &(((int *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  intv8 va1,va2,va3,va4;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,buff_size*sizeof(int));
  dma_set_size(&dma_put_dst,buff_size*sizeof(int));

  for(off = 0; off+buff_size-1<local_count; off+=buff_size)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma(dma_get_src, (long)(src_ptr3+off), (long)(local_src3));
    dma(dma_get_src, (long)(src_ptr4+off), (long)(local_src4));
    dma_wait(&replyget, 4); replyget = 0;

    for(i=0; i<buff_size; i+=simd_size) {
       simd_load(va1,local_src1+i);
       simd_load(va2,local_src2+i);
       simd_load(va3,local_src3+i);
       va1 = va1 + va2;
       simd_load(va4,local_src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,local_dst+i);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }
  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get_src,left_count*sizeof(int));
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma(dma_get_src, (long)(src_ptr3+off), (long)(local_src3));
    dma(dma_get_src, (long)(src_ptr4+off), (long)(local_src4));
    dma_wait(&replyget, 4); replyget = 0;

    for(i=0; i+simd_size-1<left_count; i+=simd_size) {
       simd_load(va1,local_src1+i);
       simd_load(va2,local_src2+i);
       simd_load(va3,local_src3+i);
       va1 = va1 + va2;
       simd_load(va4,local_src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,local_dst+i);
    }
    for(;i<left_count;i++) {
      local_dst[i]=local_src1[i]+local_src2[i]+local_src3[i]+local_src4[i];
    }
    dma_set_size(&dma_put_dst,left_count*sizeof(int));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, buff_size*sizeof(int));
  ldm_free(local_src2, buff_size*sizeof(int));
  ldm_free(local_src3, buff_size*sizeof(int));
  ldm_free(local_src4, buff_size*sizeof(int));
  ldm_free(local_dst,  buff_size*sizeof(int));
}
void sw_slave_add_i(addPara *para) {
  const int buff_size = 2*1024;
  const int simd_size = 8;
  int* local_src1 = (int*)ldm_malloc(buff_size*sizeof(int));
  int* local_src2 = (int*)ldm_malloc(buff_size*sizeof(int));
  int* local_dst  = (int*)ldm_malloc(buff_size*sizeof(int));
  intv8 vsrc1,vsrc2;
  intv8 vdst;
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int* src_ptr1 = &(((int*)para->src1)[start]);
  int* src_ptr2 = &(((int*)para->src2)[start]);
  int* dst_ptr  = &(((int *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,buff_size*sizeof(int));
  dma_set_size(&dma_put_dst,buff_size*sizeof(int));

  for(off = 0; off+buff_size-1<local_count; off+=buff_size)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i<buff_size; i+=simd_size) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get_src,left_count*sizeof(int));
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma_wait(&replyget, 2); replyget = 0;

    for(i=0; i+simd_size-1<left_count; i+=simd_size) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; // 
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<left_count;i++) {
      local_dst[i]=local_src1[i]+local_src2[i];
    }
    dma_set_size(&dma_put_dst,left_count*sizeof(int));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, buff_size*sizeof(int));
  ldm_free(local_src2, buff_size*sizeof(int));
  ldm_free(local_dst, buff_size*sizeof(int));
}
void sw_slave_add_4cg_f(addPara *para) {
  const int buff_size = 2*1024;
  const int simd_size = 4;
  float* local_src1 = (float*)ldm_malloc(buff_size*sizeof(float));
  float* local_src2 = (float*)ldm_malloc(buff_size*sizeof(float));
  float* local_src3 = (float*)ldm_malloc(buff_size*sizeof(float));
  float* local_src4 = (float*)ldm_malloc(buff_size*sizeof(float));
  float* local_dst  = (float*)ldm_malloc(buff_size*sizeof(float));
  int id = athread_get_id(-1);
  long count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  long start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float* src_ptr1 = &(((float*)para->src1)[start]);
  float* src_ptr2 = &(((float*)para->src2)[start]);
  float* src_ptr3 = &(((float*)para->src3)[start]);
  float* src_ptr4 = &(((float*)para->src4)[start]);
  float* dst_ptr  = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  floatv4 va1,va2,va3,va4;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,buff_size*sizeof(float));
  dma_set_size(&dma_put_dst,buff_size*sizeof(float));

  for(off = 0; off+buff_size-1<local_count; off+=buff_size)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma(dma_get_src, (long)(src_ptr3+off), (long)(local_src3));
    dma(dma_get_src, (long)(src_ptr4+off), (long)(local_src4));
    dma_wait(&replyget, 4); replyget = 0;

    for(i=0; i<buff_size; i+=simd_size) {
       simd_load(va1,local_src1+i);
       simd_load(va2,local_src2+i);
       simd_load(va3,local_src3+i);
       va1 = va1 + va2;
       simd_load(va4,local_src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,local_dst+i);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }
  if(off<local_count) {
    int left_count = local_count-off;
    dma_set_size(&dma_get_src,left_count*sizeof(float));
    dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));
    dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));
    dma(dma_get_src, (long)(src_ptr3+off), (long)(local_src3));
    dma(dma_get_src, (long)(src_ptr4+off), (long)(local_src4));
    dma_wait(&replyget, 4); replyget = 0;

    for(i=0; i+simd_size-1<left_count; i+=simd_size) {
       simd_load(va1,local_src1+i);
       simd_load(va2,local_src2+i);
       simd_load(va3,local_src3+i);
       va1 = va1 + va2;
       simd_load(va4,local_src4+i);
       va1 = va1 + va3; 
       va1 = va1 + va4;
       simd_store(va1,local_dst+i);
    }
    for(;i<left_count;i++) {
      local_dst[i]=local_src1[i]+local_src2[i]+local_src3[i]+local_src4[i];
    }
    dma_set_size(&dma_put_dst,left_count*sizeof(float));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, buff_size*sizeof(float));
  ldm_free(local_src2, buff_size*sizeof(float));
  ldm_free(local_src3, buff_size*sizeof(float));
  ldm_free(local_src4, buff_size*sizeof(float));
  ldm_free(local_dst,  buff_size*sizeof(float));
}

