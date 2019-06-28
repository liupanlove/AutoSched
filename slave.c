#include <slave.h>
#include <simd.h>
#include <dma.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define SIMDSIZE 4
#define SIMDSIZE_INT 8
#define SPNUM 64
#define RECV_NUM 7
#define min(a,b) ((a)<(b) ? (a) : (b))
// 向同行目标从核送数 
#define LONG_PUTR(var,dest) asm volatile ("putr %0,%1\n"::"r"(var),"r"(dest):"memory") 
// 读行通信缓冲 
#define LONG_GETR(var)  asm volatile ("getr %0\n":"=r"(var)::"memory") 
// 向同列目标从核送数 
#define LONG_PUTC(var,dest)  asm volatile ("putc %0,%1\n"::"r"(var),"r"(dest):"memory") 
// 读列通信缓冲 
#define LONG_GETC(var)    asm volatile ("getc %0\n":"=r"(var)::"memory") 
//alignment
#define align(num, alignment_size) (num + ((alignment_size - (num % alignment_size)) % alignment_size))

unsigned long rtc() 
{
   unsigned long rpcc; 
   asm volatile("rcsr  %0, 4":"=r"(rpcc)); 
   return rpcc; 
}
inline void mb()
{
    asm volatile("memb");
    asm volatile("":::"memory");
}

typedef struct _KmeansPara{
  int dims;
  int data_size;
  int cluster_count;
  int *cluster_center_num;
  float*data;
  int rank;
  //int * data_group; // record the data belongs to which cluster
  float*cluster_center;
  float*cluster_center_out;
  //float* all_data;
} KmeansPara;  // it's same with para in master.c

typedef struct _EvaluationPara{
  char * out_filename;
  int rank;
  int dims;
  int data_size;
  int all_data_size;
  int cluster_count;
  int data_start;
  int * all_data_group;
  float * cluster_center;
  float * data;
  int * data_group;
  float * all_data;
  float * cluster_distance_count;
  float * non_cluster_distance_count;
  float * radius;
} EvaluationPara;

inline void add_f(float * src,float *dst,int count)
{
      floatv4 vsrc,vdst;
      int simd_size = 4,j,k;

      for(k=0;k+simd_size-1 < count; k += simd_size){
         simd_load(vsrc,src+k);
         simd_load(vdst,dst+k);
         vdst = vsrc + vdst;
         simd_store(vdst,dst+k);
      }
      for (; k < count; k++){
         dst[k] += src[k];
      }
}
void kmeans_normal(KmeansPara *para) {
  const int max_len = 59*1024;
  int i,j,c,k,off=0;
  int id = athread_get_id(-1);

  int count = para->data_size;
  int cluster_count = para->cluster_count;
  int dims = para->dims;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float* data_ptr = &(((float*)para->data)[start*dims]);
  /*if(id == 0)
  {
    //printf("count=%d, local_count=%d, start=%d", count, local_count, start);
  }*/
  volatile int replyget_data[2],replyput_center=0,replyget_center=0,replyput_center_num=0;
  dma_desc dma_get_data[2],dma_put_center,dma_get_center,dma_put_center_num;
  // DMA settings
  int center_size = cluster_count*dims;
  float *local_center = (float*)ldm_malloc(center_size*sizeof(float));
  int local_center_size = align(center_size, SIMDSIZE); //center_size + ((SIMDSIZE - (center_size % SIMDSIZE)) % SIMDSIZE); // center_size + (center_size % SIMDSIZE);
  float *local_center_temp = (float*)ldm_malloc(local_center_size*sizeof(float));
  for(i=0;i<local_center_size;i++)
    local_center_temp[i] = 0.0;
  int cluster_count_size = align(cluster_count, SIMDSIZE_INT); //cluster_count + ((SIMDSIZE_INT - (cluster_count%SIMDSIZE_INT)) % SIMDSIZE_INT); // cluster_count + (cluster_count%SIMDSIZE_INT);
  int *local_center_num = (int*)ldm_malloc(cluster_count_size*sizeof(int));
  for(i=0;i<cluster_count_size;i++)
    local_center_num[i] = 0;
  /*if(id == 0)
  {
    printf("center_size=%d  local_center_size=%d SIMDSIZE=%d \n", center_size, local_center_size, SIMDSIZE);
    printf("cluster_count=%d  cluster_count_size=%d SIMDSIZE_I=%d \n", cluster_count, cluster_count_size, SIMDSIZE_INT);
  }*/
  int buff_size = max_len - center_size*sizeof(float) - local_center_size*sizeof(float) - cluster_count_size*sizeof(int);
  int max_data_num = buff_size/(sizeof(float)*dims);
  int float_buff_data_num = max_data_num>>1;
  /*if(id == 0)
  {
    printf("buff_size=%d max_data_num=%d float_buff_data_num=%d", buff_size, max_data_num, float_buff_data_num);
  }*/
  assert(float_buff_data_num >0);
  float *local_data = (float*)ldm_malloc(buff_size);
  //采用DMA广播方式获取中心点
  dma_set_op(&dma_get_center, DMA_GET);
  //dma_set_mode(&dma_get_center, BCAST_MODE);
  //dma_set_mask(&dma_get_center,255);
  dma_set_mode(&dma_get_center, PE_MODE);
  dma_set_reply(&dma_get_center, &replyget_center);
  dma_set_size(&dma_get_center,center_size*sizeof(float));
  //Center DMA
  dma(dma_get_center, (long)(para->cluster_center), (long)(local_center));

  for(i=0;i<2;i++)
  {
    replyget_data[i] = 0;
    dma_set_op(&dma_get_data[i], DMA_GET);
    dma_set_mode(&dma_get_data[i], PE_MODE);
    dma_set_reply(&dma_get_data[i], &replyget_data[i]);
    dma_set_size(&dma_get_data[i],float_buff_data_num*dims*sizeof(float));
  }

  dma_set_op(&dma_put_center, DMA_PUT);
  dma_set_mode(&dma_put_center, PE_MODE);
  dma_set_reply(&dma_put_center, &replyput_center);
  dma_set_size(&dma_put_center,center_size*sizeof(float));

  dma_set_op(&dma_put_center_num, DMA_PUT);
  dma_set_mode(&dma_put_center_num, PE_MODE);
  dma_set_reply(&dma_put_center_num, &replyput_center_num);
  dma_set_size(&dma_put_center_num,cluster_count*sizeof(int));

  //Wait get center
  dma_wait(&replyget_center, 1); replyget_center = 0;
  float temp = 0,sum = 0,oldsum=0,arr[4];
  int min_index = 0,max_n = local_count/float_buff_data_num -1;
  int n=0,index = 0,next=0,buff_index = 0,cluster_index=0,data_index=0;
  int len = float_buff_data_num * dims;
  intv8 va,vb,vc;
  floatv4 vsrc, vdst ,vsum;

#ifdef DEBUG
  unsigned long compute_time=0,dma_time=0,run_time=0;
#endif
  //采用双buffer机制，计算各个点属于其中心点的各维坐标加和
  for(n=0,off = 0; off+float_buff_data_num-1 < local_count; off+=float_buff_data_num,n++)
  {
#ifdef DEBUG
    run_time = rtc();
#endif
    index = n%2;
    next = (n+1)%2;
  
    // DMA get a block
    if(n == 0)
    {
      replyget_data[index] = 0;
      dma(dma_get_data[index], (long)(data_ptr+off*dims), (long)(local_data+index*len));
    }
    if(n < max_n)
    {
      replyget_data[next] = 0;
      dma(dma_get_data[next], (long)(data_ptr+off*dims+len), (long)(local_data+next*len));
    } 
    dma_wait(&replyget_data[index], 1); replyget_data[index] = 0;
#ifdef DEBUG
    dma_time += rtc() - run_time;
    run_time = rtc();
#endif
    buff_index = index*len;
    for(i=0; i< float_buff_data_num; i++) {      
      oldsum = 0;
      vsum = 0;
      data_index = buff_index + i*dims;
      /*for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
         simd_load(vsrc,local_data+data_index+k);
         simd_load(vdst,local_center+k);
         vdst = vsrc - vdst;
         vsum += vdst * vdst;
      }
      simd_store(vsum,arr);
      for(j=0;j<4;j++) oldsum += arr[j];*/
      for (k=0; k < dims; k++){
         temp = local_data[data_index+k] - local_center[k];
         oldsum += temp*temp;
      }
      min_index = 0;
      for( c = 1; c < cluster_count; c++ )  {
        cluster_index = c*dims;
        sum = 0;
        /*vsum = 0;
        for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
           simd_load(vsrc,local_data+data_index+k);
           simd_load(vdst,local_center+cluster_index+k);
           vdst = vsrc - vdst;
           vsum += vdst * vdst;
        }
        simd_store(vsum,arr);
        for(j=0;j<4;j++) sum += arr[j];*/
        for (k=0; k < dims; k++){
           temp = local_data[data_index+k] - local_center[cluster_index+k];
           sum += temp*temp;
        }
        if(sum < oldsum) 
        {
          min_index = c;
          oldsum = sum;
        }
      }
      cluster_index = min_index*dims;
      local_center_num[min_index]++;
      /*for(k=0;k+SIMDSIZE-1 < dims;k+=SIMDSIZE)
      {
        simd_load(vsrc,&local_center_temp[cluster_index+k]);
        simd_load(vdst,&local_data[data_index+k]);
        vsrc = vsrc + vdst;
        simd_store(vsrc,&local_center_temp[cluster_index+k]);
      }*/
      for (k=0; k < dims; k++){
         local_center_temp[cluster_index + k] += local_data[data_index+k];
      }
    }
#ifdef DEBUG
    compute_time += rtc() - run_time;
#endif
  } 

  if(off<local_count) {
#ifdef DEBUG
    run_time = rtc();
#endif
    index = 0;
    dma_set_size(&dma_get_data[index],(local_count-off)*dims*sizeof(float));
    dma(dma_get_data[index], (long)(data_ptr+off*dims), (long)(local_data));
    dma_wait(&replyget_data[index], 1); replyget_data[index] = 0;
#ifdef DEBUG   
    dma_time += rtc() - run_time;
    run_time = rtc();
#endif
    int tmp_count = local_count-off;
    for(i=0; i< tmp_count; i++) {
      oldsum = 0;
      vsum = 0;
      data_index = i*dims;
      /*for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
         simd_load(vsrc,local_data+data_index+k);
         simd_load(vdst,local_center+k);
         vdst = vsrc - vdst;
         vsum += vdst * vdst;
      }
      simd_store(vsum,arr);
      for(j=0;j<4;j++) oldsum += arr[j];*/
      for (k=0; k < dims; k++){
         temp = local_data[data_index+k] - local_center[k];
         oldsum += temp*temp;
      }
      min_index = 0;
      for( c = 1; c < cluster_count; c++ )  {
        sum = 0;
        vsum = 0;
        cluster_index = c*dims;
       /* for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
           simd_load(vsrc,local_data+data_index+k);
           simd_load(vdst,local_center+cluster_index+k);
           vdst = vsrc - vdst;
           vsum += vdst * vdst;
        }
        simd_store(vsum,arr);
        for(j=0;j<4;j++) sum += arr[j];*/
        for (k=0; k < dims; k++){
           temp = local_data[data_index+k] - local_center[cluster_index+k];
           sum += temp*temp;
        }
        if(sum < oldsum)
        {
          min_index = c;
          oldsum = sum;
        }
      }
      cluster_index = min_index*dims;
      local_center_num[min_index]++;
      /*for(k=0;k+SIMDSIZE-1 < dims;k+=SIMDSIZE)
      {
        simd_load(vsrc,&local_center_temp[cluster_index+k]);
        simd_load(vdst,&local_data[data_index+k]);
        vsrc = vsrc + vdst;
        simd_store(vsrc,&local_center_temp[cluster_index+k]);
      }*/
      for (k=0; k < dims; k++){
         local_center_temp[cluster_index + k] += local_data[data_index+k];
      }
    }
#ifdef DEBUG
    compute_time += rtc() - run_time;
#endif
  }//每个中心聚点，每一维坐标加和完成

#ifdef DEBUG
  unsigned long m1 = rtc() -run_time;
  run_time = rtc();
#endif
  for( i = 0; i < cluster_count_size; i+=SIMDSIZE_INT)  {
    simd_load(va, local_center_num+i);
    //simd_print_intv8(va);
    LONG_PUTR(va,8);
    for(j=0;j<RECV_NUM;j++)
    {
      LONG_GETR(vb);
      va = va + vb;
    }
    LONG_PUTC(va,8);
    for(j=0;j<RECV_NUM;j++)
    {
      LONG_GETC(vb);
      va = va + vb;
    }
    simd_store(va,local_center_num+i);
  }
  //通过寄存器通信完成64个从核中心聚点加和
  for(i=0;i<local_center_size;i+=SIMDSIZE)
  {
    simd_load(vsrc,local_center_temp+i);
    LONG_PUTR(vsrc,8);
    for(j=0;j<RECV_NUM;j++)
    {
      LONG_GETR(vdst);
      vsrc = vsrc + vdst;
    }	
    LONG_PUTC(vsrc,8);
    for(j=0;j<RECV_NUM;j++)
    {
      LONG_GETC(vdst);
      vsrc = vsrc + vdst;
    }	
    simd_store(vsrc,local_center_temp+i);
  }

#ifdef DEBUG
  unsigned long m2 = rtc() -run_time;
  run_time = rtc();
#endif
#ifdef DEBUG
  unsigned long register_time = rtc() - run_time;
  int size = float_buff_data_num * dims*sizeof(float);
  if(id<1)printf("buff size=%d register_time=%d dma_time=%d compute_time = %d max_data_num=%d float_buff_data_num=%d m1=%d m2=%d\n",\
       size,register_time,dma_time,compute_time,max_data_num,float_buff_data_num,m1,m2);
#endif
  //Complete
  if(id == 0)
  {
#ifdef DEBUG
    run_time = rtc();
#endif
    dma(dma_put_center, (long)(para->cluster_center_out), (long)(local_center_temp));
    dma_wait(&replyput_center, 1); replyput_center = 0;
    dma(dma_put_center_num, (long)(para->cluster_center_num), (long)(local_center_num));
    dma_wait(&replyput_center_num, 1); replyput_center_num = 0;
#ifdef DEBUG
    printf("result dma time = %d\n",rtc()-run_time);
#endif
  } 
  ldm_free(local_center_num, cluster_count_size*sizeof(int));  
  ldm_free(local_center, center_size*sizeof(float));
  ldm_free(local_center_temp, local_center_size*sizeof(float));
  ldm_free(local_data, buff_size);
}

void kmeans_dims_large(KmeansPara *para) {
  const int max_len = 59*1024;
  int i,j,k,c,off=0;
  int id = athread_get_id(-1);
  //if(para->rank == 0 && id == 63) printf("kmeans_dims_large start id=%d \n", id);
  int count = para->data_size;
  int cluster_count = para->cluster_count;
  int dims = para->dims;
  volatile int replyget_data=0,replyput_center=0,replyget_center=0,replyput_center_num=0;
  dma_desc dma_get_data,dma_put_center,dma_get_center,dma_put_center_num;
  int local_dims = dims/SPNUM + (id<(dims%SPNUM));
  int start_dims = id*(dims/SPNUM) + (id<(dims%SPNUM)?id:(dims%SPNUM));
  int left_space = max_len - ((2*cluster_count)* local_dims * sizeof(float) + cluster_count*sizeof(int));
  assert(left_space > 0);
  int max_data_num = left_space / (local_dims*sizeof(float));
  assert(max_data_num > 0);
  int center_size = cluster_count*local_dims;
  int local_center_size = center_size;
  int cluster_count_size = cluster_count;
  int local_center_count = cluster_count;
  int* cluster_center_num = (int*)para->cluster_center_num;
  float* data_ptr = (float*)para->data;
  float* cluster_center = (float*)para->cluster_center;
  float* cluster_center_out = (float*)para->cluster_center_out;

  float *local_data   = (float*)ldm_malloc(max_data_num*local_dims*sizeof(float));
  float *local_center = (float*)ldm_malloc(center_size*sizeof(float));
  float *local_center_temp = (float*)ldm_malloc(local_center_size*sizeof(float));
  for(i=0;i<local_center_size;i++)
    local_center_temp[i] = 0.0;

  int *local_center_num = (int*)ldm_malloc(cluster_count_size*sizeof(int));
  for(i=0;i<cluster_count_size;i++)
    local_center_num[i] = 0;

  // DMA settings
  dma_set_op(&dma_get_center, DMA_GET);
  dma_set_mode(&dma_get_center, PE_MODE);
  dma_set_reply(&dma_get_center, &replyget_center);
  dma_set_size(&dma_get_center,cluster_count*local_dims*sizeof(float));
  dma_set_bsize(&dma_get_center,local_dims*sizeof(float));
  dma_set_stepsize(&dma_get_center,(dims - local_dims)*sizeof(float));

  dma_set_op(&dma_get_data, DMA_GET);
  dma_set_mode(&dma_get_data, PE_MODE);
  dma_set_reply(&dma_get_data, &replyget_data);
  dma_set_size(&dma_get_data,max_data_num*local_dims*sizeof(float));  
  dma_set_bsize(&dma_get_data,local_dims*sizeof(float));
  dma_set_stepsize(&dma_get_data,(dims - local_dims)*sizeof(float));

  dma_set_op(&dma_put_center, DMA_PUT);
  dma_set_mode(&dma_put_center, PE_MODE);
  dma_set_reply(&dma_put_center, &replyput_center); 
  dma_set_size(&dma_put_center,cluster_count*local_dims*sizeof(float));
  dma_set_bsize(&dma_put_center,local_dims*sizeof(float));
  dma_set_stepsize(&dma_put_center,(dims - local_dims)*sizeof(float));

  dma_set_op(&dma_put_center_num, DMA_PUT);
  dma_set_mode(&dma_put_center_num, PE_MODE);
  dma_set_reply(&dma_put_center_num, &replyput_center_num);
  dma_set_size(&dma_put_center_num,cluster_count*sizeof(int));
  float temp = 0,sum = 0,oldsum=0,arr[4];
  floatv4 vold,vnew,vsrc,vdst,vsum;
  int min_index = 0,index = 0;
  int data_index = 0,cluster_index =0;
  long offset = 0;
#ifdef DEBUG
  unsigned long all_time = rtc(), dma_time = 0, run_time = rtc(), communication_time = 0;
#endif
  //Center DMA
  dma(dma_get_center, (long)(cluster_center + start_dims), (long)(local_center));
  dma_wait(&replyget_center, 1); replyget_center = 0;
#ifdef DEBUG
  dma_time += (unsigned long)rtc() - run_time;
  //run_time = rtc();
#endif
  for(off = 0; off+max_data_num-1 < count; off+=max_data_num)
  {
      // DMA get a block
      offset = off;
      offset *= dims;
      offset += start_dims;
#ifdef DEBUG
      run_time = rtc();
#endif
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
#ifdef DEBUG
      dma_time += (unsigned long)rtc() - run_time;
      run_time = rtc();
#endif
      /*if(id == 1){
        printf("max_data_num=%d\n", max_data_num);
        for(i = 0; i < local_dims * 2; ++i)
        {
          printf("%f ", local_data[i]);
          if(i % local_dims == (local_dims - 1)) printf("\n");
        }
        printf("\n");
      }*/
      for(i=0; i< max_data_num; i++) {
        oldsum = FLT_MAX;
        data_index = i*local_dims;
        min_index = 0;
        for(c = 0; c < cluster_count;c++)
        {
          sum = 0;
          vsum = 0;
          cluster_index = c*local_dims;
          for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
             simd_loadu(vsrc,local_data+data_index+k);
             simd_loadu(vdst,local_center+cluster_index+k);
             vdst = vsrc - vdst;
             vsum += vdst * vdst;
          }
          simd_store(vsum,arr);
          for(j=0;j<4;j++) sum += arr[j];
          for (; k < local_dims; k++){
             temp = local_data[data_index+k] - local_center[cluster_index+k];
             sum += temp*temp;
          }
#ifdef DEBUG
          run_time = rtc();
#endif
          vold = sum;
          LONG_PUTR(vold,8);
          //simd_loader(vold,&sum);
          for(j=0;j<RECV_NUM;j++)
          {
             LONG_GETR(vnew);
             vold = vold + vnew;
          }
          LONG_PUTC(vold,8);
          for(j=0;j<RECV_NUM;j++)
          {
             LONG_GETC(vnew);
             vold = vold + vnew;
          }
          simd_store(vold,arr);
          sum = arr[0];
          if(sum < oldsum)
          {
            min_index = c;
            oldsum = sum;
          }
#ifdef DEBUG
          communication_time += (unsigned long)rtc() - run_time;
#endif
        }
        local_center_num[min_index]++;
        index = min_index*local_dims;
        for(k=0;k+SIMDSIZE-1 < local_dims;k+=SIMDSIZE)
        {
          simd_loadu(vsrc,&local_center_temp[index+k]);
          simd_loadu(vdst,&local_data[data_index+k]);
          vsrc = vsrc + vdst;
          simd_storeu(vsrc,&local_center_temp[index+k]);
        }
        for (; k < local_dims; k++){
           local_center_temp[index + k] += local_data[data_index+k];
        }
     }
    }
    if(off < count) {
      int left_count = count - off;
      offset = off;
      offset *= dims;
      offset += start_dims;
#ifdef DEBUG
      run_time = rtc();
#endif
      dma_set_size(&dma_get_data,left_count * local_dims * sizeof(float));
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
#ifdef DEBUG
      dma_time += (unsigned long)rtc() - run_time;
      run_time = rtc();
#endif
      for(i=0; i< left_count; i++) {
        oldsum = FLT_MAX;
        data_index = i*local_dims;
        min_index = 0;
        for(c = 0; c < cluster_count; c++)
        {
          sum = 0;
          vsum = 0;
          cluster_index = c*local_dims;
          for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
             simd_loadu(vsrc,local_data+data_index+k);
             simd_loadu(vdst,local_center+cluster_index+k);
             vdst = vsrc - vdst;
             vsum += vdst * vdst;
          }
          simd_store(vsum,arr);
          for(j=0;j<4;j++) sum += arr[j];
          for (; k < local_dims; k++){
             temp = local_data[data_index+k] - local_center[cluster_index+k];
             sum += temp*temp;
          }
#ifdef DEBUG
          run_time = rtc();

#endif
          vold = sum;
          LONG_PUTR(vold,8);
          //simd_loader(vold,&sum);
          for(j=0;j<RECV_NUM;j++)
          {
             LONG_GETR(vnew);
             vold = vold + vnew;
          }
          LONG_PUTC(vold,8);
          for(j=0;j<RECV_NUM;j++)
          {
             LONG_GETC(vnew);
             vold = vold + vnew;
          }
          simd_store(vold,arr);
          sum = arr[0];
          if(sum < oldsum)
          {
            min_index = c;
            oldsum = sum;
          }
#ifdef DEBUG
          communication_time += (unsigned long)rtc() - run_time;
#endif
        }
        local_center_num[min_index]++;
        index = min_index*local_dims;
        for(k=0;k+SIMDSIZE-1 < local_dims;k+=SIMDSIZE)
        {
          simd_loadu(vsrc,&local_center_temp[index+k]);
          simd_loadu(vdst,&local_data[data_index+k]);
          vsrc = vsrc + vdst;
          simd_storeu(vsrc,&local_center_temp[index+k]);
        }
        for (; k < local_dims; k++){
           local_center_temp[index + k] += local_data[data_index+k];
        }
      }
  }//每个中心聚点，每一维坐标加和完成
#ifdef DEBUG
  run_time = rtc();
#endif
  mb();
  dma(dma_put_center, (long)(cluster_center_out + start_dims), (long)(local_center_temp));
  dma_wait(&replyput_center, 1); replyput_center = 0;
  mb();
  if(id == 0){
    dma(dma_put_center_num, (long)(cluster_center_num), (long)(local_center_num));
    dma_wait(&replyput_center_num, 1); replyput_center_num = 0;
  }
#ifdef DEBUG
  dma_time += (unsigned long)rtc() - run_time;
  all_time = (unsigned long)rtc() - all_time;
  if(id == 0) printf("dma_time=%lu  all_time=%lu communication_time=%lu\n", dma_time, all_time, communication_time);
#endif
  ldm_free(local_center_num, cluster_count_size*sizeof(int));
  ldm_free(local_center, center_size*sizeof(float));
  ldm_free(local_center_temp, local_center_size*sizeof(float));
  ldm_free(local_data,max_data_num*local_dims*sizeof(float));
}

void kmeans_cluster_count_large(KmeansPara *para) {
  const int max_len = 59*1024;
  int i,j,k,c,off=0;
  int id = athread_get_id(-1);
  int count = para->data_size;
  int cluster_count = para->cluster_count;
  int dims = para->dims;
  volatile int replyget_data=0,replyput_center=0,replyget_center=0,replyput_center_num=0;
  dma_desc dma_get_data,dma_put_center,dma_get_center,dma_put_center_num;

  int local_cluster_count = cluster_count/SPNUM + (id<(cluster_count%SPNUM));
  int start_cluster_count = id*(cluster_count/SPNUM) + (id<(cluster_count%SPNUM)?id:(cluster_count%SPNUM));
  int left_space = max_len - ((2*local_cluster_count)* dims * sizeof(float) + local_cluster_count*sizeof(int));
  int max_data_num = left_space / (dims*sizeof(float));
  assert(max_data_num > 0);
  int center_size = local_cluster_count*dims;
  int local_center_size = center_size;  
  int cluster_count_size = local_cluster_count;
  int local_center_count = start_cluster_count + local_cluster_count;

  int* cluster_center_num = (int*)para->cluster_center_num;
  float* data_ptr = (float*)para->data;
  float* cluster_center = (float*)para->cluster_center;
  float* cluster_center_out = (float*)para->cluster_center_out;
  float *local_data   = (float*)ldm_malloc(max_data_num*dims*sizeof(float));
  float *local_center = (float*)ldm_malloc(center_size*sizeof(float));
  float *local_center_temp = (float*)ldm_malloc(local_center_size*sizeof(float));
  for(i=0;i<local_center_size;i++)
    local_center_temp[i] = 0.0;
  
  int *local_center_num = (int*)ldm_malloc(cluster_count_size*sizeof(int));
  for(i=0;i<cluster_count_size;i++)
    local_center_num[i] = 0;

  // DMA settings
  dma_set_op(&dma_get_center, DMA_GET);
  dma_set_mode(&dma_get_center, PE_MODE);
  dma_set_reply(&dma_get_center, &replyget_center);
  dma_set_size(&dma_get_center,local_cluster_count*dims*sizeof(float));

  dma_set_op(&dma_get_data, DMA_GET);
  dma_set_mode(&dma_get_data, PE_MODE);
  dma_set_reply(&dma_get_data, &replyget_data);
  dma_set_size(&dma_get_data,max_data_num*dims*sizeof(float));  

  dma_set_op(&dma_put_center, DMA_PUT);
  dma_set_mode(&dma_put_center, PE_MODE);
  dma_set_reply(&dma_put_center, &replyput_center); 
  dma_set_size(&dma_put_center,local_cluster_count*dims*sizeof(float));
  
  dma_set_op(&dma_put_center_num, DMA_PUT);
  dma_set_mode(&dma_put_center_num, PE_MODE);
  dma_set_reply(&dma_put_center_num, &replyput_center_num); 
  dma_set_size(&dma_put_center_num,local_cluster_count*sizeof(int));
  float temp = 0,sum = 0,oldsum=0,arr[4];
  floatv4 vold,vnew,vsrc,vdst,vsum;
  int min_index = 0,index = 0;
  int data_index = 0,cluster_index =0;
#ifdef DEBUG
  unsigned long compute_time=0,dma_time=0,run_time=0;
#endif
   //Center DMA
  dma(dma_get_center, (long)(cluster_center + start_cluster_count*dims), (long)(local_center));
  dma_wait(&replyget_center, 1); replyget_center = 0;
  for(off = 0; off+max_data_num-1 < count; off+=max_data_num)
  {
#ifdef DEBUG
      run_time = rtc();
#endif
      // DMA get a block
      dma(dma_get_data, (long)(data_ptr + off*dims), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
#ifdef DEBUG
      dma_time += rtc() - run_time;
      run_time = rtc();
#endif
      for(i=0; i< max_data_num; i++) {      
        oldsum = 0;
        vsum = 0;
        data_index = i*dims;
        for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
             simd_loadu(vsrc,local_data+data_index+k);
             simd_load(vdst,local_center+k);
             vdst = vsrc - vdst;
             vsum += vdst * vdst;
        }
        simd_store(vsum,arr);
        for(j=0;j<4;j++) oldsum += arr[j];
        for (; k < dims; k++){
           temp = local_data[data_index+k] - local_center[k];
           oldsum += temp*temp;
        }
        min_index = start_cluster_count;
        for(c = 1; c < local_cluster_count;c++)
        {
          sum = 0;
          vsum = 0;
          cluster_index = c*dims;
          for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
             simd_loadu(vsrc,local_data+data_index+k);
             simd_loadu(vdst,local_center+cluster_index+k);
             vdst = vsrc - vdst;
             vsum += vdst * vdst;
          }
          simd_store(vsum,arr);
          for(j=0;j<4;j++) sum += arr[j];
          for (; k < dims; k++){
             temp = local_data[data_index+k] - local_center[cluster_index+k];
             sum += temp*temp;
          }
          if(sum < oldsum){
            oldsum = sum;
            min_index = c + start_cluster_count;
          }
        }
        arr[0] = min_index;
        arr[1] = oldsum;
        arr[2] = oldsum;
        arr[3] = oldsum;
        
        simd_load(vold,arr);
        LONG_PUTR(vold,8);
        //simd_loadr(vold,arr);
        for(j=0;j<RECV_NUM;j++)
        {
          LONG_GETR(vnew);
          simd_store(vnew,arr);
          if(arr[1] < oldsum)
          {
             min_index = arr[0];
             oldsum = arr[1];
          } 
        }	
    
        arr[0] = min_index;
        arr[1] = oldsum;
        arr[2] = oldsum;
        arr[3] = oldsum;

        simd_load(vold,arr);
        LONG_PUTC(vold,8);
        for(j=0;j<RECV_NUM;j++)
        {
           LONG_GETC(vnew);
           simd_store(vnew,arr);
           if(arr[1] < oldsum)
           {
             min_index = arr[0];
             oldsum = arr[1];
           } 
        }
        if(min_index >= start_cluster_count && min_index < local_center_count){
          index = min_index - start_cluster_count;
          local_center_num[index]++;
          index = index*dims;
          for(k=0;k+SIMDSIZE-1 < dims;k+=SIMDSIZE)
          {
            simd_loadu(vsrc,&local_center_temp[index+k]);
            simd_loadu(vdst,&local_data[data_index+k]);
            vsrc = vsrc + vdst;
            simd_storeu(vsrc,&local_center_temp[index+k]);
          }
          for (; k < dims; k++){
            local_center_temp[index + k] += local_data[data_index+k];
          }
        }
     }
#ifdef DEBUG
      compute_time += rtc() - run_time;
#endif
    } 
    if(off < count) {
#ifdef DEBUG
      run_time = rtc();
#endif
      int left_count = count-off;
      dma_set_size(&dma_get_data,left_count * dims * sizeof(float));
      dma(dma_get_data, (long)(data_ptr+off*dims), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
#ifdef DEBUG   
      dma_time += rtc() - run_time;
      run_time = rtc();
#endif
      for(i=0; i< left_count; i++) {      
        oldsum = 0;
        vsum = 0;
        data_index = i*dims;
        for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
             simd_loadu(vsrc,local_data+data_index+k);
             simd_load(vdst,local_center+k);
             vdst = vsrc - vdst;
             vsum += vdst * vdst;
        }
        simd_store(vsum,arr);
        for(j=0;j<4;j++) oldsum += arr[j];
        for (; k < dims; k++){
           temp = local_data[data_index+k] - local_center[k];
           oldsum += temp*temp;
        }
        min_index = start_cluster_count;
        for(c = 1; c < local_cluster_count;c++)
        {
          sum = 0;
          vsum = 0;
          cluster_index = c*dims;
          for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
             simd_loadu(vsrc,local_data+data_index+k);
             simd_loadu(vdst,local_center+cluster_index+k);
             vdst = vsrc - vdst;
             vsum += vdst * vdst;
          }
          simd_store(vsum,arr);
          for(j=0;j<4;j++) sum += arr[j];
          for (; k < dims; k++){
             temp = local_data[data_index+k] - local_center[cluster_index+k];
             sum += temp*temp;
          }
          if(sum < oldsum) 
          {
            min_index = c+start_cluster_count;
            oldsum = sum;
          }
        }
        arr[0] = min_index;
        arr[1] = oldsum;
        arr[2] = oldsum;
        arr[3] = oldsum;
        
        simd_load(vold,arr);
        LONG_PUTR(vold,8);
        for(j=0;j<RECV_NUM;j++)
        {
          LONG_GETR(vnew);
          simd_store(vnew,arr);
          if(arr[1] < oldsum)
          {
             min_index = arr[0];
             oldsum = arr[1];
          } 
        }	
    
        arr[0] = min_index;
        arr[1] = oldsum;
        arr[2] = oldsum;
        arr[3] = oldsum;

        simd_load(vold,arr);
        LONG_PUTC(vold,8);
        for(j=0;j<RECV_NUM;j++)
        {
          LONG_GETC(vnew);
          simd_store(vnew,arr);
          if(arr[1] < oldsum)
          {
             min_index = arr[0];
             oldsum = arr[1];
          } 
        }
        if(min_index >= start_cluster_count && min_index < local_center_count){
          index = min_index - start_cluster_count;
          local_center_num[index]++;
          index = index*dims;
          for(k=0;k+SIMDSIZE-1 < dims;k+=SIMDSIZE)
          {
            simd_loadu(vsrc,&local_center_temp[index+k]);
            simd_loadu(vdst,&local_data[data_index+k]);
            vsrc = vsrc + vdst;
            simd_storeu(vsrc,&local_center_temp[index+k]);
          }
          for (; k < dims; k++){
            local_center_temp[index + k] += local_data[data_index+k];
          }
        }
      }
#ifdef DEBUG
      compute_time += rtc() - run_time;
#endif
  }//每个中心聚点，每一维坐标加和完成

#ifdef DEBUG
  unsigned long m2 = rtc() -run_time;
  run_time = rtc();
#endif
  mb();
  dma(dma_put_center, (long)(cluster_center_out + start_cluster_count*dims), (long)(local_center_temp));
  dma_wait(&replyput_center, 1); replyput_center = 0;
  dma(dma_put_center_num, (long)(cluster_center_num + start_cluster_count), (long)(local_center_num));
  dma_wait(&replyput_center_num, 1); replyput_center_num = 0;
  mb();
  
  ldm_free(local_center_num, cluster_count_size*sizeof(int));  
  ldm_free(local_center, center_size*sizeof(float));
  ldm_free(local_center_temp, local_center_size*sizeof(float));
  ldm_free(local_data,max_data_num*dims*sizeof(float));

}

void kmeans_both_large(KmeansPara *para) {
  const int max_len = 59*1024;
  int i,j,k,c,off=0;
  int id = athread_get_id(-1);
  int count = para->data_size;
  int cluster_count = para->cluster_count;
  int dims = para->dims;
  volatile int replyget_data=0,replyput_center=0,replyget_center=0,replyput_center_num=0,replyget_center_temp=0;
  dma_desc dma_get_data,dma_put_center,dma_get_center,dma_put_center_num,dma_get_center_temp;
  int local_cluster_count = cluster_count/SPNUM + (id<(cluster_count%SPNUM));
  int start_cluster_count = id*(cluster_count/SPNUM) + (id<(cluster_count%SPNUM)?id:(cluster_count%SPNUM));
  int local_center_count = start_cluster_count + local_cluster_count;
  int local_dims = dims/SPNUM + (id<(dims%SPNUM));
  int start_dims = id*(dims/SPNUM) + (id<(dims%SPNUM)?id:(dims%SPNUM));
  int left_space = (max_len - local_cluster_count*sizeof(int) - local_dims*sizeof(float));
  assert(left_space > 0);
  int max_data_num = left_space / ((2*local_dims+1)*sizeof(float) + sizeof(int));
  assert(max_data_num > 0);
  int* cluster_center_num = (int*)para->cluster_center_num;
  float* data_ptr = (float*)para->data;
  float* cluster_center = (float*)para->cluster_center;
  float* cluster_center_out = (float*)para->cluster_center_out;
  float *local_center = (float*)ldm_malloc(max_data_num*local_dims*sizeof(float));
  float *local_data   = (float*)ldm_malloc(max_data_num*local_dims*sizeof(float));
  float *local_oldsum = (float*)ldm_malloc(max_data_num*sizeof(float));
  int *local_data_index = (int*)ldm_malloc(max_data_num*sizeof(int));
  int *local_center_num = (int*)ldm_malloc(local_cluster_count*sizeof(int));
  float *local_center_temp = (float*)ldm_malloc(local_dims*sizeof(float));
  for(i=0;i<local_cluster_count;i++)
    local_center_num[i] = 0;
  // DMA settings
  dma_set_op(&dma_get_center, DMA_GET);
  dma_set_mode(&dma_get_center, PE_MODE);
  dma_set_reply(&dma_get_center, &replyget_center);
  dma_set_size(&dma_get_center,max_data_num*local_dims*sizeof(float));
  dma_set_bsize(&dma_get_center,local_dims*sizeof(float));
  dma_set_stepsize(&dma_get_center,(dims - local_dims)*sizeof(float));

  dma_set_op(&dma_get_data, DMA_GET);
  dma_set_mode(&dma_get_data, PE_MODE);
  dma_set_reply(&dma_get_data, &replyget_data);
  dma_set_size(&dma_get_data,max_data_num*local_dims*sizeof(float));
  dma_set_bsize(&dma_get_data,local_dims*sizeof(float));
  dma_set_stepsize(&dma_get_data,(dims - local_dims)*sizeof(float));

  dma_set_op(&dma_get_center_temp, DMA_GET);
  dma_set_mode(&dma_get_center_temp, PE_MODE);
  dma_set_reply(&dma_get_center_temp, &replyget_center_temp);
  dma_set_size(&dma_get_center_temp,local_dims*sizeof(float));

  dma_set_op(&dma_put_center, DMA_PUT);
  dma_set_mode(&dma_put_center, PE_MODE);
  dma_set_reply(&dma_put_center, &replyput_center);
  dma_set_size(&dma_put_center,local_dims*sizeof(float));

  dma_set_op(&dma_put_center_num, DMA_PUT);
  dma_set_mode(&dma_put_center_num, PE_MODE);
  dma_set_reply(&dma_put_center_num, &replyput_center_num);
  dma_set_size(&dma_put_center_num,local_cluster_count*sizeof(int));
  float temp = 0,sum = 0,arr[4],*p_data,*c_data;
  floatv4 vold,vnew,vsrc,vdst,vsum;
  int index = 0,m=0;
  long offset = 0;
#ifdef DEBUG
  unsigned long compute_time=0,dma_time=0,run_time=0;
#endif

  for(off = 0; off+max_data_num-1 < count; off+=max_data_num)
  {
#ifdef DEBUG
      run_time = rtc();
#endif
      // DMA get a block
      offset = off;
      offset *= dims;
      offset += start_dims;
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
#ifdef DEBUG
      dma_time += rtc() - run_time;
      run_time = rtc();
#endif

      for(c = 0; c+max_data_num-1 < cluster_count;c+=max_data_num)
      {
        //Center DMA
        dma_set_size(&dma_get_center,max_data_num*local_dims*sizeof(float));
        dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
        dma_wait(&replyget_center, 1); replyget_center = 0;
        for(i=0; i< max_data_num; i++) {
          p_data = local_data + i *local_dims;
          for(m = 0;m < max_data_num;m++){
             sum = 0;
             vsum = 0;
             c_data = local_center + m*local_dims;
             for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
             }
             simd_store(vsum,arr);
             for(j=0;j<4;j++) sum += arr[j];
             for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
             }
             vold = sum;
             LONG_PUTR(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETR(vnew);
               vold = vold + vnew;
             }

             LONG_PUTC(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETC(vnew);
               vold = vold + vnew;
             }
             simd_store(vold,arr);
             sum = arr[0];
             if(c==0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
             }
             else if(sum < local_oldsum[i])
             {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
             }
           }
          }
        }
        if(c < cluster_count){
           //Center DMA
           int left_cluster_count = cluster_count - c;
           dma_set_size(&dma_get_center,left_cluster_count*local_dims*sizeof(float));
           dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
           dma_wait(&replyget_center, 1); replyget_center = 0;
           for(i=0; i< max_data_num; i++) {
            p_data = local_data + i *local_dims;
            for(m = 0;m < left_cluster_count;m++){
              sum = 0;
              vsum = 0;
              c_data = local_center +  m*local_dims;
              for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
              }
              simd_store(vsum,arr);
              for(j=0;j<4;j++) sum += arr[j];
              for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
              }
              vold = sum;
              LONG_PUTR(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETR(vnew);
                vold = vold + vnew;
              }

              LONG_PUTC(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETC(vnew);
                vold = vold + vnew;
              }
              simd_store(vold,arr);
              sum = arr[0];
              if(c == 0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
              }
              else if(sum < local_oldsum[i]) 
              {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
              }
            }
          }
        }

        for(i=0; i< max_data_num; i++) {
          if(local_data_index[i] >= start_cluster_count && local_data_index[i] < local_center_count){
            index = local_data_index[i] - start_cluster_count;
            local_center_num[index]++;
          }
          p_data = local_data + i*local_dims;
          dma(dma_get_center_temp, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyget_center_temp, 1); replyget_center_temp = 0;
          for(k=0;k+SIMDSIZE-1 < local_dims;k+=SIMDSIZE)
          {
              simd_load(vsrc,local_center_temp+k);
              simd_loadu(vdst,p_data+k);
              vsrc = vsrc + vdst;
              simd_store(vsrc,local_center_temp+k);
          }
          for (; k < local_dims; k++){
              local_center_temp[k] += p_data[k];
          }
          mb();
          dma(dma_put_center, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyput_center, 1); replyput_center = 0;
       }
#ifdef DEBUG
      compute_time += rtc() - run_time;
#endif
    }
    if(off < count) {
#ifdef DEBUG
      run_time = rtc();
#endif
      int left_data_num = count - off; 
      offset = off;
      offset *= dims;
      offset += start_dims;

      dma_set_size(&dma_get_data,left_data_num * local_dims * sizeof(float));
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
#ifdef DEBUG   
      dma_time += rtc() - run_time;
      run_time = rtc();
#endif
      for(c = 0; c+max_data_num-1 < cluster_count;c+=max_data_num)
      {
        //Center DMA
        dma_set_size(&dma_get_center,max_data_num*local_dims*sizeof(float));
        dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
        dma_wait(&replyget_center, 1); replyget_center = 0;
        for(i=0; i< left_data_num; i++) {      
          p_data = local_data + i *local_dims;
          for(m = 0;m < max_data_num;m++){
             sum = 0;
             vsum = 0;
             c_data = local_center + m*local_dims;
             for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
             }
             simd_store(vsum,arr);
             for(j=0;j<4;j++) sum += arr[j];
             for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
             }
             vold = sum;
             LONG_PUTR(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETR(vnew);
               vold = vold + vnew;
             }

             LONG_PUTC(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETC(vnew);
               vold = vold + vnew;
             }
             simd_store(vold,arr);
             sum = arr[0];
             if(c == 0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
             }
             else if(sum < local_oldsum[i]) 
             {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
             }
           }
          }
        }
        if(c < cluster_count){
           //Center DMA
           int left_cluster_count = cluster_count - c;
           dma_set_size(&dma_get_center,left_cluster_count*local_dims*sizeof(float));
           dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
           dma_wait(&replyget_center, 1); replyget_center = 0;
           for(i=0; i< left_data_num; i++) {      
            p_data = local_data + i *local_dims;
            for(m = 0;m < left_cluster_count;m++){
              sum = 0;
              vsum = 0;
              c_data = local_center +  m*local_dims;
              for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
              }
              simd_store(vsum,arr);
              for(j=0;j<4;j++) sum += arr[j];
              for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
              }
              vold = sum;
              LONG_PUTR(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETR(vnew);
                vold = vold + vnew;
              }

              LONG_PUTC(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETC(vnew);
                vold = vold + vnew;
              }
              simd_store(vold,arr);
              sum = arr[0];
              if(c == 0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
              }
              else if(sum < local_oldsum[i]) 
              {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
              }
            }
          }
        }

        for(i=0; i< left_data_num; i++) {   
          if(local_data_index[i] >= start_cluster_count && local_data_index[i] < local_center_count){
            index = local_data_index[i] - start_cluster_count;
            local_center_num[index]++;
          }
          p_data = local_data + i*local_dims;
          dma(dma_get_center_temp, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyget_center_temp, 1); replyget_center_temp = 0;
          for(k=0;k+SIMDSIZE-1 < local_dims;k+=SIMDSIZE)
          {
              simd_load(vsrc,local_center_temp+k);
              simd_loadu(vdst,p_data+k);
              vsrc = vsrc + vdst;
              simd_store(vsrc,local_center_temp+k);
          }
          for (; k < local_dims; k++){
              local_center_temp[k] += p_data[k];
          }
          mb();
          dma(dma_put_center, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyput_center, 1); replyput_center = 0;
       }
#ifdef DEBUG
      compute_time += rtc() - run_time;
#endif
  }//每个中心聚点，每一维坐标加和完成

#ifdef DEBUG
  unsigned long m2 = rtc() -run_time;
  run_time = rtc();
#endif
  dma(dma_put_center_num, (long)(cluster_center_num + start_cluster_count), (long)(local_center_num));
  dma_wait(&replyput_center_num, 1); replyput_center_num = 0;
  
  ldm_free(local_data_index, max_data_num*sizeof(int));  
  ldm_free(local_oldsum, max_data_num*sizeof(float));  
  ldm_free(local_center_num, local_cluster_count*sizeof(int));  
  ldm_free(local_center, max_data_num*local_dims*sizeof(float));
  ldm_free(local_center_temp, local_dims*sizeof(float));
  ldm_free(local_data,max_data_num*local_dims*sizeof(float));
}
void kmeans_both_large_but_dims_smaller(KmeansPara *para) {
  const int max_len = 59*1024;
  int i,j,k,c,off=0;
  int id = athread_get_id(-1);

  //if(para->rank == 0 && id == 63) printf("kmeans_both_large_but_dims_smaller start id = %d \n", id);
  int count = para->data_size;
  int cluster_count = para->cluster_count;
  int dims = para->dims;
  volatile int replyget_data=0,replyput_center=0,replyget_center=0,replyput_center_num=0,replyget_center_temp=0;
  int local_cluster_count = cluster_count/SPNUM + (id<(cluster_count%SPNUM));
  dma_desc dma_get_data,dma_put_center,dma_get_center,dma_put_center_num,dma_get_center_temp;
  int start_cluster_count = id*(cluster_count/SPNUM) + (id<(cluster_count%SPNUM)?id:(cluster_count%SPNUM));
  int local_center_count = start_cluster_count + local_cluster_count;
  int batch_cluster_size = local_cluster_count;
  int left_space = max_len - local_cluster_count*sizeof(int) - (batch_cluster_size+1) *dims*sizeof(float) ;
  int min_size = (dims +1)*sizeof(float)+sizeof(int);
  while(left_space < min_size && batch_cluster_size > 1){
    batch_cluster_size = batch_cluster_size >> 1;
    left_space = max_len - local_cluster_count*sizeof(int) - (batch_cluster_size+1) *dims*sizeof(float) ;
  }
  if(batch_cluster_size < 1) batch_cluster_size = 1;
  assert(left_space >= min_size);
  int max_data_num = (max_len - local_cluster_count*sizeof(int) - (batch_cluster_size+1) *dims*sizeof(float)) / min_size;
  assert(max_data_num >0);
  int* cluster_center_num = (int*)para->cluster_center_num;
  float* data_ptr = (float*)(para->data);
  float* cluster_center = (float*)(para->cluster_center);
  float* cluster_center_out = (float*)(para->cluster_center_out);
  float *local_center = (float*)ldm_malloc(batch_cluster_size*dims*sizeof(float));
  float *local_data   = (float*)ldm_malloc(max_data_num*dims*sizeof(float));
  float *local_center_temp = (float*)ldm_malloc(dims*sizeof(float));
  float *local_oldsum = (float*)ldm_malloc(max_data_num*sizeof(float));
  int *local_data_index = (int*)ldm_malloc(max_data_num*sizeof(int));
  int *local_center_num = (int*)ldm_malloc(local_cluster_count*sizeof(int));
  for(i=0;i<local_cluster_count;i++)
    local_center_num[i] = 0;

  // DMA settings
  dma_set_op(&dma_get_center, DMA_GET);
  dma_set_mode(&dma_get_center, PE_MODE);
  dma_set_reply(&dma_get_center, &replyget_center);
  dma_set_size(&dma_get_center,batch_cluster_size*dims*sizeof(float));

  dma_set_op(&dma_get_data, DMA_GET);
  dma_set_mode(&dma_get_data, PE_MODE);
  dma_set_reply(&dma_get_data, &replyget_data);
  dma_set_size(&dma_get_data,max_data_num*dims*sizeof(float));  

  dma_set_op(&dma_get_center_temp, DMA_GET);
  dma_set_mode(&dma_get_center_temp, PE_MODE);
  dma_set_reply(&dma_get_center_temp, &replyget_center_temp); 
  dma_set_size(&dma_get_center_temp,dims*sizeof(float));
  dma_set_op(&dma_put_center, DMA_PUT);
  dma_set_mode(&dma_put_center, PE_MODE);
  dma_set_reply(&dma_put_center, &replyput_center); 
  dma_set_size(&dma_put_center,dims*sizeof(float));

  dma_set_op(&dma_put_center_num, DMA_PUT);
  dma_set_mode(&dma_put_center_num, PE_MODE);
  dma_set_reply(&dma_put_center_num, &replyput_center_num); 
  dma_set_size(&dma_put_center_num,local_cluster_count*sizeof(int));
  float temp = 0,sum = 0,oldsum = 0,arr[4];
  float *p_data,*c_data;
  floatv4 vold,vnew,vsrc,vdst,vsum;
  int m=0,min_index=0;
  long offset = 0;
  for(off = 0; off+max_data_num-1 < count; off+=max_data_num)
  {
      // DMA get a block
      offset = off;
      offset *= dims;
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;

      for(c = start_cluster_count; c+batch_cluster_size-1 < local_center_count;c+=batch_cluster_size)
      {
        //Center DMA
        dma_set_size(&dma_get_center,batch_cluster_size*dims*sizeof(float));
        dma(dma_get_center, (long)(cluster_center + c*dims), (long)(local_center));
        dma_wait(&replyget_center, 1); replyget_center = 0;
        for(i=0; i< max_data_num; i++) {
          p_data = local_data + i *dims;
          for(m = 0;m < batch_cluster_size;m++){
             sum = 0;
             vsum = 0;
             c_data = local_center + m*dims;
             for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
             }
             simd_store(vsum,arr);
             for(j=0;j<4;j++) sum += arr[j];
             for (; k < dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
             }
             if(c==start_cluster_count && m == 0){
                local_data_index[i] = start_cluster_count;
                local_oldsum[i] = sum;
             }
             else if(sum < local_oldsum[i])
             {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
             }
           }
          }
        }
        if(c < local_center_count){
           //Center DMA
           int left_cluster_count = local_center_count - c;
           dma_set_size(&dma_get_center,left_cluster_count*dims*sizeof(float));
           dma(dma_get_center, (long)(cluster_center + c*dims), (long)(local_center));
           dma_wait(&replyget_center, 1); replyget_center = 0;
           for(i=0; i< max_data_num; i++) {      
            p_data = local_data + i *dims;
            for(m = 0;m < left_cluster_count;m++){
              sum = 0;
              vsum = 0;
              c_data = local_center +  m*dims;
              for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
                simd_loadu(vsrc,p_data+k);
                simd_loadu(vdst,c_data+k);
                vdst = vsrc - vdst;
                vsum += vdst * vdst;
              }
              simd_store(vsum,arr);
              for(j=0;j<4;j++) sum += arr[j];
              for (; k < dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
              }
              if(c == start_cluster_count && m == 0){
                local_data_index[i] = start_cluster_count;
                local_oldsum[i] = sum;
              }
              else if(sum < local_oldsum[i]) 
              {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
              }
            }
          }
        }

        for(i=0; i< max_data_num; i++) {      
          oldsum = local_oldsum[i];
          min_index = local_data_index[i];
          arr[0] = min_index;
          arr[1] = oldsum;
          arr[2] = oldsum;
          arr[3] = oldsum;

          simd_load(vold,arr);
          LONG_PUTR(vold,8);
          for(j=0;j<RECV_NUM;j++)
          {
            LONG_GETR(vnew);
            simd_store(vnew,arr);
            if(arr[1] < oldsum)
            {
               min_index = arr[0];
               oldsum = arr[1];
            }
          }

          arr[0] = min_index;
          arr[1] = oldsum;
          arr[2] = oldsum;
          arr[3] = oldsum;

          simd_load(vold,arr);
          LONG_PUTC(vold,8);
          for(j=0;j<RECV_NUM;j++)
          {
            LONG_GETC(vnew);
            simd_store(vnew,arr);
            if(arr[1] < oldsum)
            {
              min_index = arr[0];
              oldsum = arr[1];
            } 
          }
          mb();
          if(min_index >= start_cluster_count && min_index < local_center_count){
            dma(dma_get_center_temp, (long)(cluster_center_out + min_index*dims), (long)(local_center_temp));
            local_center_num[min_index - start_cluster_count]++;
            p_data = local_data + i*dims;
            dma_wait(&replyget_center_temp, 1); replyget_center_temp = 0;
            for(k=0;k+SIMDSIZE-1 < dims;k+=SIMDSIZE)
            {
              simd_load(vsrc,local_center_temp+k);
              simd_loadu(vdst,p_data+k);
              vsrc = vsrc + vdst;
              simd_store(vsrc,local_center_temp+k);
            }
            for (; k < dims; k++){
              local_center_temp[k] += p_data[k];
            }
            mb();
            dma(dma_put_center, (long)(cluster_center_out + min_index*dims), (long)(local_center_temp));
            dma_wait(&replyput_center, 1); replyput_center = 0;
          }
       }
    } 
    if(off < count) {
      int left_data_num = count - off; 
      offset = off;
      offset *= dims;
      dma_set_size(&dma_get_data,left_data_num * dims * sizeof(float));
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
      for(c = start_cluster_count; c+batch_cluster_size-1 < local_center_count;c+=batch_cluster_size)
      {
        //Center DMA
        dma_set_size(&dma_get_center,batch_cluster_size*dims*sizeof(float));
        dma(dma_get_center, (long)(cluster_center + c*dims), (long)(local_center));
        dma_wait(&replyget_center, 1); replyget_center = 0;
        for(i=0; i< left_data_num; i++) {      
          p_data = local_data + i *dims;
          for(m = 0;m < batch_cluster_size;m++){
             sum = 0;
             vsum = 0;
             c_data = local_center + m*dims;
             for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
                simd_loadu(vsrc,p_data+k);
                simd_loadu(vdst,c_data+k);
                vdst = vsrc - vdst;
                vsum += vdst * vdst;
             }
             simd_store(vsum,arr);
             for(j=0;j<4;j++) sum += arr[j];
             for (; k < dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
             }
             if(c == start_cluster_count && m == 0){
                local_data_index[i] = start_cluster_count;
                local_oldsum[i] = sum;
             }
             else if(sum < local_oldsum[i]) 
             {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
             }
           }
          }
        }
        if(c < local_center_count){
           //Center DMA
           int left_cluster_count = local_center_count - c;
           dma_set_size(&dma_get_center,left_cluster_count*dims*sizeof(float));
           dma(dma_get_center, (long)(cluster_center + c*dims), (long)(local_center));
           dma_wait(&replyget_center, 1); replyget_center = 0;
           for(i=0; i< left_data_num; i++) {
            p_data = local_data + i *dims;
            for(m = 0;m < left_cluster_count;m++){
              sum = 0;
              vsum = 0;
              c_data = local_center + m*dims;
              for(k=0;k+SIMDSIZE-1 < dims; k += SIMDSIZE){
                simd_loadu(vsrc,p_data+k);
                simd_loadu(vdst,c_data+k);
                vdst = vsrc - vdst;
                vsum += vdst * vdst;
              }
              simd_store(vsum,arr);
              for(j=0;j<4;j++) sum += arr[j];
              for (; k < dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
              }
              if(c == start_cluster_count && m == 0){
                local_data_index[i] = start_cluster_count;
                local_oldsum[i] = sum;
              }
              else if(sum < local_oldsum[i]) 
              {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
              }
            }
          }
        }

        for(i=0; i< left_data_num; i++) {   
          oldsum = local_oldsum[i];
          min_index = local_data_index[i];
          arr[0] = min_index;
          arr[1] = oldsum;
          arr[2] = oldsum;
          arr[3] = oldsum;

          simd_load(vold,arr);
          LONG_PUTR(vold,8);
          for(j=0;j<RECV_NUM;j++)
          {
            LONG_GETR(vnew);
            simd_store(vnew,arr);
            if(arr[1] < oldsum)
            {
               min_index = arr[0];
               oldsum = arr[1];
            } 
          }	
    
          arr[0] = min_index;
          arr[1] = oldsum;
          arr[2] = oldsum;
          arr[3] = oldsum;

          simd_load(vold,arr);
          LONG_PUTC(vold,8);
          for(j=0;j<RECV_NUM;j++)
          {
            LONG_GETC(vnew);
            simd_store(vnew,arr);
            if(arr[1] < oldsum)
            {
              min_index = arr[0];
              oldsum = arr[1];
            } 
          }
          mb();
          if(min_index >= start_cluster_count && min_index < local_center_count){
            dma(dma_get_center_temp, (long)(cluster_center_out + min_index*dims), (long)(local_center_temp));
            local_center_num[min_index - start_cluster_count]++;
            p_data = local_data + i*dims;
            dma_wait(&replyget_center_temp, 1); replyget_center_temp = 0;
            for(k=0;k+SIMDSIZE-1 < dims;k+=SIMDSIZE)
            {
               simd_load(vsrc,local_center_temp+k);
               simd_loadu(vdst,p_data+k);
               vsrc = vsrc + vdst;
               simd_store(vsrc,local_center_temp+k);
            }
            for (; k < dims; k++){
               local_center_temp[k] += p_data[k];
            }
            mb();
            dma(dma_put_center, (long)(cluster_center_out + min_index*dims), (long)(local_center_temp));
            dma_wait(&replyput_center, 1); replyput_center = 0;
          }
       }
  }//每个中心聚点，每一维坐标加和完成

  dma(dma_put_center_num, (long)(cluster_center_num + start_cluster_count), (long)(local_center_num));
  dma_wait(&replyput_center_num, 1); replyput_center_num = 0;
  //for(i = 0; i < 64; ++i)
  //{
    /*if(i == id)
    {
      printf("id=%d", id);
    }*/
  if(para->rank == 0 && id == 0)
  {
    int m;
    for(m = 0; m < local_cluster_count; ++m)
    {
      printf("slave %d %d\n", m, local_center_num[m]);
    }
  }
  //}
  ldm_free(local_data_index, max_data_num*sizeof(int));  
  ldm_free(local_oldsum, max_data_num*sizeof(float));  
  ldm_free(local_center_num, local_cluster_count*sizeof(int));  
  ldm_free(local_center, batch_cluster_size*dims*sizeof(float));
  ldm_free(local_center_temp, dims*sizeof(float));
  ldm_free(local_data,max_data_num*dims*sizeof(float));
}
void kmeans_dims_large_exception(KmeansPara *para) {
  const int max_len = 59*1024; 
  int i,j,k,c,off=0;
  int id = athread_get_id(-1);
  int count = para->data_size;
  int cluster_count = para->cluster_count;
  int dims = para->dims;
  volatile int replyget_data=0,replyput_center=0,replyget_center=0,replyput_center_num=0,replyget_center_temp=0;
  dma_desc dma_get_data,dma_put_center,dma_get_center,dma_put_center_num,dma_get_center_temp;
  int local_cluster_count = cluster_count;
  int start_cluster_count = 0;
  int local_center_count = start_cluster_count + local_cluster_count;
  int local_dims = dims/SPNUM + (id<(dims%SPNUM));
  int start_dims = id*(dims/SPNUM) + (id<(dims%SPNUM)?id:(dims%SPNUM));
  int max_data_num = (max_len - local_cluster_count*sizeof(int) - local_dims*sizeof(float)) / ((2*local_dims+1)*sizeof(float) + sizeof(int));
  int left_space = (max_len - local_cluster_count*sizeof(int) - local_dims*sizeof(float));
  assert(max_data_num > 0);
  int* cluster_center_num = (int*)para->cluster_center_num;
  float* data_ptr = (float*)para->data;
  float* cluster_center = (float*)para->cluster_center;
  float* cluster_center_out = (float*)para->cluster_center_out;
  float *local_center = (float*)ldm_malloc(max_data_num*local_dims*sizeof(float));
  float *local_data   = (float*)ldm_malloc(max_data_num*local_dims*sizeof(float));
  float *local_center_temp = (float*)ldm_malloc(local_dims*sizeof(float));
  float *local_oldsum = (float*)ldm_malloc(max_data_num*sizeof(float));
  int *local_data_index = (int*)ldm_malloc(max_data_num*sizeof(int));
  int *local_center_num = (int*)ldm_malloc(local_cluster_count*sizeof(int));
  for(i=0;i<local_cluster_count;i++)
    local_center_num[i] = 0;
  // DMA settings
  dma_set_op(&dma_get_center, DMA_GET);
  dma_set_mode(&dma_get_center, PE_MODE);
  dma_set_reply(&dma_get_center, &replyget_center);
  dma_set_size(&dma_get_center,max_data_num*local_dims*sizeof(float));
  dma_set_bsize(&dma_get_center,local_dims*sizeof(float));
  dma_set_stepsize(&dma_get_center,(dims - local_dims)*sizeof(float));

  dma_set_op(&dma_get_data, DMA_GET);
  dma_set_mode(&dma_get_data, PE_MODE);
  dma_set_reply(&dma_get_data, &replyget_data);
  dma_set_size(&dma_get_data,max_data_num*local_dims*sizeof(float));  
  dma_set_bsize(&dma_get_data,local_dims*sizeof(float));
  dma_set_stepsize(&dma_get_data,(dims - local_dims)*sizeof(float));

  dma_set_op(&dma_get_center_temp, DMA_GET);
  dma_set_mode(&dma_get_center_temp, PE_MODE);
  dma_set_reply(&dma_get_center_temp, &replyget_center_temp); 
  dma_set_size(&dma_get_center_temp,local_dims*sizeof(float));
  dma_set_op(&dma_put_center, DMA_PUT);
  dma_set_mode(&dma_put_center, PE_MODE);
  dma_set_reply(&dma_put_center, &replyput_center); 
  dma_set_size(&dma_put_center,local_dims*sizeof(float));

  dma_set_op(&dma_put_center_num, DMA_PUT);
  dma_set_mode(&dma_put_center_num, PE_MODE);
  dma_set_reply(&dma_put_center_num, &replyput_center_num); 
  dma_set_size(&dma_put_center_num,local_cluster_count*sizeof(int));
  float temp = 0,sum = 0,arr[4],*p_data,*c_data;
  floatv4 vold,vnew,vsrc,vdst,vsum;
  int index = 0,m=0;
  long offset = 0;
#ifdef DEBUG
  unsigned long compute_time=0,dma_time=0,run_time=0;
#endif

  for(off = 0; off+max_data_num-1 < count; off+=max_data_num)
  {
#ifdef DEBUG
      run_time = rtc();
#endif
      // DMA get a block
      offset = off;
      offset *= dims;
      offset += start_dims;
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;

      /*if(id == 0)
      {
        int m;
        printf("max_data_num = %d\n", max_data_num);
        for(m = 0; m < 10; ++m)
        {
          printf("%f ", *(data_ptr + offset + m));
        }
      }*/
#ifdef DEBUG
      dma_time += rtc() - run_time;
      run_time = rtc();
#endif

      for(c = 0; c+max_data_num-1 < cluster_count;c+=max_data_num)
      {
        //Center DMA
        dma_set_size(&dma_get_center,max_data_num*local_dims*sizeof(float));
        dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
        dma_wait(&replyget_center, 1); replyget_center = 0;
        for(i=0; i< max_data_num; i++) {
          p_data = local_data + i *local_dims;
          for(m = 0;m < max_data_num;m++){
             sum = 0;
             vsum = 0;
             c_data = local_center + m*local_dims;
             for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
             }
             simd_store(vsum,arr);
             for(j=0;j<4;j++) sum += arr[j];
             for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
             }
             vold = sum;
             LONG_PUTR(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETR(vnew);
               vold = vold + vnew;
             }

             LONG_PUTC(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETC(vnew);
               vold = vold + vnew;
             }
             simd_store(vold,arr);
             sum = arr[0];
             if(c==0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
             }
             else if(sum < local_oldsum[i])
             {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
             }
           }
          }
        }
        if(c < cluster_count){
           //Center DMA
           int left_cluster_count = cluster_count - c;
           dma_set_size(&dma_get_center,left_cluster_count*local_dims*sizeof(float));
           dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
           dma_wait(&replyget_center, 1); replyget_center = 0;
           for(i=0; i< max_data_num; i++) {      
            p_data = local_data + i *local_dims;
            for(m = 0;m < left_cluster_count;m++){
              sum = 0;
              vsum = 0;
              c_data = local_center +  m*local_dims;
              for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
              }
              simd_store(vsum,arr);
              for(j=0;j<4;j++) sum += arr[j];
              for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
              }
              vold = sum;
              LONG_PUTR(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETR(vnew);
                vold = vold + vnew;
              }	
    
              LONG_PUTC(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETC(vnew);
                vold = vold + vnew;
              }	
              simd_store(vold,arr);
              sum = arr[0];
              if(c == 0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
              }
              else if(sum < local_oldsum[i]) 
              {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
              }
            }
          }
        }

        for(i=0; i< max_data_num; i++) {      
          if(id < 1){
            local_center_num[local_data_index[i]]++;
          }
          p_data = local_data + i*local_dims;
          dma(dma_get_center_temp, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyget_center_temp, 1); replyget_center_temp = 0;
          for(k=0;k+SIMDSIZE-1 < local_dims;k+=SIMDSIZE)
          {
              simd_load(vsrc,local_center_temp+k);
              simd_loadu(vdst,p_data+k);
              vsrc = vsrc + vdst;
              simd_store(vsrc,local_center_temp+k);
          }
          for (; k < local_dims; k++){
              local_center_temp[k] += p_data[k];
          }
          mb();
          dma(dma_put_center, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyput_center, 1); replyput_center = 0;
       }
#ifdef DEBUG
      compute_time += rtc() - run_time;
#endif
    } 
    if(off < count) {
#ifdef DEBUG
      run_time = rtc();
#endif
      int left_data_num = count - off; 
      offset = off;
      offset *= dims;
      offset += start_dims;
      dma_set_size(&dma_get_data,left_data_num * local_dims * sizeof(float));
      dma(dma_get_data, (long)(data_ptr + offset), (long)(local_data));
      dma_wait(&replyget_data, 1); replyget_data = 0;
#ifdef DEBUG   
      dma_time += rtc() - run_time;
      run_time = rtc();
#endif
      for(c = 0; c+max_data_num-1 < cluster_count;c+=max_data_num)
      {
        //Center DMA
        dma_set_size(&dma_get_center,max_data_num*local_dims*sizeof(float));
        dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
        dma_wait(&replyget_center, 1); replyget_center = 0;
        for(i=0; i< left_data_num; i++) {      
          p_data = local_data + i *local_dims;
          for(m = 0;m < max_data_num;m++){
             sum = 0;
             vsum = 0;
             c_data = local_center + m*local_dims;
             for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
             }
             simd_store(vsum,arr);
             for(j=0;j<4;j++) sum += arr[j];
             for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
             }
             vold = sum;
             LONG_PUTR(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETR(vnew);
               vold = vold + vnew;
             }	
    
             LONG_PUTC(vold,8);
             for(j=0;j<RECV_NUM;j++)
             {
               LONG_GETC(vnew);
               vold = vold + vnew;
             }	
             simd_store(vold,arr);
             sum = arr[0];
             if(c == 0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
             }
             else if(sum < local_oldsum[i]) 
             {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
             }
           }
          }
        }
        if(c < cluster_count){
           //Center DMA
           int left_cluster_count = cluster_count - c;
           dma_set_size(&dma_get_center,left_cluster_count*local_dims*sizeof(float));
           dma(dma_get_center, (long)(cluster_center + c*dims + start_dims), (long)(local_center));
           dma_wait(&replyget_center, 1); replyget_center = 0;
           for(i=0; i< left_data_num; i++) {      
            p_data = local_data + i *local_dims;
            for(m = 0;m < left_cluster_count;m++){
              sum = 0;
              vsum = 0;
              c_data = local_center +  m*local_dims;
              for(k=0;k+SIMDSIZE-1 < local_dims; k += SIMDSIZE){
               simd_loadu(vsrc,p_data+k);
               simd_loadu(vdst,c_data+k);
               vdst = vsrc - vdst;
               vsum += vdst * vdst;
              }
              simd_store(vsum,arr);
              for(j=0;j<4;j++) sum += arr[j];
              for (; k < local_dims; k++){
                temp = p_data[k] - c_data[k];
                sum += temp*temp;
              }
              vold = sum;
              LONG_PUTR(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETR(vnew);
                vold = vold + vnew;
              }	
    
              LONG_PUTC(vold,8);
              for(j=0;j<RECV_NUM;j++)
              {
                LONG_GETC(vnew);
                vold = vold + vnew;
              }	
              simd_store(vold,arr);
              sum = arr[0];
              if(c == 0 && m == 0){
                local_data_index[i] = 0;
                local_oldsum[i] = sum;
              }
              else if(sum < local_oldsum[i]) 
              {
                local_data_index[i] = c+m;
                local_oldsum[i] = sum;
              }
            }
          }
        }

        for(i=0; i< left_data_num; i++) {   
          if(id < 1){
            local_center_num[local_data_index[i]]++;
          }
          p_data = local_data + i*local_dims;
          dma(dma_get_center_temp, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyget_center_temp, 1); replyget_center_temp = 0;
          for(k=0;k+SIMDSIZE-1 < local_dims;k+=SIMDSIZE)
          {
              simd_load(vsrc,local_center_temp+k);
              simd_loadu(vdst,p_data+k);
              vsrc = vsrc + vdst;
              simd_store(vsrc,local_center_temp+k);
          }
          for (; k < local_dims; k++){
              local_center_temp[k] += p_data[k];
          }
          mb();
          dma(dma_put_center, (long)(cluster_center_out + local_data_index[i]*dims + start_dims), (long)(local_center_temp));
          dma_wait(&replyput_center, 1); replyput_center = 0;
       }
#ifdef DEBUG
      compute_time += rtc() - run_time;
#endif
  }//每个中心聚点，每一维坐标加和完成

#ifdef DEBUG
  unsigned long m2 = rtc() -run_time;
  run_time = rtc();
#endif
  if(id < 1){
    dma(dma_put_center_num, (long)(cluster_center_num + start_cluster_count), (long)(local_center_num));
    dma_wait(&replyput_center_num, 1); replyput_center_num = 0;
  }
  ldm_free(local_data_index, max_data_num*sizeof(int));  
  ldm_free(local_oldsum, max_data_num*sizeof(float));  
  ldm_free(local_center_num, local_cluster_count*sizeof(int));  
  ldm_free(local_center, max_data_num*local_dims*sizeof(float));
  ldm_free(local_center_temp, local_dims*sizeof(float));
  ldm_free(local_data,max_data_num*local_dims*sizeof(float));
}
//原则，必须保证64个CPE一次完整处理一个维度（无论是单个CPE，还是64个CPE一起），最大能处理的维度大小是128K
void sw_slave_kmeans_f(KmeansPara *para) {
  const int max_len = 59*1024;
  const int max_dims = 4097;
  //const int max_dims = 2048;
  int id = athread_get_id(-1);
  int count = para->data_size;
  int cluster_count = para->cluster_count;
  int dims = para->dims;
  int local_cluster_count = cluster_count;
  int start_cluster_count = 0,is_splited_cluster_count = 0,is_splited_dims = 0;
  long total_size = align(2*cluster_count*dims, SIMDSIZE) * sizeof(float) + align(cluster_count, SIMDSIZE_INT) * sizeof(int);
  //long total_size = (2*cluster_count*dims + (SIMDSIZE - (2*cluster_count*dims)%SIMDSIZE) % SIMDSIZE)*sizeof(float) + (cluster_count + (SIMDSIZE_INT - cluster_count%SIMDSIZE_INT)%SIMDSIZE_INT)*sizeof(int);
  long left_space = max_len - total_size;
  //if(id == 0) printf("left_space=%ld  left_space<1? %d\n", left_space, (left_space < 1));
  left_space = left_space / (long)(dims * sizeof(float) * 2);
  //if(id == 0) printf("left_space=%ld  left_space<1? %d\n", left_space, (left_space < 1));
  int start_dims = 0;
  int local_dims = dims;
  if(cluster_count < dims){
    if(left_space < 1 && (dims / SPNUM ) >0)
    {
       local_dims = dims/SPNUM + (0 < (dims%SPNUM));
       start_dims = id*(dims/SPNUM) + (id<(dims%SPNUM)?id:(dims%SPNUM));
       is_splited_dims = 1;
    }
    total_size = align(2*local_cluster_count*local_dims, SIMDSIZE) * sizeof(float) + align(local_cluster_count, SIMDSIZE_INT) * sizeof(int); //(2*local_cluster_count*local_dims + (2*local_cluster_count*local_dims)%SIMDSIZE)*sizeof(float) + (local_cluster_count + local_cluster_count%SIMDSIZE_INT)*sizeof(int);
    left_space = max_len - total_size;
    if(left_space < 1 && (cluster_count / SPNUM ) >0)
        is_splited_cluster_count = 1;

    if(id == 0) printf("is_splited_cluster_count=%d  is_splited_dims=%d  total_size=%ld\n", is_splited_cluster_count, is_splited_dims, total_size);
  }
  else
  {
    //if((cluster_count / SPNUM ) >0)
    if(left_space < 1 && (cluster_count / SPNUM ) >0)
    {
       local_cluster_count = cluster_count/SPNUM + (0<(cluster_count%SPNUM));
       //start_cluster_count = id*(cluster_count/SPNUM) + (id<(cluster_count%SPNUM)?id:(cluster_count%SPNUM));
       is_splited_cluster_count = 1;
    }
    total_size = (2*local_cluster_count*local_dims + (2*local_cluster_count*local_dims)%SIMDSIZE)*sizeof(float) \
               + (local_cluster_count + local_cluster_count%SIMDSIZE_INT)*sizeof(int);
    //if(id == 0) printf("total_size=%d\n", total_size);
    left_space = max_len - total_size;
    left_space = left_space / (long)(dims * sizeof(float));

    if(left_space < 1 && (dims / SPNUM ) >0)
        is_splited_dims = 1;
    if(id == 0) printf("is_splited_cluster_count=%d  is_splited_dims=%d\n", is_splited_cluster_count, is_splited_dims);
  }
  if(is_splited_cluster_count <1 && is_splited_dims < 1)
  {
    /*if(id < 1)
    {
      printf("kmeans_normal\n");
      printf("n = %d, k = %d, d = %d", para->data_size, para->dims, para->cluster_count);
      int tmp = 0;
      for(tmp = 0; tmp < para->data_size * para->dims; ++tmp)
      {
        printf("%f ", para->data[tmp]);
        if(tmp % para->dims == 2)
        {
          printf("\n");
        }
      }
    }*/
    //printf("kmeans_normal\n");
    kmeans_normal(para);
  }
  else if(is_splited_cluster_count >0 && is_splited_dims < 1)
  {
     //if(id < 1) printf("kmeans_cluster_count_large\n");
     kmeans_cluster_count_large(para);
  }
  else if(is_splited_cluster_count < 1 && is_splited_dims >0)
  {
    int local_dims = dims/SPNUM + (0<(dims%SPNUM));
    int left_space = max_len - ((2*cluster_count)* local_dims * sizeof(float) + cluster_count*sizeof(int));
    int max_data_num = left_space / (int)(local_dims*sizeof(float));
    if(max_data_num > 0 && left_space > 0)
    {
       //if(id == 0) printf("kmeans_dims_large\n");
       kmeans_dims_large(para);
    }
    else
    {
      //if(id == 0) printf("kmeans_dims_large_exception\n");
      kmeans_dims_large_exception(para);
    }
  }
  else{
     if(dims < max_dims){
       //if(id == 0)  printf("kmeans_both_large_but_dims_smaller %d\n", id);
       kmeans_both_large_but_dims_smaller(para);
     }
     else{
       //if(id < 1)printf("kmeans_both_large\n");
       kmeans_both_large(para);
     }
  }
}

inline float get_euclidean_distance(float * data1, float * data2, int dims)
{
  float ans = 0, tmp;
  int i;

  for(i = 0; i < dims; ++i)
  {
    tmp = (*(data1 + i) - *(data2 + i));
    ans += tmp * tmp;
  }
  ans = sqrt(ans);
  return ans;
}

inline float get_squared_euclidean_distance(float * data1, float * data2, int dims)
{
  /*float ans = 0, tmp;
  int i;
  for(i = 0; i < dims; ++i)
  {
    tmp = (*(data1 + i) - *(data2 + i));
    ans += tmp * tmp;
  }*/
  float ans = 0, tmp;
  int i;
  floatv4 va, vb, vsum = 0;
  for(i = 0; i + SIMDSIZE - 1 < dims; i += SIMDSIZE)
  {
    simd_loadu(va, data1 + i);
    simd_loadu(vb, data2 + i);
    va = va - vb;
    vsum += (va * va);
  }
  for(; i < dims; i++)
  {
    tmp = (*(data1 + i) - *(data2 + i));
    ans += tmp * tmp;
  }
  float arr[SIMDSIZE];
  simd_store(vsum, arr);
  for(i = 0; i < SIMDSIZE; ++i)
  {
    ans += arr[i];
  }
  return ans;
}
void get_data_group(EvaluationPara * para)
{
  // firstly, no using ldm and simd
  int id = athread_get_id(-1);

  //float *data = para->data;
  int dims = para->dims;
  int data_size = para->data_size;
  int local_data_size = data_size / SPNUM + (id < (data_size % SPNUM));
  int start = id * (data_size / SPNUM) + (id < (data_size % SPNUM) ? id : (data_size % SPNUM));
  float * data = &(((float*)para->data)[start*dims]);

  int cluster_count = para->cluster_count;
  float * cluster_center = para->cluster_center;
  int * data_group = &(((int*)para->data_group)[start]);

  int i, j, min_index;
  float min, distance;

  /*if(id == 0)
  {
    int m;
    for(m = 0; m < dims; ++m)
    {
      printf("%f ", data[i]);
    }
    printf("\n");
  }*/
  for(i = 0; i < local_data_size; ++i)
  {
    min = FLT_MAX;
    for(j = 0; j < cluster_count; ++j)
    {
      distance = get_squared_euclidean_distance(data + i*dims, cluster_center + j*dims, dims);
      if(distance < min)
      {
        min = distance;
        min_index = j;
      }

    }
    data_group[i] = min_index;
  }
}

inline void register_communicate(float * data, int data_size)  // make sure data_size % SIMDSIZE == 0
{
  int i, j;
  //floatv4 va, vb;
  floatv4 va, vb;
  for(i = 0; i < data_size; i += SIMDSIZE)
  {
    simd_load(va, data + i);
    LONG_PUTR(va, 8);

    for(j = 0; j < RECV_NUM; ++j)
    {
      LONG_GETR(vb);
      va = va + vb;
    }
    LONG_PUTC(va, 8);
    for(j = 0; j < RECV_NUM; ++j)
    {
      LONG_GETC(vb);
      va = va + vb;
    }
    simd_store(va, data + i);
  }
}

void load_data(int fd, float * data, long len, long offset)
{
  int ret;
  lseek(fd, offset, SEEK_SET);
  ret = read(fd, data, len);

  if(ret < 0)
  {
    perror("read error\n");
  }
}

void caculate_radius(EvaluationPara * para)
{
  int max_len = 59 * 1024;
  int id = athread_get_id(-1);

  int rank = para->rank; // just for DEBUG
  int dims = para->dims;

  int data_size = para->data_size;
  int all_data_size = para->all_data_size;

  int local_data_size = all_data_size / SPNUM + (id < (all_data_size % SPNUM));

  int start = id * (all_data_size / SPNUM) + (id < (all_data_size % SPNUM) ? id : (all_data_size % SPNUM));
  int cluster_count = para->cluster_count;

  float * data = para->data;
  int data_start = para->data_start;
  int * data_group = para->all_data_group;
  float * local_all_data = &(((float*)para->all_data)[start*dims]);

  int i, j, cluster_num, tmp;
  float distance;

  int align_data_size = align(data_size, SIMDSIZE);
  float * radius = (float *)malloc(align_data_size * sizeof(float));
  for(i = 0; i < align_data_size; ++i)
  {
    radius[i] = 0.0;
  }

  for(i = 0; i < align_data_size; ++i)
  {
    cluster_num = data_group[data_start + i];
    for(j = 0; j < local_data_size; ++j)
    {
      tmp = data_group[start + j];
      if(tmp == cluster_num)
      {
        distance = get_euclidean_distance(data + i * dims, local_all_data + j * dims, dims);
        if(distance > radius[i])
        {
          radius[i] = distance;
        }
      }
    }
  }

  int k;
  floatv4 va, vb;
  float arr1[4], arr2[4];
  for(i = 0; i < align_data_size; i += SIMDSIZE)
  {
    simd_load(va, radius + i);
    simd_store(va, arr1);
    LONG_PUTR(va, 8);

    for(j = 0; j < RECV_NUM; ++j)
    {
      LONG_GETR(vb);
      simd_store(vb, arr2);

      for(k = 0; k < SIMDSIZE; ++k)
      {
        if(arr2[k] > arr1[k])
        {
          arr1[k] = arr2[k];
        }
      }
    }
    simd_load(va, arr1);
    LONG_PUTC(va, 8);

    for(j = 0; j < RECV_NUM; ++j)
    {
      LONG_GETC(vb);
      simd_store(vb, arr2);

      for(k = 0; k < SIMDSIZE; ++k)
      {
        if(arr2[k] > arr1[k])
        {
          arr1[k] = arr2[k];
        }
      }
    }
    simd_load(va, arr1);
    simd_store(va, radius + i);
  }

  for(i = 0; i < data_size; ++i)
  {
    para->radius[i] = radius[i];
  }
  free(radius);
}
void caculate_evaluation_function(EvaluationPara * para)
{
  int max_len = 59 * 1024;
  int id = athread_get_id(-1);

  int rank = para->rank;
  int dims = para->dims;
  int data_size = para->data_size;
  int all_data_size = para->all_data_size;
  int local_data_size = all_data_size / SPNUM + (id < (all_data_size % SPNUM));
  float * distance = (float *)malloc(local_data_size * sizeof(float));

  int start = id *(all_data_size / SPNUM) + (id < (all_data_size % SPNUM) ? id : (all_data_size % SPNUM)); // use in data;
  int cluster_count = para->cluster_count;
  int fd = open(para->out_filename, O_RDONLY, S_IRWXU|S_IRWXG|S_IRWXO);//= para->fd;
  if(fd < 0)
  {
    printf("open failed\n");
    return ;
  }
  /*if(id == 0)
  {
    if(rank == 1) printf("data_start=%d\n", para->data_start);
    printf("slave fd=%d \n", fd);
  }*/

  float * data = para->data;

  int data_start = para->data_start;
  int * data_group = para->all_data_group;

  float * local_all_data = &(((float*)para->all_data)[start*dims]);

  int align_cluster_count_size = align(cluster_count, SIMDSIZE);
  float * cluster_distance_count;
  if(max_len > align_cluster_count_size * sizeof(float))
  {
    //if(rank == 0 && id == 0) printf("cluster_distance_count\n");
    cluster_distance_count = (float *) ldm_malloc(align_cluster_count_size * sizeof(float));
  }
  else
    cluster_distance_count = (float *) malloc(align_cluster_count_size * sizeof(float));

  float * non_cluster_distance_count;
  if(max_len > 2 * align_cluster_count_size * sizeof(float))
  {
    //if(rank == 0 && id == 0) printf("ldm non_cluster_distance_count  %d\n", max_len - 2 * align_cluster_count_size * sizeof(float));
    non_cluster_distance_count = (float *) ldm_malloc(align_cluster_count_size * sizeof(float));
  }
  else
  {
    //if(rank == 0 && id == 0)  printf("non_cluster_distance_count\n");
    non_cluster_distance_count = (float *) malloc(align_cluster_count_size * sizeof(float));
  }
  int i, j;
  for(i = 0; i < align_cluster_count_size; ++i)
  {
    cluster_distance_count[i] = 0.0;
    non_cluster_distance_count[i] = 0.0;
  }

  long offset;
  int cluster_num, tmp;
  for(i = 0; i < data_size; ++i) // data_size  27338
  {
    offset = (long)(data_start + i) * (long) all_data_size + (long)start;
    offset = (long)offset * (long)sizeof(float);
    load_data(fd, distance, local_data_size * sizeof(float), offset);
    cluster_num = data_group[data_start + i];
    /*if(id == 0 && rank == 0 && i == 0)
    {
      int m;
      printf("cluster_num=%d\n", cluster_num);
      for(m = 0; m < local_data_size; ++m)
        printf("%d ", data_group[start + m]);
      printf("offset=%ld\n", offset);
      for(m = 0; m < local_data_size; ++m)
      {
        printf("%f ", distance[m]);
      }
      printf("\n");
    }*/
    for(j = 0; j < local_data_size; ++j)
    {
      tmp = data_group[start + j];
      if(tmp == cluster_num)
      {
        cluster_distance_count[cluster_num] += distance[j];
        /*if(id == 0 && rank == 0 && i == 0)
        {
          printf("j=%d, %f\n", j, distance[j]);
        }*/
      }
      else
      {
        non_cluster_distance_count[cluster_num] += distance[j];
      }
    }

    /*if(id == 0 && rank == 0)
    {
      printf("%d\n", i);
      int m;
      for(m = 0; m < cluster_count; ++m)
      {
        printf("%f ", cluster_distance_count[m]);
      }
      printf("\n");
    }*/
  }
  register_communicate(cluster_distance_count, align_cluster_count_size);
  register_communicate(non_cluster_distance_count, align_cluster_count_size);
  if(id == 0)
  {
    for(i = 0; i < cluster_count; i++)
    {
      para->cluster_distance_count[i] = cluster_distance_count[i];
      para->non_cluster_distance_count[i] = non_cluster_distance_count[i];
      //printf("slave cluster_distance_count= %f\n", cluster_distance_count[i]);
    }
    /*if(rank == 0)
    {
      int m;
      printf("slave end ");
      for(m = 0; m < cluster_count; ++m)
        printf("%f ", cluster_distance_count[m]);
      printf("\n");
    }*/
    /*dma_desc put_distance1, put_distance2;
    volatile int reply_put_distance1 = 0, reply_put_distance2 = 0;

    dma_set_op(&put_distance1, DMA_PUT);
    dma_set_mode(&put_distance1, PE_MODE);
    dma_set_reply(&put_distance1, &reply_put_distance1);
    dma_set_size(&put_distance1, cluster_count * sizeof(float));

    dma_set_op(&put_distance2, DMA_PUT);
    dma_set_mode(&put_distance2, PE_MODE);
    dma_set_reply(&put_distance2, &reply_put_distance2);
    dma_set_size(&put_distance2, cluster_count * sizeof(float));

    printf("use dma store cluster_distance_count\n");
    mb();
    dma(put_distance1, (long)(para->cluster_distance_count), (long)cluster_distance_count);
    dma_wait(&reply_put_distance1, 1); reply_put_distance1 = 0;

    dma(put_distance2, (long)(para->non_cluster_distance_count), (long)non_cluster_distance_count);
    dma_wait(&reply_put_distance2, 1); reply_put_distance2 = 0;
    mb();*/
  }

  free(distance);
  if(max_len > align_cluster_count_size * sizeof(float))
    ldm_free(cluster_distance_count, align_cluster_count_size);
  else
    free(cluster_distance_count);

  if(max_len > 2 * align_cluster_count_size * sizeof(float))
    ldm_free(non_cluster_distance_count, align_cluster_count_size);
  else
    free(non_cluster_distance_count);
  close(fd);
}

void caculate_evaluation_function_old(EvaluationPara * para)
{
  int max_len = 59*1024;
  int id = athread_get_id(-1);
  int dims = para->dims;
  int data_size = para->data_size;
  int all_data_size = para->all_data_size;
  int local_data_size = all_data_size / SPNUM + (id < (all_data_size % SPNUM));

  int start = id *(all_data_size / SPNUM) + (id < (all_data_size % SPNUM) ? id : (all_data_size % SPNUM)); // use in data;
  int cluster_count = para->cluster_count;

  float * cluster_center = para->cluster_center;
  float * data = para->data;

  int data_start = para->data_start; // the start index of data in all_data;
  int * data_group = para->data_group;
  float * local_all_data = &(((float*)para->all_data)[start*dims]);

  int align_cluster_count_size = align(cluster_count, SIMDSIZE);
  float * cluster_distance_count;
  if(max_len - align_cluster_count_size * sizeof(float) > 0)
    cluster_distance_count = (float *) ldm_malloc(align_cluster_count_size * sizeof(float));
  else
    cluster_distance_count = (float *) malloc(align_cluster_count_size * sizeof(float));

  float * non_cluster_distance_count;
  if(max_len - 2 * align_cluster_count_size * sizeof(float) > 0)
    non_cluster_distance_count = (float *) ldm_malloc(align_cluster_count_size * sizeof(float));
  else
    non_cluster_distance_count = (float *) malloc(align_cluster_count_size * sizeof(float));

  int i, j;
  for(i = 0; i < align_cluster_count_size; ++i)
  {
    cluster_distance_count[i] = 0.0;
    non_cluster_distance_count[i] = 0.0;
  }

  int cluster_num, tmp;
  float distance;

  if(id == 0)
  {
    printf("data_size=%d, local_data_size=%d\n", data_size, local_data_size);
  }
  unsigned long run_time;
  for(i = 0; i < data_size; ++i)
  {
    cluster_num = data_group[data_start + i];
    for(j = 0; j < local_data_size; ++j)
    {
      if((data_start + i) >= (start + j)) continue;
      //run_time = rtc();
      distance = get_squared_euclidean_distance(data + i * dims, local_all_data + j * dims, dims); // get_euclidean_distance
      run_time = rtc() - run_time;
      //if(id == 1)
      //{
      //  printf("%d  runtime=%lu \n", id, run_time);
      //}
      tmp = data_group[start + j];
      if(tmp == cluster_num)  // it can be optimized
      {
        cluster_distance_count[cluster_num] += distance;
        cluster_distance_count[tmp] += distance;
      }
      else
      {
        non_cluster_distance_count[cluster_num] += distance;
        non_cluster_distance_count[tmp] += distance;
      }
    }
    if(id == 60)
    {
      printf("%d\n", i);
    }
  }

  //if(id == 0)
  //{
  printf("start register commulication: %d\n", id);
  //}
  register_communicate(cluster_distance_count, align_cluster_count_size);
  register_communicate(non_cluster_distance_count, align_cluster_count_size);

  dma_desc put_distance1, put_distance2;
  volatile int reply_put_distance1, reply_put_distance2;

  dma_set_op(&put_distance1, DMA_PUT);
  dma_set_mode(&put_distance1, PE_MODE);
  dma_set_reply(&put_distance1, &reply_put_distance1);
  dma_set_size(&put_distance1, cluster_count * sizeof(float));

  dma_set_op(&put_distance2, DMA_PUT);
  dma_set_mode(&put_distance2, PE_MODE);
  dma_set_reply(&put_distance2, &reply_put_distance2);
  dma_set_size(&put_distance2, cluster_count * sizeof(float));

  if(id == 0)
  {
    printf("use dma store cluster_distance_count\n");
    dma(put_distance1, (long)(para->cluster_distance_count), (long)cluster_distance_count);
    dma_wait(&reply_put_distance1, 1); reply_put_distance1 = 0;

    dma(put_distance2, (long)(para->non_cluster_distance_count), (long)non_cluster_distance_count);
    dma_wait(&reply_put_distance2, 1); reply_put_distance2 = 0;


  }
  if(max_len - align_cluster_count_size * sizeof(float) > 0)
    ldm_free(cluster_distance_count, align_cluster_count_size);
  else
    free(cluster_distance_count);

  if(max_len - 2 * align_cluster_count_size * sizeof(float) > 0)
    ldm_free(non_cluster_distance_count, align_cluster_count_size);
  else
    free(non_cluster_distance_count);
}
