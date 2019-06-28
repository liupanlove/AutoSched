#include <slave.h>
#include <simd.h>
#include <dma.h>
#include <math.h>

#define SIMDSIZE 4
#define SIMDSIZE_INT 8
#define SPNUM 64

typedef struct _GetDistancePara{
    int data_size;
    int all_data_size;
    int dims;

    float * distance;
    float * data;
    float * all_data;
} GetDistancePara;

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
  return sqrt(ans);
  //return ans;
}

void caculate_distance(GetDistancePara * get_distance_para)
{
    int id = athread_get_id(-1);
    int dims = get_distance_para->dims;
    int data_size = get_distance_para->data_size;
    int all_data_size = get_distance_para->all_data_size;

    int local_data_size = all_data_size / SPNUM + (id < (all_data_size % SPNUM));
    int start = id *(all_data_size / SPNUM) + (id < (all_data_size % SPNUM) ? id : (all_data_size % SPNUM)); // use in data;

    //int distance_start = start * data_size;
    float * data = get_distance_para->data;
    //float * all_data = get_distance_para->all_data;
    float * distance = get_distance_para->distance;
    float * local_distance = (float *) ldm_malloc(local_data_size * sizeof(float));
    ///assert(local_distance != NULL);
    if(local_distance == NULL)
    {
      printf("local_distance OOM\n");
    }
    float * local_all_data = &(((float*)get_distance_para->all_data)[start*dims]);
    int i, j;

    float tmp;
    volatile int reply = 0;
    dma_desc put_distance;

    dma_set_op(&put_distance, DMA_PUT);
    dma_set_mode(&put_distance, PE_MODE);
    dma_set_reply(&put_distance, &reply);
    dma_set_size(&put_distance, local_data_size * sizeof(float));
    for(i = 0; i < data_size; ++i)
    {
        for(j = 0; j < local_data_size; ++j)
        {
            tmp = get_squared_euclidean_distance(data + i * dims, local_all_data + j * dims, dims);
            //distance[i * all_data_size + start + j] = tmp;
            local_distance[j] = tmp;
        }
        dma(put_distance, (long)(distance + i * all_data_size + start), (long)local_distance);
        dma_wait(&reply, 1); reply = 0;
        if(id == 0)
        {
          printf("%d\n", i);
          //printf("local_distance=%f, distance=%f\n", local_distance[0], distance[i * all_data_size]);
        }
    }
}
