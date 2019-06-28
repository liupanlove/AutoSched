#include <stdio.h>
#include <simd.h>
#include <assert.h>
#include <athread.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include "mpi.h"

extern SLAVE_FUN(caculate_distance)();

typedef struct _GetDistancePara{
    int data_size;
    int all_data_size;
    int dims;

    float * distance;
    float * data;
    float * all_data;
} GetDistancePara;

int init_data(char *filename,char bRdOrWr)
{
    mode_t mode;
	  if(bRdOrWr > 0)
		    mode = O_CREAT |  O_WRONLY;
	  else
	      mode = O_RDONLY;

    int fd;

    fd = open(filename,mode,S_IRWXU|S_IRWXG|S_IRWXO);

    if(fd<0) {
	    printf("open %s failed\n",filename);
        return -1;
    }
    return fd;
}

void load_data(int fd, float *data,long len,long offset)
{
    int ret;
	  lseek(fd,offset,SEEK_SET);
    ret = read(fd,data,len);

    if (ret < 0) {
        perror("read error:");
        exit(-1);
    }
}

void store_data(int fd, float * data, int len)
{
    int ret;
    ret = write(fd, data, len);
    if(ret < 0)
    {
        perror("write error:");
        exit(-1);
    }
}

int readDataFromFile(int rank, int start, int dims,int data_size, unsigned char *filename, float ** data, float* all_data)
{
    int rdfd = 0;
	  struct stat thestat;

    rdfd = init_data(filename,0);
    //printf("%s\n", filename);
    if(rdfd < 0)
    {
        if(rank == 0)
            printf("init_data data fileName error filename =%s\n",filename);
        return -1;
    }
    if(fstat(rdfd, &thestat) < 0) {
        close(rdfd);
        if(rank == 0)
            printf("fstat error %s\n",filename);
        return -1;
    }
	  long size = thestat.st_size/sizeof(float);
    long total_data_size = data_size * dims;
	  if(size != total_data_size)
	  {
        if(rank == 0)
		    printf("data format and size error:%s %ld %ld %d\n",filename,size,total_data_size,size/dims);
		    return -1;
	  }
	  /*long len = local_count;
    len *= dims;
    len *= sizeof(float);
    long offset = start;
    offset *= dims;
    offset *= sizeof(float);*/

	  load_data(rdfd, all_data, thestat.st_size, 0); // 0 == offset
    /*if(rank == 0) printf("bcast all_data start...\n");
    caffe_mpi_bcast_f(all_data, total_data_size, 0, MPI_COMM_WORLD);
    if(rank == 0) printf("bcast all_data done...\n");*/  // it's not making sense
    //printf("%x  %f\n", all_data + 1, *(all_data + 1));
    *data = all_data + start * dims;//&(((float*)all_data)[start*dims]);
    printf("total_data_size=%d\n", thestat.st_size);
    close(rdfd);
	  return 1;
}

void write_data(char * out_filename, int rank, int numprocs, int data_size, float * distance, long distance_size)
{
    /*if(rank == 0)
    {
        int fdwr = init_data(filename, 1);
        if(fdwr == 0)
        {
            printf("cannot open the file = %s \n", filename);
            exit(0);
        }

        store_data(fdwr, distance, (long)(distance_size * (long)sizeof(float)));
        printf("store_data completes\n");
        int i;
        int other_distance_size;
        for(i = 1; i < numprocs; ++i)// MPI_RECV(void*, int )
        {
            other_distance_size = data_size/numprocs + (i<(data_size%numprocs));
            MPI_Recv(distance, other_distance_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            store_data(fdwr, distance, other_distance_size * sizeof(float));
            printf("%d\n", i);
        }
        close(fdwr);
    }
    else
    {
      printf("rank = %d start %d \n", rank, distance_size);
      MPI_Send(distance, distance_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      printf("rank = %d stop \n", rank);
    }*/

    char filename[50];
    sprintf(filename, "./data/distance_%d.dat", rank);

    int fdwr = init_data(filename, 1);
    if(fdwr == 0)
    {
        printf("cannot open the file = %s\n", filename);
        exit(0);
    }

    store_data(fdwr, distance, distance_size);

    close(fdwr);

    if(rank == 0)
    {
        int i, fdrd;
        fdwr = init_data(out_filename, 1);
        struct stat thestat;

        for(i = 0; i < numprocs; ++i)
        {
            char filename1[50];
            sprintf(filename1, "./data/distance_%d.dat", i);

            fdrd = init_data(filename1, 0);
            if(fdrd < 0)
            {
                printf("init_data filename error\n");
                return;
            }
            if(fstat(fdrd, &thestat) < 0)
            {
                close(fdrd);
                if(rank == 0)
                {
                    printf("fstat error \n");
                    return;
                }
            }
            printf("file size %d %ld\n", i, thestat.st_size);
            load_data(fdrd, distance, thestat.st_size, 0);
            store_data(fdwr, distance, thestat.st_size);
            close(fdrd);
        }
        close(fdwr);
    }
}

int main(int argc, char**argv)
{
    if(argc != 5)
    {
        printf("There are 4 arguments, the first one is data file, the second one is n, the third one is d, the fourth one is out filename\n");
        exit(0);
    }
    athread_init();
    unsigned char filename[50];
    strcpy(filename, argv[1]);
    unsigned char out_filename[50];
    strcpy(out_filename, argv[4]);
    int data_size = atoi(argv[2]);
    int dims = atoi(argv[3]);
    int rank = 0, numprocs = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm, &numprocs);

    int start=rank*(data_size/numprocs) + (rank<(data_size%numprocs) ? rank : (data_size%numprocs));
    int local_count=data_size/numprocs + (rank<(data_size%numprocs));
    float * data = NULL;
    float * all_data = (float *)malloc(data_size * dims * sizeof(float));
    assert(all_data != NULL);
    int ret = readDataFromFile(rank, start, dims, data_size, filename, &data, all_data);
    if(ret < 1)
    {
        free(all_data);
        MPI_Finalize();
        return -1;
    }
    assert(data != NULL);

    long distance_size = (long)local_count * (long)data_size * (long)sizeof(float);
    printf("distance_size=%ld\n", distance_size);
    float * distance = (float *) malloc(distance_size); //local_count * data_size * sizeof(float));
    assert(distance != NULL);

    GetDistancePara get_distance_para;
    get_distance_para.data_size = local_count;
    get_distance_para.all_data_size = data_size;
    get_distance_para.dims = dims;
    get_distance_para.data = data;
    get_distance_para.all_data = all_data;
    get_distance_para.distance = distance;


    printf("athread_spawn start\n");
    athread_spawn(caculate_distance, (void *)(&get_distance_para));
    athread_join();

    write_data(out_filename, rank, numprocs, data_size, distance, distance_size);

    free(distance);
    free(all_data);

    MPI_Finalize();
    athread_halt();
    return 0;
}
