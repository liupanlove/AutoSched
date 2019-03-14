#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>  
#include <fcntl.h>
#include <string.h>  
#include <time.h>  
#include <assert.h>
#include <simd.h>
int init_data(char *filename,char bRdOrWr)
{
    mode_t mode;

	if(bRdOrWr > 0) 
		mode = O_CREAT |  O_RDWR;
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

void load_data(int fd, float *data, int len,int offset)
{
    int ret;
	  lseek(fd,offset,SEEK_SET);
    ret = read(fd,data,len);

    if (ret < 0) {
        perror("read error:");
        exit(-1);
    } 
}

void store_data(int fd, float *data, int len)
{
    int ret;
    ret = write(fd,data,len);

    if (ret < 0) {
        perror("write error:");
        exit(-1);
    } 
}  
void precess_data(char *in_filename,char *out_filename,int dims)
{
  FILE *fp;
  fp = fopen(in_filename, "r+");
  if(fp == NULL)
  {
     printf("Cannot open the file = %s!\n",in_filename);
     exit(0);
  }
  int fdwr = init_data(out_filename,1);
  if(fdwr == 0)
  {
     printf("Cannot open the file = %s!\n",out_filename);
     exit(0);
  }
  float val[dims];
  int size = 0,count = 0,i=0;
  char *line = NULL;
  char delims[] = ",";  
  char *result = NULL,*save = NULL;  
  while (getline(&line,&size,fp) != -1)
  {  
     result = strtok_r(line,delims,&save);
     result = strtok_r(NULL,delims,&save);
     i = 0;
     while(result){
        val[i++] = atof(result);
        result = strtok_r(NULL,delims,&save);
     }
     store_data(fdwr,val,(dims - 1) * sizeof(float));
     printf("\nProcessing count=%d\n",count++);
  }
  if(line) free(line);
  if(save) free(save);
  fclose(fp);
  close(fdwr);
}
void precess_data1(char *in_filename,char *out_filename)
{
  int rdfd = 0;
	struct stat thestat;
  int wrfd = open(out_filename,O_WRONLY | O_APPEND);
  if(wrfd == 0)
  {
     printf("Cannot open the file = %s!\n",out_filename);
     exit(0);
  }
    
  rdfd = open(in_filename,O_RDONLY);
	if(rdfd < 0)
	{
     printf("init_data data fileName error filename =%s\n",in_filename);
     exit(0);
	}
	if(fstat(rdfd, &thestat) < 0) {
		 close(rdfd);
     printf("fstat error %s\n",in_filename);
     exit(0);
  }
  unsigned char * data = (unsigned char *)malloc(thestat.st_size);
  assert(data != NULL);
  int ret;
  ret = read(rdfd,data,thestat.st_size);

  if (ret < 0) {
     perror("read error:");
     exit(-1);
  } 
  ret = write(wrfd,data,thestat.st_size);

  if (ret < 0) {
     perror("write error:");
     exit(-1);
  } 
  free(data);
  close(rdfd);
  close(wrfd);
}
inline unsigned long rpcc_usec()
{
   struct timeval   val;
   gettimeofday(&val,NULL);
   return (val.tv_sec*1000000 + val.tv_usec);
}
int main(int argc, char* argv[]){
  /*
  FILE *fp;
  fp = fopen(argv[2], "r+");
  if(fp == NULL)
  {
     printf("Cannot open the file = %s!\n",argv[2]);
     exit(0);
  }
  int count = atoi(argv[3]) -1;
  int i,j;
  float val;
  for(i=0;i<10;i++){
    for(j=0;j<count;j++){
      fread(&val,1,sizeof(float),fp);
      printf("%lf ",val);
    }
    printf("\n");
  }
  fclose(fp);
  return 0;
  */
    if( argc != 4){  
        printf("This application need other parameter to run:"  
                "\n\t\tthe first is input file name that contain data"  
                "\n\t\tthe second is output filename"  
                "\n\t\tthe third indicate the data dimension"  
                "\n");  
        exit(0);  
    } 
    printf("process data start!");
    int dims=atoi(argv[3]);
    precess_data(argv[1],argv[2],dims);
    //precess_data1(argv[1],argv[2]);
    printf("process data done!");
    return 0;
}
