LINK = mpicc
CC = mpicc -host
SWCC = sw5cc.new 
FLAGS = -O3 -OPT:IEEE_arith=2
FLAGS += -DUSE_MPI
#FLAGS += -DUSE_4CG
#FLAGS += -DDEBUG
FLAGS += -DPRINT_ONE_ROUND_TIME
#FLAGS += -DDEBUG_CENTER
#LDFLAGS = -lm -lpthread  -allshare
LDFLAGS = -lm 

all: kmeansTest preprocess_data read_tiff

read_tiff.o:read_tiff.c
	$(SWCC) -host  $(FLAGS)  -c -msimd  read_tiff.c
read_tiff:read_tiff.o
	$(LINK) read_tiff.o -o read_tiff
preprocess_data.o:preprocess_data.c
	$(SWCC) -host  $(FLAGS)  -c preprocess_data.c
preprocess_data:preprocess_data.o
	$(LINK) preprocess_data.o -o preprocess_data

kmeansTest:master.o slave.o sw_memcpy.o sw_add.o sw_div.o sw_slave_memcpy.o sw_slave_add.o sw_slave_div.o sw_slave_memset.o sw_memset.o
	$(LINK) master.o slave.o sw_memcpy.o sw_add.o sw_div.o sw_slave_memcpy.o sw_slave_add.o sw_slave_div.o sw_slave_memset.o sw_memset.o  -o kmeansTest $(LDFLAGS)
	
master.o:master.c
	$(CC)  $(FLAGS)  -c -msimd  -I/usr/sw-mpp/mpi2/include master.c 
	
sw_memset.o:./util/sw_memset.c
	$(SWCC) -host  $(FLAGS)  -c -msimd  ./util/sw_memset.c
sw_memcpy.o:./util/sw_memcpy.c
	$(SWCC) -host  $(FLAGS)  -c -msimd  ./util/sw_memcpy.c
sw_add.o:./util/sw_add.c
	$(SWCC) -host  $(FLAGS)  -c -msimd  ./util/sw_add.c
sw_div.o:./util/sw_div.c
	$(SWCC) -host  $(FLAGS)  -c -msimd  ./util/sw_div.c

sw_slave_memset.o:./util/sw_slave_memset.c 
	$(SWCC) -slave $(FLAGS)  -c -msimd  ./util/sw_slave_memset.c
slave.o:slave.c 
	$(SWCC) -slave $(FLAGS)  -c -msimd  slave.c
sw_slave_memcpy.o:./util/sw_slave_memcpy.c 
	$(SWCC) -slave $(FLAGS)  -c -msimd  ./util/sw_slave_memcpy.c
sw_slave_add.o:./util/sw_slave_add.c 
	$(SWCC) -slave $(FLAGS)  -c -msimd  ./util/sw_slave_add.c
sw_slave_div.o:./util/sw_slave_div.c 
	$(SWCC) -slave $(FLAGS)  -c -msimd  ./util/sw_slave_div.c
	

read_tiff_run:
	bsub -b -I -q q_sw_expr -n 1 -np 4 -cgsp 64  -host_stack 2048 -share_size 4096  ./read_tiff  ../data/zy302a_mux.tif ../data/remote_sensing_image.dat 
generate_data:
#	bsub -b -I -q q_sw_share -n 1 -np 1 -cgsp 64  -host_stack 2048 -share_size 4096  ./preprocess_data  ../data/census1990.txt ../data/census1990.dat 69 
	bsub -b -I -q q_sw_yfb -n 1 -np 1 -cgsp 64  -host_stack 2048 -share_size 4096  ./preprocess_data  ./data/3D_spatial_network.txt ./data/3D_spatial_network.dat 4 
#	bsub -b -I -q q_sw_yfb -n 1 -np 1 -cgsp 64  -host_stack 2048 -share_size 4096  ./preprocess_data  ../data/Reaction_Network.txt ../data/Reaction_Network.dat 29 
#	bsub -b -I -q q_sw_yfb -n 1 -np 1 -cgsp 64  -host_stack 2048 -share_size 4096  ./preprocess_data  ../data/remote_image_114_030_0803.dat ../data/remote_image_114_030_0816.dat 29 
mpi_run:
	bsub -b -I -q q_sw_yfb -n 16 -np 1 -cgsp 64 -sw3runarg "-a 1" -host_stack 1024 -cross_size 28000  ./kmeansTest  ../data/cluster.dat 2256119 100000 4 1 
run:
	bsub -b -I -q q_sw_share -N 1 -np 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -host_stack 2048 -cross_size 26000  ./kmeansTest  ../data/3D_spatial_network.dat 434874 100000 3 0 
	#bsub -b -I -q q_sw_share -N 1 -np 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -host_stack 2048 -cross_size 26000  ./kmeansTest  ../data/Reaction_Network.dat 65554 8192 28 0 
	#bsub -b -I -q q_sw_share -N 1 -np 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -host_stack 1024 -cross_size 28000  ./kmeansTest  ../data/census1990.dat 2458285 4 68 0 
#	bsub -b -I -q q_sw_yfb -N 1024 -np 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -host_stack 1024 -cross_size 28000  ./kmeansTest  ../data/imagenet_kmeans_256.dat 1265723 1000 196608 0 
#1265720
												
clean:
	-rm -f kmeansTest *.o
