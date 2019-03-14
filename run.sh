cd data/ && ./clear.sh && cd ..
#make clean
make kmeansTest
bsub -b -I -q q_sw_expr -n 1 -cgsp 64 -share_size 6144 -host_stack 128 ./kmeansTest  ./data/gse2109_54675_895.dat 54675 1000 895 0
#bsub -b -I -q q_sw_expr -N 1 -np 1 -cgsp 64 ./kmeansTest  ./data/gse2109_54675_895.dat 54675 1 895 0
#bsub -b -I -q q_sw_expr -N 1 -np 4 -cgsp 64 -node 80 ./kmeansTest  ./data/3D_spatial_network.dat 434874 3 3 0
#bsub -b -I -q q_sw_expr -N 1 -np 4 -cgsp 64 ./kmeansTest  ./data/test.dat 4 1 2 0
#bsub -b -I -q q_sw_expr -N 1 -np 4 -cgsp 64 ./kmeansTest  ./data/3D_spatial_network.dat 434874 100 3 0

#bsub -b -I -q q_sw_expr -N 1 -np 1 -cgsp 64 ./kmeansTest  ./data/3D_spatial_network.dat 434874 100 3 0


#cd /home/export/online1/swyf/swdnn/fjr/SWCaffe/SWCaffe-lld-911/KMeans

#bsub -b -I -q q_sw_expr -N 1 -np 4 -cgsp 64  -host_stack 256 -share_size 6800  ./kmeansTest  ../data/imagenet_kmeans_256.dat 1265723 20 196608 0 
