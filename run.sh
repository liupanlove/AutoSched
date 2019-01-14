cd /home/export/online1/swyf/swdnn/fjr/SWCaffe/SWCaffe-lld-911/KMeans

bsub -b -I -q q_sw_expr -N 1 -np 4 -cgsp 64  -host_stack 256 -share_size 6800  ./kmeansTest  ../data/imagenet_kmeans_256.dat 1265723 20 196608 0 
