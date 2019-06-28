#cd data/ && ./clear.sh && cd ..
make clean
make kmeansTest

for ((i=573; i<=768; i+=20))
do
  #for ((j=1; j<=10; ++j))
  #do
  bsub -b -q q_sw_expr -n 4 -cgsp 64 -share_size 6144 -host_stack 128 ./kmeansTest  ./data/new_MULTMYEL.dat 54675 $i 559 0 ./data/distance1.dat 0 0
  #bsub -b -q q_sw_expr -n 1 -cgsp 64 -share_size 6144 -host_stack 128 ./kmeansTest  ./data/4251.dat 4251 $i 54675 0 ./data/distance1.dat 2 0
    #bsub -b -q q_sw_expr -n 1 -cgsp 64 -share_size 6144 -host_stack 128 ./kmeansTest  ./data/MULTMYEL.dat 559 $i 54675 0 ./data/distance1.dat 2 0
    #echo $i;

  #done
done
