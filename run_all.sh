#cd data/ && ./clear.sh && cd ..
make clean
make kmeansTest

for i in {1..10}
do
  bsub -b -q q_sw_expr -n 1 -cgsp 64 -share_size 6144 -host_stack 128 ./kmeansTest  ./data/MULTMYEL.dat 559 $i 54675 0 ./data/distance1.dat 2 0
  #echo $i;
done
