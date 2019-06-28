#cd data/ && ./clear.sh && cd ..
#make clean
#make kmeansTest

for i in {1..1000}
do
  bsub -b -q q_sw_expr -n 4 -cgsp 64 -share_size 6144 -host_stack 128 -o log/log_$i.txt ./kmeansTest  ./data/4251.dat 4251 $i 54675 0 ./data/distance1.dat 0 0
  #echo $i;
done
