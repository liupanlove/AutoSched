#cd data/ && ./clear.sh && cd ..
#make clean
#make kmeansTest

for i in {1..1000}
do
  ((tmp=$i/250))  
  ((tmp=$tmp+1)) 
  ((var=$tmp*4))
  #echo $i.$var
  bsub -b -q q_sw_expr -n $var -cgsp 64 -share_size 6144 -host_stack 128 -o log1/log_$i.txt ./kmeansTest  ./data/4251.dat 4251 $i 54675 0 ./data/distance1.dat 0 0
  #echo $i;
done
