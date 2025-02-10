export DATADIR=.
for task_name in 1_set_covering 2_independent_set 3_combinatorial_auction 4_facility_location 5_multi_knapsack
do
python 01_generate_instance.py --task_name $task_name --num_offline 10000 --num_online 200 --num_valid 20 --num_standard 100 --num_transfer 20 --instance_dir $DATADIR/instances
done
