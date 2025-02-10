export DATADIR=.

for task_name in 1_set_covering 2_independent_set 3_combinatorial_auction 4_facility_location 5_multi_knapsack
do
python evaluate.py --model_name SCIP --task_name $task_name --eval_level test_standard --instance_dir $DATADIR/instances --save_dir $DATASET/checkpoints --time_limit 3600
done