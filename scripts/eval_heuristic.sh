export DATADIR=.

for task_name in 1_set_covering 2_independent_set 3_combinatorial_auction 4_facility_location 5_multi_knapsack
do
python evaluate.py --model_name heuristic --task_name 1_set_covering --FSB_probability 0.05 --eval_level test_standard --instance_dir $DATADIR/instances --save_dir $DATASET/checkpoints --time_limit 3600

python evaluate.py --model_name heuristic --task_name 1_set_covering --FSB_probability 1.0 --eval_level test_standard --instance_dir $DATADIR/instances --save_dir $DATASET/checkpoints --time_limit 3600
done
