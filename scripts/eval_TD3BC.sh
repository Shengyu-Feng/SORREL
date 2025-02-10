export DATADIR=.

python evaluate.py --model_name TD3BC --task_name 1_set_covering --eval_level test_standard --tree --time_limit 3600 --ckpt_path $DATADIR/checkpoints/1_set_covering/trained_models/PPO_4.pth --instance_dir $DATADIR/instances

python evaluate.py --model_name TD3BC --task_name 2_independent_set --eval_level test_standard --tree --time_limit 3600 --ckpt_path $DATADIR/checkpoints/2_independent_set/trained_models/PPO_4.pth --instance_dir $DATADIR/instances

python evaluate.py --model_name TD3BC --task_name 3_combinatorial_auction --eval_level test_standard --tree --time_limit 3600 --ckpt_path $DATADIR/checkpoints/3_combinatorial_auction/trained_models/PPO_2.pth --instance_dir $DATADIR/instances

python evaluate.py --model_name TD3BC --task_name 4_facility_location --eval_level test_standard --tree --time_limit 3600 --ckpt_path $DATADIR/checkpoints/4_facility_location/trained_models/PPO_2.pth --instance_dir $DATADIR/instances

python evaluate.py --model_name TD3BC --task_name 5_multi_knapsack --eval_level test_standard --tree --time_limit 3600 --ckpt_path $DATADIR/checkpoints/5_multi_knapsack/trained_models/PPO_0.pth --instance_dir $DATADIR/instances


