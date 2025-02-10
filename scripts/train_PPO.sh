export DATADIR=.

python 04_train_online.py --seed 4 --model_name PPO --task 1_set_covering --eval_metric Nb_nodes --batch_size 32 --epoch_size 20 --num_job 20 --batch_size 32  --time_limit 600 --patience 10 --early_stop 20 --ckpt_path $DATADIR/checkpoints/1_set_covering/trained_models/TD3BC_0.01_2.pth --instance_dir $DATADIR/instances  --save_dir $DATADIR/checkpoints --warmup --eval_level valid --tree

#python 04_train_online.py --seed 4 --model_name PPO --task 2_independent_set --eval_metric Nb_nodes --batch_size 32 --epoch_size 20 --num_job 20 --batch_size 32  --time_limit 600 --patience 10 --early_stop 20 --ckpt_path $DATADIR/checkpoints/2_independent_set/trained_models/TD3BC_0.01_0.pth --instance_dir $DATADIR/instances  --save_dir $DATADIR/checkpoints --warmup --eval_level valid --tree

#python 04_train_online.py --seed 2 --model_name PPO --task 3_combinatorial_auction --eval_metric Nb_nodes --batch_size 32 --epoch_size 20 --num_job 20 --batch_size 32  --time_limit 600 --patience 10 --early_stop 20 --ckpt_path $DATADIR/checkpoints/3_combinatorial_auction/trained_models/TD3BC_0.1_2.pth --instance_dir $DATADIR/instances  --save_dir $DATADIR/checkpoints --warmup --eval_level valid --tree

#python 04_train_online.py --seed 2 --model_name PPO --task 4_facility_location --eval_metric Nb_nodes --batch_size 32 --epoch_size 20 --num_job 20 --batch_size 32  --time_limit 600 --patience 10 --early_stop 20 --ckpt_path $DATADIR/checkpoints/4_facility_location/trained_models/TD3BC_0.1_1.pth --instance_dir $DATADIR/instances  --save_dir $DATADIR/checkpoints --warmup --eval_level valid --tree

#python 04_train_online.py --seed 1 --model_name PPO --task 5_multi_knapsack --eval_metric Nb_nodes --batch_size 32 --epoch_size 20 --num_job 20 --batch_size 32  --time_limit 600 --patience 10 --early_stop 20 --ckpt_path $DATADIR/checkpoints/5_multi_knapsack/trained_models/TD3BC_2.5_1.pth --instance_dir $DATADIR/instances  --save_dir $DATADIR/checkpoints --warmup --eval_level valid --tree