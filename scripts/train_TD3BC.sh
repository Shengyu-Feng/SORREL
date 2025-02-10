export DATADIR=.

python 03_train_offline.py --seed 2 --model_name TD3BC --task 1_set_covering  --eval_metric Nb_nodes --batch_size 32 --epoch_size 624 --alpha 0.01 --kappa 0.8 --gamma 0.9  --time_limit 600  --patience 5 --early_stop 10 --warmup --eval_level valid --target_update_period 2 --tree  --num_job 20 --dataset_dir $DATADIR/datasets --instance_dir $DATADIR/instances --save_dir $DATADIR/checkpoints 

#python 03_train_offline.py --seed 0 --model_name TD3BC --task 2_independent_set  --eval_metric Nb_nodes --batch_size 32 --epoch_size 624 --alpha 0.01 --kappa 0.8 --gamma 0.9  --time_limit 600  --patience 5 --early_stop 10 --warmup --eval_level valid --target_update_period 2 --tree  --num_job 20 --dataset_dir $DATADIR/datasets --instance_dir $DATADIR/instances --save_dir $DATADIR/checkpoints 

#python 03_train_offline.py --seed 2 --model_name TD3BC --task 3_combinatorial_auction --eval_metric Nb_nodes --batch_size 32 --epoch_size 624 --alpha 0.1 --kappa 0.8 --gamma 0.9  --time_limit 600  --patience 5 --early_stop 10 --warmup --eval_level valid --target_update_period 2 --tree  --num_job 20 --dataset_dir $DATADIR/datasets --instance_dir $DATADIR/instances --save_dir $DATADIR/checkpoints 

#python 03_train_offline.py --seed 1 --model_name TD3BC --task 4_facility_location --eval_metric Nb_nodes --batch_size 32 --epoch_size 624 --alpha 0.1 --kappa 0.8 --gamma 0.9  --time_limit 600  --patience 5 --early_stop 10 --warmup --eval_level valid --target_update_period 2 --tree  --num_job 20 --dataset_dir $DATADIR/datasets --instance_dir $DATADIR/instances --save_dir $DATADIR/checkpoints 

#python 03_train_offline.py --seed 1 --model_name TD3BC --task 5_multi_knapsack --eval_metric Nb_nodes --batch_size 32 --epoch_size 624 --alpha 2.5 --kappa 0.8 --gamma 0.9  --time_limit 600  --patience 5 --early_stop 10 --warmup --eval_level valid --target_update_period 2 --tree  --num_job 20 --dataset_dir $DATADIR/datasets --instance_dir $DATADIR/instances --save_dir $DATADIR/checkpoints 