export DATADIR=.

python 02_generate_dataset.py --task_name 1_set_covering --FSB_probability 0.05 --tree --instance_dir $DATADIR/instances --dataset_dir $DATADIR/datasets --num_samples 1000 --num_processes 10 --length_limit 200 --sampling_threshold 100

#python 02_generate_dataset.py --task_name 2_independent_set --FSB_probability 0.05 --tree --instance_dir $DATADIR/instances --dataset_dir $DATADIR/datasets --num_samples 100000 --num_processes 20 --length_limit 500 --length_limit_min 100

#python 02_generate_dataset.py --task_name 3_combinatorial_auction --FSB_probability 0.05 --tree --instance_dir $DATADIR/instances --dataset_dir $DATADIR/datasets --num_samples 100000 --num_processes 20 --length_limit 2000 --sampling_threshold 200

#python 02_generate_dataset.py --task_name 4_facility_location --FSB_probability 0.05 --tree --instance_dir $DATADIR/instances --dataset_dir $DATADIR/datasets --num_samples 100000 --num_processes 20 --length_limit 1000 --sampling_threshold 200

#python 02_generate_dataset.py --task_name 5_multi_knapsack --FSB_probability 0.05 --tree --instance_dir $DATADIR/instances --dataset_dir $DATADIR/datasets --num_samples 100000 --num_processes 20 --length_limit 1000 --sampling_threshold 200