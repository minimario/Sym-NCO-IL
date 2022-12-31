export CUDA_VISIBLE_DEVICES=2
python run.py \
	--label_aug \
	--consistancy_learning \
    --problem cvrp \
	--val_dataset expert_data/vrp20_valid_problem_3456.pkl \
	--train_dataset expert_data/vrp20_problem_2345.pkl \
	--wandb_run_name orig_label_cons \
	--labelpath expert_data/vrp20_solution_2345.pkl

# sleep 5

# export CUDA_VISIBLE_DEVICES=7
# python run.py \
#     --problem cvrp \
# 	--label_aug \
# 	--consistancy_learning \
# 	--val_dataset expert_data/vrp20_valid_problem_3456.pkl \
# 	--train_dataset expert_data/vrp20_problem_2345.pkl \
# 	--num_equivariant_samples 5 \
# 	--wandb_run_name eq_label_cons_lambda5 \
# 	--supervise_lambda 5 \
# 	--labelpath expert_data/vrp20_solution_2345.pkl &

# sleep 5

# export CUDA_VISIBLE_DEVICES=6
# python run.py \
#     --problem cvrp \
# 	--val_dataset expert_data/vrp20_valid_problem_3456.pkl \
# 	--train_dataset expert_data/vrp20_problem_2345.pkl \
# 	--num_equivariant_samples 50 \
# 	--wandb_run_name eq_50 \
# 	--labelpath expert_data/vrp20_solution_2345.pkl &

# sleep 5

# export CUDA_VISIBLE_DEVICES=7
# python run.py \
#     --problem cvrp \
# 	--label_aug \
# 	--consistancy_learning \
# 	--val_dataset expert_data/vrp20_valid_problem_3456.pkl \
# 	--train_dataset expert_data/vrp20_problem_2345.pkl \
# 	--wandb_run_name minsu_flags_all \
# 	--labelpath expert_data/vrp20_solution_2345.pkl &