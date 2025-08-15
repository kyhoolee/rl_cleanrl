## Common

tensorboard --logdir gem_ppo_runs/

## I. Policy-based

python -u  _1_1_gem_ppo.py --log-folder gem_ppo_runs --env-id CartPole-v1 --total-timesteps 50000 > _1_log_cartpole_gem_ppo.log


python -u _1_1_gem_ppo.py --run-hyperparam-search --log-folder gem_ppo_runs --env-id CartPole-v1 --total-timesteps 50000 > _1_log_hyperparam_search.log


## II. Value-based 

nohup python -u  _2_1_dqn.py --log-folder dqn_runs --env-id CartPole-v1 --total-timesteps 50000 > _2_1_log_cartpole_dqn.log 2>&1 &


nohup python -u  _2_2_c51.py --log-folder c51_runs --env-id CartPole-v1 --total-timesteps 50000 > _2_2_log_cartpole_c51.log 2>&1 &