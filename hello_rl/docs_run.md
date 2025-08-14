python -u  _1_1_gem_ppo.py --log-folder gem_ppo_runs --env-id CartPole-v1 --total-timesteps 50000 > _1_log_cartpole_gem_ppo.log


python -u _1_1_gem_ppo.py --run-hyperparam-search --log-folder gem_ppo_runs --env-id CartPole-v1 --total-timesteps 50000 > _1_log_hyperparam_search.log