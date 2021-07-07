# Fisher_BRC
Implementation of Fisher_BRC in "Offline Reinforcement Learning with Fisher Divergence Critic Regularization" based on BRAC family.
 
**Usage**:
 Plug this file into BRAC architecture[https://github.com/google-research/google-research/tree/master/behavior_regularized_offline_rl] and run train_offline.py. 
 
 ```
python train_offline.py \
    --alsologtostderr --sub_dir=fisher_brc+$ALPHA \
    --root_dir=$ROOT_DIR \
    --env_name=$ENV \
    --agent_name=fisher_brc \
    --data_root_dir='data' \
    --data_name=$DATA \
    --total_train_steps=$TRAIN_STEPS \
    --seed=$SEED \
    --gin_bindings="fisherbrc_agent.Agent.alpha=1.0" \
    --gin_bindings="fisherbrc_agent.Agent.train_alpha_entropy=False" \
    --gin_bindings="fisherbrc_agent.Agent.alpha_entropy=0.0" \
    --gin_bindings="train_eval_offline.model_params=(((300, 300), (200, 200), (750, 750)), 2)" \
    --gin_bindings="train_eval_offline.batch_size=256" \
    --gin_bindings="train_eval_offline.seed=$SEED" \
    --gin_bindings="train_eval_offline.weight_decays=[$L2_WEIGHT]" \
    --gin_bindings="train_eval_offline.optimizers=(('adam', 1e-3), ('adam', $PLR), ('adam', 1e-3), ('adam', 1e-3))" 
```
