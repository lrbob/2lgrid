export OMP_NUM_THREADS=1    # 防止线程超卖
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g \
  python main.py --alg PPO --env-id bus14 --difficulty 1 \
                 --action-type topology --cuda true \
                 --n-envs 32 --n-steps 64000 \
                 --actor-layers 512 512 256 \
                 --critic-layers 512 512 256 \
                 --total-timesteps 200000000 \
                 --seed $g --wandb-mode offline &
done
wait