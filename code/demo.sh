# SCRIPTS FOR EVOLVING NEURAL NETWORKS FOR ATARI GAMES

#evolve on PONG
python evolve.py --num_hidden 64 --population_size 20 --num_gens 100 --num_select 10 
--game_name Pong-v0 --exp_root_dir /datadrive/muneeb/atari/experiments/ --save_every 10