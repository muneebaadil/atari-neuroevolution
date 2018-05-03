# SCRIPTS FOR EVOLVING NEURAL NETWORKS FOR ATARI GAMES

#evolve on PONG
python evolve.py --num_hidden 256 --population_size 500 --num_gens 300 --num_select 10 
--game_name pong-v0 --exp_root_dir /datadrive/muneeb/atari/experiments/ --save_every 25