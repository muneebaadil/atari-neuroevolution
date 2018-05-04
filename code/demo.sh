# SCRIPTS FOR EVOLVING NEURAL NETWORKS FOR ATARI GAMES

#AIR-RAID
python evolve.py --game_name AirRaid-v0 --input_dim 250x160x3 --num_hidden 32 \
 --num_actions 6 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 \
 --exp_root_dir /datadrive/muneeb/atari/experiments/

#BREAKOUT
python evolve.py --game_name Breakout-v0 --input_dim 210x160x3 --num_hidden 32 \
 --num_actions 4 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 \ 
 --exp_root_dir /datadrive/muneeb/atari/experiments/

#ASTERIX
python evolve.py --game_name Asterix-v0 --input_dim 210x160x3 --num_hidden 32 \
 --num_actions 9 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 \ 
 --exp_root_dir /datadrive/muneeb/atari/experiments/

#ASSAULT
python evolve.py --game_name Assault-v0 --input_dim 250x160x3 --num_hidden 32 \
 --num_actions 7 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 \ 
 --exp_root_dir /datadrive/muneeb/atari/experiments/

#BATTLE-ZONE
python evolve.py --game_name BattleZone-v0 --input_dim 210x160x3 --num_hidden 32 \
 --num_actions 18 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 \ 
 --exp_root_dir /datadrive/muneeb/atari/experiments/

#BEAM-RIDER
python evolve.py --game_name BeamRider-v0 --input_dim 210x160x3 --num_hidden 32 \
 --num_actions 9 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 \ 
 --exp_root_dir /datadrive/muneeb/atari/experiments/