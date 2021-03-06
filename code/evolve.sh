# SCRIPTS FOR ATARI GAMES USING EVOLVED NETWORKS

#AIR-RAID
python evolve.py --game AirRaid-v0 --input_dim 250x160x3 --num_hidden 32 --num_actions 6 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 --exp_root_dir ../experiments/ --exp_name air-raid

#ASTERIX
python evolve.py --game Asterix-v0 --input_dim 210x160x3 --num_hidden 32 --num_actions 9 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 --exp_root_dir ../experiments/ --exp_name asterix

#ASSAULT
python evolve.py --game Assault-v0 --input_dim 250x160x3 --num_hidden 32 --num_actions 7 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 --exp_root_dir ../experiments/ --exp_name assault

#BEAM-RIDER
python evolve.py --game BeamRider-v0 --input_dim 210x160x3 --num_hidden 32 --num_actions 9 --population_size 10 --num_select 10 --num_gens 100 --save_every 10 --exp_root_dir ../experiments/ --exp_name beamrider