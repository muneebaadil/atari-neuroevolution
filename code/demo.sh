#SCIPTS TO PLAY GAMES USING EVOLVED NETWORK

#AIR-RAID
python play.py --game AirRaid-v0 --input_dim 250x160x3 --num_hidden 32 --num_actions 6 --pretrained ../experiments/air-raid/network.npy --num_episodes 2

#ASTERIX 
python play.py --game Asterix-v0 --input_dim 210x160x3 --num_hidden 32 --num_actions 9 --pretrained ../experiments/asterix/network.npy --num_episodes 2

#ASSAULT
python play.py --game Assault-v0 --input_dim 250x160x3 --num_hidden 32 --num_actions 7 --pretrained ../experiments/assault/network.npy --num_episodes 2

#BEAMRIDER
python play.py --game BeamRider-v0 --input_dim 210x160x3 --num_hidden 32 --num_actions 9 --pretrained ../experiments/beamrider/network.npy