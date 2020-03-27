# DRL-implementations
Implementation of DRL algorithms for solving gym environments. Setup dependencies by installing spinningup. There are baseline implementations of some DRL algorithms in spinningup. 


## install spinningup
install openMPI
```
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
install spinningup
```
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```


## run experiments
1. set hyparams in "hyparams" folder
2. run script, e.g
```
python dqn --name=dqn-01
```
3. run the following to check tensorboard
```
tensorboard --logdir=./experiment
```


## TODO
### dqn.py
1. check ReplayBuffer; (DONE) append(), (DONE) get_batch(), (DONE) __len__()
2. check QNetwork; (DONE) forward()
3. check DQNAgent; (DONE) replay(), (DONE) greedy_infer(), (DONE) epsilon_greedy_infer(), (DONE) remember(), (DONE) update_target_network()
4. debug training; loss exploded, q value exploded
5. try a range of target_update_freq

### general
2. (DONE) tensorboard_logger for logging training process

