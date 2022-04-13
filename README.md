# DRL-implementations
Implementation of DRL algorithms for solving gym environments. Setup dependencies by installing spinningup. There are baseline implementations of some DRL algorithms in spinningup. 


## run examples
```bash
# can use command line to modify the parameters of dqn agent
# use tensorboard to visualize the results
# python example/dqn.py --name <configuration-name> --render 
python <name-of-the-algorithm>.py --name <configuration-name> --render 
```

## run trained agent
```bash
python run_drl.py --name <configuration-name> --render --agent <name-of-agent> --ckpt <ckpt-filepath-of-agent>
```
![LunarLander-v2 dqn](./LunarLander-v2_dqn.gif?raw=true "LunarLander-v2 dqn")


## remarks
- plotting_utils.py modified from https://github.com/NVIDIA/tacotron2/blob/dd49ffa85085d33f5c9c6cb6dc60ad78aa00fc04/logger.py
- Trainer and Agent only works for discrete action space
- Do not support pixel observation yet
- nec_agent is still testing


## TODO
1. unittests