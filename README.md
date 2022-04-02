# DRL-implementations
Implementation of DRL algorithms for solving gym environments. Setup dependencies by installing spinningup. There are baseline implementations of some DRL algorithms in spinningup. 


## run examples
```bash
# can use command line to modify the parameters of dqn agent
# use tensorboard to visualize the results
# e.g python example/dqn.py --name <configuration-name> --render 
python <name-of-the-algorithm>.py --name <configuration-name> --render 
```

### nec.py
```bash
python nec.py --name nec-01 --render
```

## remarks
- plotting_utils.py modified from https://github.com/NVIDIA/tacotron2/blob/dd49ffa85085d33f5c9c6cb6dc60ad78aa00fc04/logger.py


## TODO
2. unittests