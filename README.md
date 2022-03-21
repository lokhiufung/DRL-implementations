# DRL-implementations
Implementation of DRL algorithms for solving gym environments. Setup dependencies by installing spinningup. There are baseline implementations of some DRL algorithms in spinningup. 


## install torchviz for visualizing computation graph
```
sudo apt update
sudo apt-get install graphviz

pip install torchviz
```

## install spinningup (**may not need now**)
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
```bash
# can use command line to modify the parameters of dqn agent
# use tensorboard to visualize the results
python <name-of-the-algorithm>.py --name <configuration-name> --render 
```


### a3c.py (needed pytorch lightning version)
1. check if clip_grad_norm_ works 
2. check if step() correctly applied on model
3. prob_tensor becomes nan after several steps
4.   File "a3c.py", line 66, in train
    action_tensor = prob_tensor.multinomial(num_samples=1).detach()
RuntimeError: invalid multinomial distribution (encountering probability entry < 0)


### nec.py
```bash
python nec.py --name nec-01 --render
```

## remarks
- plotting_utils.py modified from https://github.com/NVIDIA/tacotron2/blob/dd49ffa85085d33f5c9c6cb6dc60ad78aa00fc04/logger.py
- [ dqn ] loss boosts up everytime after target update; score may drop after many steps; loss may increase after many steps; very unstable
- [ a3c ] no improvement in early epidsodes; very unstable


## TODO
1. refactor algorithms (e.g extract common components, make a common training loop)
2. unittests