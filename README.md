
# DQN Readme




In order to run en experiment you should execute the python `dqn.py` file and pass the arguments from the terminal

An example with each parameter through the terminal
```bash
  python3 dqn.py --learning-rate 1e-4 --batch-size 64 --number-of-layers 3
    --num-of-neurons 128 --optimizer adam --exploration_p egreedy --epsilon 0.1
    --no-replay-buffer --no-target-network --gamma 1.0 --memory-size 10e4
    --numpy-filename numpy_filename_to_save_results

```
Note: that each argument is optional except the `--numpy-filename`

