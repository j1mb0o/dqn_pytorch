import argparse


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exploration_p', type=str,default='egreedy', action='store')
    parser.add_argument('--no-replay-buffer',  action='store_true')
    parser.add_argument('--no-target-network', action='store_true')
    parser.add_argument('--learning-rate', type=float,default=1e-4)
    parser.add_argument('--batch-size',type=int, default=128)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default= 0.1)
    parser.add_argument('--target-update',type=int, default=500)
    parser.add_argument('--memory-size',type=int,default=10e4)
    
    args = parser.parse_args()
    
    print(args)