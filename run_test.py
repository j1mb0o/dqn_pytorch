import os

if __name__ == '__main__':
'''
    egreedy_exp = [0.01, 0.05, 0.2]
    
    input(f'You will run the following test {egreedy_exp=}, press enter to continue')
    for i in egreedy_exp:

        os.system(f'python3 dqn.py --epsilon {i}' )
'''
    temperature = [0.01,0.1,1]
    input(f'You will run the following test {temperature=}, press enter to continue')
    for i in temperature:

        os.system(f'python3 dqn.py --temperature {i}' )
'''
    lr = []
    batch_size = []

    input(f'You will run the following tests {lr=} and then {batch_size=}, press enter to continue')

    for i in lr:
        os.system(f'python3 dqn.py --learning-rate {i}' )

    for i in batch_size:
        os.system(f'python3 dqn.py --batch-size {i}' )
'''
