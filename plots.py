import numpy as np
import matplotlib.pyplot as plt

x = np.load('batch_size_128_epsilon_0.1_exploration_p_egreedy_gamma_1.0_learning_rate_0.0001_memory_size_100000_no_replay_buffer_False_no_target_network_False_target_update_500_temperature_0.1.npy')
plt.plot(x)
# plt.show()
plt.savefig('temp.png')