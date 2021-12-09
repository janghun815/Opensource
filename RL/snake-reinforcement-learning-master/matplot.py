import matplotlib.pyplot as plt
import numpy as np


class pplot:
    def reward_loss_plot(self):

        x, y, z = np.loadtxt('test_plot.txt', delimiter = ', ', unpack = True)

        plt.subplot(2,1,1)
        plt.plot(x,z,'o-')
        plt.title('average loss per episode')
        plt.xlabel('episode')
        plt.ylabel('average loss')

        plt.subplot(2,1,2)
        plt.plot(x,y,'.-')
        plt.title('average reward per episode')
        plt.xlabel('episode')
        plt.ylabel('average reward')
        plt.tight_layout()
        plt.savefig('save_averageloss.png')
