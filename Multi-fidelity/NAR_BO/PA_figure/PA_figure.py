import numpy as np
import pickle
import matplotlib.pyplot as plt

for i in range(4,5):
    with open('PA'+str(i)+'.pickle','rb') as f:
        dataset = pickle.load(f)
        
    x = dataset['x'][i]

    plt.plot(x, dataset['low_y'][0], color='gray', label='low-fidelity')
    plt.plot(x, dataset['high_y'][0], color='black', label='high-fidelity')
    #plt.xticks(fontsize=8)
    #plt.xticks(fontsize=8)
    plt.legend(fontsize=15)
    plt.show()
    '''
    for j in range(3):
        plot(x, dataset['low_y'][j])
        plot(x, dataset['high_y'][j])
        show()
    '''

