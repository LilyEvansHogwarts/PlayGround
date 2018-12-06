import numpy as np
import pickle
import matplotlib.pyplot as plt

for i in range(4,5):
    with open('PA'+str(i)+'.pickle','rb') as f:
        dataset = pickle.load(f)

    print(dataset['x'][0,0])
    print(dataset['x'][1,0])
    print(dataset['x'][2,0])
    print(dataset['x'][3,0])
        
    x = dataset['x'][i]

    plt.plot(x, dataset['low_y'][0], label='exact low fidelity', linewidth=2, color='m')#'darkgray')#'lightslategray')
    plt.plot(x, dataset['high_y'][0], label='exact high fidelity', linewidth=2, color='black')
    #plt.xticks(fontsize=8)
    #plt.xticks(fontsize=8)
    plt.xlabel('Vb(V)', {'size':15})
    plt.ylabel('Eff(%)', {'size':15})
    plt.legend(fontsize=12)
    plt.show()
    '''
    for j in range(3):
        plot(x, dataset['low_y'][j])
        plot(x, dataset['high_y'][j])
        show()
    '''

