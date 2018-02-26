import itertools
import pandas as pd

#perceptron
lst = list(itertools.product([0, 1], repeat=3))
perceptron_df = pd.DataFrame(lst)

w = pd.DataFrame([[.6, -.7, .4],[ .5, -.6, .8]]).transpose()
b = [-.4, -.5]

layer2 = np.dot(perceptron_df, w) + b
perceptron_df['output'] = np.sum(layer2, 1) + .5


# sigmoid network
sigmoid_df = perceptron_df.iloc[:,0:3]

z1 = np.dot(sigmoid_df, w) + b
a1 = 1/(1 + np.exp(-z1))

z2 = np.sum(a1, 1) - .5

sigmoid_df['output'] = 1/(1 + np.exp(-z2))
