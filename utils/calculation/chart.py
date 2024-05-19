
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
def Get_histogram(dataset, column_name):
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.title = f'histogram of {column_name}'
    ax.hist(dataset[column_name], bins=50)
    return fig, ax


def Get_Density_Destributation(dataset, column_name):
    plt.style.use('ggplot')
    data = dataset[column_name]
    density = gaussian_kde(data)
    x_axs = np.linspace(data.min()-10, data.max()+10, 100)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    fig, ax = plt.subplots()
    ax.plot(x_axs, density(x_axs))
    fig.title = f'Desnity of {column_name}'

    return fig, ax


def Get_Box_Plot(dataset, column_name):
    plt.style.use('ggplot')
    data = dataset[column_name]
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize =(10, 7))

    plt.boxplot(data)
    #plot.boxplot(data)
    return fig

