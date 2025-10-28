import matplotlib.pyplot as plt
import numpy as np


def plot_graph(values: list[float], x_title: str, y_title: str, plot_title: str, D: int):

    print(f"len: {len(values)}")

    # the x axis is the steps, which are the indices of the values lis
    x_axis = (np.arange(1, len(values)+1) / D) * 1
    
    plt.figure()
    plt.plot(x_axis, values)
    # plt.xscale("log")
    # plt.xlim(1e-1, 1e2)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plot_title)
    plt.grid(True)
    
    
    safe_plot_title = plot_title.replace(" ", "_").lower()
    safe_x_title = x_title.replace(" ", "_").lower()
    safe_y_title = y_title.replace(" ", "_").lower()
    
    filename = f"{safe_plot_title}_{safe_x_title}_{safe_y_title}.png"
    plt.savefig(filename)
    plt.close() 
    