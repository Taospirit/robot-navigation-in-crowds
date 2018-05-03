"""
Take the data in the results folder and plot it so we can stop using stupid
Excel.
"""

import glob
import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def movingaverage(y, window_size):
    """
    Moving average function from:
    http://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(y, window, 'same')


def readable_output(filename):
    readable = ''
    # Example:
    # path_data-128-128-100-10000-9.csv
    f_parts = filename.split('-')

    if f_parts[0] == 'path_data':
        readable += 'distance: '
    else:
        readable += 'loss: '

    readable += f_parts[1] + ', ' + f_parts[2] + ' | '
    readable += f_parts[3] + ' | '
    readable += f_parts[4].split('.')[0]

    return readable


def plot_file(filename, type='loss'):
    with open(f, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Turn our column into an array.
        y = []
        x = []
        for row in reader:
            if type == 'loss':
                y.append(float(row[0]))
            else:
                x.append(float(row[0]))
                y.append(float(row[1]))

        # Running tests will be empty.
        if len(y) == 0:
            return

        print(readable_output(f))

        # Get the moving average so the graph isn't so crazy.
        if type == 'loss':
            window = 100
            y_av = movingaverage(y, window)
        else:
            y_av = y

        # Use our moving average to get some metrics.
        arr = np.array(y_av)
        if type == 'loss':
            print("%f\t%f\n" % (arr.min(), arr.mean()))
        else:
            print("%f\t%f\n" % (arr.max(), arr.mean()))

        # Plot it.
        plt.clf()  # Clear.
        plt.title(f)
        # The -50 removes an artificial drop at the end caused by the moving
        # average.
        if type == 'loss':
            plt.plot(y_av)
            plt.ylabel('Loss')
            plt.xlabel('Num of Frames')
            plt.ylim(0, 100000)
          
        else:
            plt.plot(x, y_av)
            plt.ylabel('Path Length')
            plt.xlabel('Episode')
            plt.ylim(0, 4000)

        plt.savefig(f + '.png', bbox_inches='tight')


if __name__ == "__main__":
    # Get our loss result files.
    os.chdir("results/logs-0")

    for f in glob.glob("path*.csv"):
        plot_file(f, 'path')

    for f in glob.glob("loss*.csv"):
        plot_file(f, 'loss')
