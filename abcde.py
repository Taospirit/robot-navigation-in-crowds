
import numpy as np
import random
import csv
import os.path


def log_results(path_length):
    # Save the results to a file so we can graph it later.
    with open('results/logs/loss_data-' + 'abcde' + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        wr.writerows(path_length)

if __name__ == "__main__":
    path_length = []
    for i in range(10):
        path_length.append([i,i**2])
    log_results(path_length)

