import numpy as np
from sklearn import metrics
import h5py

import sys
# seed_number = "0"
seed_number = sys.argv[1]

results = []
for index in range(10):
    name = "IP_" + str(index) + ".npy"
    results.append(np.load(name))

results = np.array(results).transpose(1,0)
print(results.shape)

final_results = []
for index in range(results.shape[0]):
    max_times = np.argmax(np.bincount(results[index]))
    final_results.append(max_times)

final_results = np.array(final_results)
print(final_results.shape, np.unique(final_results))

f = h5py.File(r'./data/IP_10_10249_28_28_3.h5', 'r')  # 路径
labels = f['label'][:]  # (42776, )
f.close()

matrix = metrics.confusion_matrix(labels, final_results)
print(matrix)
OA = np.sum(np.trace(matrix)) / 10249.0 * 100.0
print('OA = ', round(OA, 2))

f = open('meta_learning/IP_prediction_' + str(round(OA, 2)) + '_finalresults_' + seed_number + '.txt', 'w')
for i in range(final_results.shape[0]):
    f.write(str(final_results[i]) + '\n')