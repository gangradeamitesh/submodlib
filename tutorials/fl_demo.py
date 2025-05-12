import numpy as np
groundData =np.array( [(3,12.5), (4.5,13.5), (5,13.5), (5.5,13.5), (14.5,13.5), (15,13.5), (15.5,13.5),
(4.5,13), (5,13), (5.5,13), (14.5,13), (15,13), (15.5,13),
(4.5,12.5), (5,12.5), (5.5,12.5), (14.5,12.5), (15,12.5), (15.5,12.5),
(4.5,7.5), (5,7.5), (5.5,7.5), (14.5,7.5), (15,7.5), (15.5,7.5),
(4.5,7), (5,7), (5.5,7), (14.5,7), (15,7), (15.5,7),
(4.5,6.5), (5,6.5), (5.5,6.5), (14.5,6.5), (15,6.5), (15.5,6.5),
(7.5,10), (12.5,10), (10,12.5), (10,7.5), (8,12.5), (8,7.5), (14,12.5), (14,7.5), (4.5, 15.5), (5,9.5), (5,10.5)] )
groundxs = [x[0] for x in groundData]
groundys = [x[1] for x in groundData]
repData =np.array( [(6.7,13.5), (7.2,13.5), (7.7,13.5), (16.7,13.5), (17.2,13.5), (17.7,13.5),
(6.7,13), (7.2,13), (7.7,13), (16.7,13), (17.2,13), (17.7,13),
(6.7,12.5), (7.2,12.5), (7.7,12.5), (16.7,12.5), (17.2,12.5), (17.7,12.5),
(6.7,7.5), (7.2,7.5), (7.7,7.5), (16.7,7.5), (17.2,7.5), (17.7,7.5),
(6.7,7), (7.2,7), (7.7,7), (16.7,7), (17.2,7), (17.7,7),
(6.7,6.5), (7.2,6.5), (7.7,6.5), (16.7,6.5), (17.2,6.5), (17.7,6.5)] )
repxs = [x[0] for x in repData]
repys = [x[1] for x in repData]


import matplotlib.pyplot as plt
#plt.scatter(groundxs, groundys, s=50, color='black', label="Images")
plt.scatter(groundxs, groundys, s=50, facecolors='none', edgecolors='black', label="Images")
plt.scatter(repxs, repys, s=50, color='green', label="Images")


from submodlib import FacilityLocationFunction
objFL = FacilityLocationFunction(n=48, data=groundData, separate_rep=True, n_rep=36, data_rep=repData, mode="dense", metric="euclidean")
greedyList = objFL.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
print(greedyList)
greedyXs = [groundxs[x[0]] for x in greedyList]
greedyYs = [groundys[x[0]] for x in greedyList]
plt.scatter(groundxs, groundys, s=50, facecolors='none', edgecolors='black', label="Images")
plt.scatter(repxs, repys, s=50, color='green', label="Images")
plt.scatter(greedyXs, greedyYs, s=50, color='blue', label="Greedy Set")
for label, element in enumerate(greedyList):
    plt.annotate(label, (groundxs[element[0]], groundys[element[0]]), (groundxs[element[0]]+0.1, groundys[element[0]]+0.1))