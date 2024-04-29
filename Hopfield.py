import numpy as np
import matplotlib.pyplot as plt

class Hopfield:

    def __init__(self, dataSize):
        self.W = None
        self.dataSize = dataSize
        self.thres = np.zeros(dataSize*dataSize)

    def fit(self, trainData):
        self.W = np.dot(trainData.T, trainData)/trainData.shape[0]
        np.fill_diagonal(self.W, 0)
        return self.W
    
    def update(self, testData):
        randind = np.random.randint(self.dataSize**2)
        testData[:, randind] = np.sign(np.dot(self.W[randind], testData.T) - self.thres[randind])
        return testData
    
    def calcEnergy(self, testData):
        energy = -0.5*np.dot(testData, np.dot(self.W, testData.T)) + np.dot(self.thres, testData.T)
        return np.squeeze(np.diag(energy))
    
    def plot_show(self, testData):
        plotData = np.reshape(testData, (self.dataSize, self.dataSize))
        plt.imshow(plotData, cmap='Greys')
        plt.show()

d = 5 # data size of one side
p = 0.1 # noise probability

# train data static defined
trainData = np.array([[[1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1],
                      [-1, -1, 1, -1, -1],
                      [-1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, 1]],
                      [[-1, -1, 1, -1, -1],
                      [-1, -1, 1, -1, -1],
                      [-1, -1, 1, -1, -1],
                      [-1, -1, 1, -1, -1],
                      [-1, -1, 1, -1, -1]]
                      ])
trainData = np.reshape(trainData, (trainData.shape[0], 25))

# testData = np.random.randint(2, size=(5, 5))
# testData = testData*2 - 1

# generate noise
noise = np.random.choice(a=[1, -1], size=(1, d**2), p=[1-p, p])
testData = trainData*noise
print(f'testdata {testData.shape}')

hn = Hopfield(dataSize=d)
hn.fit(trainData)
print(f'self.W {hn.W.shape}, self.thres {hn.thres.shape}')

data = []
energy = []
data.append(testData.copy()) # append initial state
energy.append(hn.calcEnergy(testData))

for it in range(1, 101):
    newTestData = hn.update(testData)
    newEnergy = hn.calcEnergy(newTestData)
    if it%20 == 0:
        data.append(newTestData.copy())
        energy.append(newEnergy.copy())

fig = plt.figure(figsize=(14, 4))

# plot about each test data
for q in range(2):
    for i in range(len(data)):
        plotData = np.reshape(data[i][q], (d, d))
        plt.subplot(1, len(data), i+1)
        plt.imshow(plotData, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.title(f't = {20 * i}, E = {energy[i][q]}')

    plt.savefig(f'result/fig_pattern{q}.png')
# plt.show()