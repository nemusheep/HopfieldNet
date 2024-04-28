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
        testData[0][randind] = np.sign(np.dot(self.W[randind], testData[0]) - self.thres[randind])
        return testData
    
    def calcEnergy(self, testData):
        energy = -0.5*np.dot(testData, np.dot(self.W, testData.T)) + np.dot(self.thres, testData[0])
        return np.squeeze(energy)
    
    def plot_show(self, testData):
        plotData = np.reshape(testData, (self.dataSize, self.dataSize))
        plt.imshow(plotData, cmap='Greys')
        plt.show()


trainData = np.array([[1, -1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, -1, 1, -1, -1], [-1, -1, -1, 1, -1], [-1, -1, -1, -1, 1]])
trainData = np.reshape(trainData, (1, 25))
testData = np.random.randint(2, size=(5, 5))
testData = testData*2 - 1
testData = np.reshape(testData, (1, 25))

hn = Hopfield(dataSize=5)
hn.fit(trainData)
data = []
energy = []

for it in range(101):
    newTestData = hn.update(testData)
    e = hn.calcEnergy(newTestData)
    print(e)
    if it%20 == 0:
        datai = newTestData.copy()
        energyi = e.copy()
        data.append(datai)
        energy.append(energyi)

fig = plt.figure(figsize=(14, 4))

for i in range(len(data)):
    plotData = np.reshape(data[i], (5, 5))
    plt.subplot(1, len(data), i+1)
    plt.imshow(plotData, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.title(f't = {20 * i}, E = {energy[i]}')

plt.savefig('fig/figure1.png')
plt.show()