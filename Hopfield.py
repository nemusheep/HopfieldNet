import numpy as np
import matplotlib.pyplot as plt
from data import *

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


def associate(d, p, Q, model, plot=False):
    tmax = 500
    # generate noise
    noise = np.random.choice(a=[1, -1], size=(1, d**2), p=[1-p, p])
    testData = trainData*noise

    data = []
    energy = []
    data.append(testData.copy()) # append initial state
    energy.append(model.calcEnergy(testData))

    for it in range(1, 1+tmax):
        newTestData = model.update(testData)
        newEnergy = model.calcEnergy(newTestData)
        if it%(tmax/5) == 0:
            data.append(newTestData.copy())
            energy.append(newEnergy.copy())

    # calculate eval value
    similarity = np.count_nonzero(data[len(data)-1] - trainData == 0, axis=1)/(d**2)
    accuracy = np.count_nonzero(similarity == 1.0)/Q

    # plot about each test data
    if plot:
        fig = plt.figure(figsize=(14, 4))
        for q in range(Q):
            for i in range(len(data)):
                plotData = np.reshape(data[i][q], (d, d))
                plt.subplot(1, len(data), i+1)
                plt.imshow(plotData, cmap='Greys')
                plt.xticks([])
                plt.yticks([])
                E = energy[i]
                if Q > 1:
                    E = energy[i][q]
                plt.title(f't = {(tmax/5) * i}, E = {E :.1f}')

            plt.savefig(f'result/fig{q}_d{d}_p{p}.png')
        plt.close(fig)
    
    return similarity, accuracy

if __name__ == '__main__':
    it = 1000
    d = 5
    data = lines

    haxis = np.linspace(0.0, 1.0, 11)
    fig1 = plt.figure()
    fig2 = plt.figure()

    for q in [1, 3, 5]:

        simList = []
        accList = []

        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            # train model
            trainData = np.array(data[:q])
            trainData = np.reshape(trainData, (trainData.shape[0], d**2))
            hn = Hopfield(dataSize=d)
            hn.fit(trainData)

            # associate
            similarity = np.zeros(q)
            accuracy = 0
            for _ in range(it):
                sim_i, acc_i = associate(d, p, q, model=hn)
                similarity += sim_i
                accuracy += acc_i
            similarity = np.mean(similarity/it)
            accuracy /= it
            print(similarity, accuracy)
            simList.append(similarity)
            accList.append(accuracy)
        
        plt.figure(fig1.number)
        plt.plot(haxis, simList, label=f'num_{q}')
        plt.figure(fig2.number)
        plt.plot(haxis, accList, label=f'num_{q}')
    
    plt.figure(fig1.number)
    plt.xlabel('noise prob')
    plt.ylabel('similarity')
    plt.legend()
    plt.savefig('fig/sim.png')
    plt.figure(fig2.number)
    plt.xlabel('noise prob')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('fig/acc.png')
