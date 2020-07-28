from kNN import *
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # kNN calc distanse
    group, labels = createDataSet()
    print(group)
    print(labels)
    ret = classify0([0, 0], group, labels, 3)
    print(ret)

    # calc dating
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingDataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0* array(datingLabels))
    #plt.show()

    # autonorm
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(normMat)

    # dating test
    #datingClassTest()
    #classifyPerson()

    # handwriting 
    handwritingClassTest()
