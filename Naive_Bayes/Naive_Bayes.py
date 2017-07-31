'''载入数据, 返回特征值矩阵，和相应的类别列表'''


def loadData():
    try:
        with open("Naive_Bayes.in", "r") as f:
            # 一行一行读取数据
            flines = f.readlines()
        dataSet = []
        categorys = []
        for line in flines[1:]:
            item = line.split()
            # 分类是最后一个元素，存入categorys
            categorys.append(item[-1])
            del item[0], item[-1]
            dataSet.append(item)
        return dataSet, categorys
    except:
        print("数据载入出错！")


def loadPredictSet():
    return [["sunny", "cool", "high", "strong"]]


'''训练朴素贝叶斯分类器'''


def trainNB(dataSet, categorys):
    model = {}
    # 将data分成（index，item）后遍历
    for index, item in enumerate(dataSet):
        # 获取当前的categorys
        category = categorys[index]
        if category not in model:
            model[category] = {}
            model[category][category] = 0
        model[category][category] += 1
        for featureWord in item:
            if featureWord not in model[category]:
                model[category][featureWord] = 0
            model[category][featureWord] += 1
    return model, len(dataSet)


def predictNB(predictSet, model, trainSize):
    for item in predictSet:
        bestRate = 0.0
        bestCategory = ""
        # 与数据中的频率进行比较
        for currentCategory in model.keys():
            currentRate = model[currentCategory][currentCategory] / trainSize
            for featureWord in item:
                currentRate *= model[currentCategory][featureWord] / model[currentCategory][currentCategory]

            print(str(item) + " --> " + currentCategory + " rate = " + str(currentRate))

            if currentRate > bestRate:
                bestRate = currentRate
                bestCategory = currentCategory

        print(str(item) + " --->>>> " + bestCategory)


def main():
    dataSet, categorys = loadData()

    print("训练样本:")
    for index, item in enumerate(dataSet):
        print(str(item) + " -> " + str(categorys[index]))
    print("***************************************")
    print("预测数据:")

    model, trainSize = trainNB(dataSet, categorys)

    predictSet = loadPredictSet()
    predictNB(predictSet, model, trainSize)


if __name__ == "__main__":
    main()
