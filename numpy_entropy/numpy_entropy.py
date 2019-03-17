# authors:
# Jan Pavovský : 618792cf-25ec-11e8-9de3-00505601122b
# Jaroslav Knotek : 

#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line

    dataPointList = {}
    modelList={}

    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            if(line in dataPointList):
                dataPointList[line] += 1
            else:
                dataPointList[line] = 1
                modelList[line] = 0.

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.

    # Load model distribution, each line `word \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            tokens = line.split("\t")
            if(tokens[0] in modelList):
                modelList[tokens[0]] = float(tokens[1])

    data = np.array(list(dataPointList.values()))
    dist = data / np.sum(data)

    model = np.array(list(modelList.values()))

    # TODO: Create a NumPy array containing the model distribution.

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).

    entropy=-(np.sum(np.log(dist) * dist))

    print("{:.2f}".format(entropy))

    cross_entropy=-(np.sum(np.log(model) * dist))

    print("{:.2f}".format(cross_entropy))
    
    kl= cross_entropy - entropy

    print("{:.2f}".format(kl))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)

