import numpy as np

def get_minibatches(data, minibatch_size, shuffle=True):
    minibatches = []

    inputs, labels = data
    indices = np.arange(len(inputs))
    # print indices
    if shuffle:
        np.random.shuffle(indices)
    # print indices
    for i in range(0, len(indices), minibatch_size):
        inputsBatch = [inputs[indices[j]] for j in range(i, min(i + minibatch_size, len(indices)))]
        labelsBatch = [labels[indices[j]] for j in range(i, min(i + minibatch_size, len(indices)))]
        minibatches.append((inputsBatch, labelsBatch))
    return minibatches

def sanity_get_minibatches():
    inputs = [[1,2,3], [4,5,6], [7,8,9],[1,2,3], [4,5,6], [7,8,9],[1,2,3], [4,5,6], [7,8,9]]
    labels = [1,2,3,4,5,6,7,8,9]
    minibatches = get_minibatches((inputs, labels), 3)
    print minibatches[0]

if __name__ == "__main__":
    sanity_get_minibatches()
