import numpy as np
fileToSaveWordVectors = "./data/word2Vec"

fin = open(fileToSaveWordVectors, "a")
np.save(fin, [1,2,3])
fin.close()

fin = open(fileToSaveWordVectors)
a = np.load(fin)
print a
