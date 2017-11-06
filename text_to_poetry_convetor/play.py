#encoding=utf-8

from utilGeneral import getCosineSimilarities
from utilData import UtilData
import numpy as np

vocabularyDic = UtilData().prepareVocabularyDic()

fin = open("./output/word2Vec")
wordFeatureVectors = np.load(fin)
fin.close()

similarities = getCosineSimilarities(wordFeatureVectors, vocabularyDic)

for i in xrange(len(similarities)):
    if similarities[i][2] < 0.9 and similarities[i][2] > 0.8:
        print similarities[i][0].encode('utf-8'), similarities[i][1].encode('utf-8'), similarities[i][2]
