#encoding=utf-8

from utilGeneral import getCosineSimilarities
from utilData import UtilData

utilData = UtilData()
similarities = getCosineSimilarities("./wordVec/wordVecSkipGram", utilData.getMostFrequentChars(500), utilData.prepareVocabularyDic())
for sim in similarities:
    print sim[0].encode('utf-8'),
    print sim[1].encode('utf-8'),
    print sim[2]

# chars = utilData.getMostFrequentChars(1000)
# for c in chars:
#     print c,
# print chars
