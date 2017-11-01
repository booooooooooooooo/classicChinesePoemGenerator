#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

ENCODE = 'utf-8'
corpusdir = "./data/corpus_std_poem_all_from_rnnpg_data_emnlp-2014/"

corpus = []
for filePath in os.listdir(corpusdir):
    fin = open(corpusdir + filePath)
    charList = list(fin.read().decode(ENCODE))
    charList = charList[0 : int(proportion * len(charList) )]
    corpus += charList

for i in xrange(3):
    fin = open("./data/corpus_std_poem_all_from_rnnpg_data_emnlp-2014/" + flist[i])
    fout = open("./q5/" + flist[i], 'w')
    lines = fin.read().decode(ENCODE).split()
    for line in lines:
        cleanLine = ''.join( line.split() )
        if len(cleanLine) == 20:
            fout.write(cleanLine.encode(ENCODE) + '\n')
    fout.close()
