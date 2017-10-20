#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

ENCODE = 'utf-8'

flist = ['qtrain', 'qvalid', 'qtest']
for i in xrange(3):
    fin = open("./rnnpg_data_emnlp-2014/partitions_in_Table_2/rnnpg/" + flist[i])
    fout = open("./q5/" + flist[i], 'w')
    lines = fin.read().decode(ENCODE).split()
    for line in lines:
        cleanLine = ''.join( line.split() )
        if len(cleanLine) == 20:
            fout.write(cleanLine.encode(ENCODE) + '\n')
    fout.close()
