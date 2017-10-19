import tensorflow as tf

wordList = ["apple", "orange", "peach"]

wp = "apple"
wc = ["orange", "peach"]

key = wordList.index(wp)
values = [wordList.index(w) for w in wc]

print key
print values
