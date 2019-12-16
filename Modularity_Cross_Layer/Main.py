#!/usr/bin/python
# -*- coding: utf-8 -*-


import Test_set  # The method of import local Python file


# Define the parameter class
class Parakeyward:
    """ Parameter Class """
    def __init__(self, path, q, p, num_walks, walks_length, r):
        self.path = path  # The file of path
        self.p2return = p  # the parameter of return
        self.q2return = q  # the parameter of In-Out
        self.num_walks = num_walks  # the windows size of Skip-gram
        self.walks_length = walks_length  # the length of walk, like as the sequence of word
        self.radio = r  # the jump propotation of interlayers


# The main function for code
if __name__ == "__main__":
    file = ["CS-Aarhus", "Pierreauger", "CKM", "ArXiv"]
    datasets_len = len(file)
    for i in range(0, datasets_len):
        for j in range(5):
            for f in range(1,10):
                args = Parakeyward("pickle/"+file[i], 1, 2, 5, 20, 0.2)
                print("-----------%s:%s------------"%(file[i], str(f)))
                MN = Test_set.Mergeing_vec_N2V(args.path, args.p2return, args.q2return, args.num_walks, args.walks_length, args.radio, f*0.1)
                MN.run()