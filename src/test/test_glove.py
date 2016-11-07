#!/usr/bin/env python -tt

import time
from glove import Glove

def main():
    GLOVE_FILE="/home/jshetty/testTool/glovebin"
    glove = Glove()
    model_test = glove.load(GLOVE_FILE)
    print(model_test.most_similar("salesforce", number=10))

if __name__=="__main__":
    start = time.time()
    main()
    end = time.time()
    time_taken = end - start
    print("The time taken by the nlp test word2vec =" + repr(time_taken))