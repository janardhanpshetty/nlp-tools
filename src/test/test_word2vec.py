#!/usr/bin/env python -tt

import time
from gensim.models.word2vec import Word2Vec

def main():
    WORD2VEC_FILE = "/home/jshetty/testTool/word2vecText"
    model_test = Word2Vec.load_word2vec_format(WORD2VEC_FILE)
    print("Model doesn't match " + model_test.doesnt_match("crm salesforce content".split()))
    print(model_test.similarity("SFDC", "salesforce"))
    print(model_test.most_similar("salesforce"))

if __name__=="__main__":
    start = time.time()
    main()
    end = time.time()
    time_taken = end - start
    print("The time taken by the nlp test word2vec =" + repr(time_taken))