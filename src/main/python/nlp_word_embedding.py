#!/usr/bin/env python -tt
from __future__ import unicode_literals, print_function

import os
import sys
import time
import logging
import multiprocessing
import argparse
from gensim.models.word2vec import Word2Vec
from glove import Corpus, Glove


class Sentences(object):
    """
    This class supports sentence formation as preprocessor for word2vec and glove word embeddings.
    """
    def __init__(self, dirname):
        """
        __init__(self, dirname)

        """
        self.dirname = dirname

    def __iter__(self):
        """
        :return: Iterator of sentence
        """
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


def enable_logging(logger, logfile):
    """
    Logging module
    """
    handlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handlr.setFormatter(formatter)
    logger.addHandler(handlr)
    logger.setLevel(logging.WARNING)


def word2vec_embedding(corpus_dir, output_file, output_format, embed_size, window, min_count):
    """
    Generate word2vec vectors and save the output file to the given directory
    :param corpus_dir: string
    :param output_file: string
    :param output_format: bin/text
    :return: None but saves the file in given file format
    """
    sentences = Sentences(corpus_dir)
    model = Word2Vec(size=embed_size, window=window, min_count=min_count, workers=multiprocessing.cpu_count())
    #  Build vocab
    model.build_vocab(sentences)
    #  Train
    model.train(sentences)
    #  Finished training the model
    model.init_sims(replace=True)
    if(output_format == "binary"):
        model.save_word2vec_format(output_file, binary=True)
    elif(output_format == "text"):
        # Save in text format
        model.save_word2vec_format(output_file, binary=False)


def glove_embedding(corpus_dir, output_file, output_format, embed_size, window, learning_rate, epochs, threads):
    """
    Generate glove vectors and save the output file to the given directory
    :param corpus_dir: string
    :param output_file: string
    :param output_format: bin/text
    :param embed_size: int
    :param window: int
    :param learning_rate: float
    :param epochs: int
    :param threads: int
    :return: None but saves the file in given file format
    """
    sentences = Sentences(corpus_dir)
    corpus = Corpus()
    corpus.fit(sentences, window)
    # components: latent dimensions
    glove = Glove(no_components=embed_size, learning_rate=learning_rate)
    glove.fit(corpus.matrix, epochs=epochs, no_threads=threads, verbose=True)
    # Supply a word-id dictionary to allow similarity queries.
    glove.add_dictionary(corpus.dictionary)
    glove.save(output_file)
    if(output_format == "binary"):
        glove.save(output_file)
    elif(output_format == "text"):
        # glove.save(output_file, binary=False)
        pass


def get_args():
    """This function parses and return arguments passed in"""
    parser = argparse.ArgumentParser(description="Word Embedding Tool")
    # Add arguments
    parser.add_argument('-e', '--embed', type=str, help='Embedding Type', required=True)
    parser.add_argument('-c', '--corpus', type=str, help='Corpus Directory', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output File', required=True)
    parser.add_argument('-of', '--outputformat', type=str, help='Output Format', required=True)
    parser.add_argument('-w', '--window', type=str, help='Window Size', required=False, default=5)
    parser.add_argument('-es', '--embedsize', type=int, help='Embedding Size', required=False, default=300)
    parser.add_argument('-mc', '--mincount', type=int, help='Minimum Count', required=False, default=3)
    # Glove arguments
    parser.add_argument('-ep', '--epochs', type=int, help='Number of Epochs', required=False, default=10)
    parser.add_argument('-lr', '--learningrate', type=float, help='Learning Rate', required=False, default=0.05)
    parser.add_argument('-t', '--threads', type=int, help='Number of Threads', required=False, default=4)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    embedding_type = args.embed
    corpus_dir = args.corpus
    output_file = args.output
    output_format = args.outputformat
    window = args.window
    embed_size = args.embedsize
    min_count = args.mincount
    learning_rate = args.learningrate
    threads = args.threads
    epochs = args.epochs
    return embedding_type, corpus_dir, output_file, output_format, window, embed_size, min_count, learning_rate, threads, epochs


def main():
    """
    Main method
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger("NLP Word Embeddings")
    logfile = "nlp_word_embedding.log"
    enable_logging(logger, logfile)
    # Check command line arguments to be 4 including script name
    if len(sys.argv) < 4:
        print("Usage: nlp_embedding.py <embedding_type> <corpus_dir> <output_file> <output_format> <optional:embed_size> <optional:window> <optional:min_count>")
        print("Example: nlp_embedding.py -e word2vec -c /data/deeplearning/corpus -o /data/testTools/output_file -of binary -es 300 -w 5 -mc 3")
        sys.exit()
    else:
        # Call get_args
        embedding_type, corpus_dir, output_file, output_format, window, embed_size, min_count, learning_rate, threads, epochs = get_args()
        if(embedding_type == "word2vec"):
            word2vec_embedding(corpus_dir, output_file, output_format, embed_size, window, min_count)
        elif(embedding_type == "glove"):
            glove_embedding(corpus_dir, output_file, output_format, embed_size, window, learning_rate, epochs, threads)
        else:
            print("Usage:" + repr(embedding_type) + "embeddings is not supported")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    time_taken = end - start
    print("The time taken by the NLP-Embedding Tool = " + repr(time_taken) + " seconds")