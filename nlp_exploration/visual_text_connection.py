#### https://boudinfl.github.io/pke/build/html/unsupervised.html
#### https://networkx.github.io/documentation/networkx-1.9/examples/drawing/labels_and_colors.html
#### @InProceedings{boudin:2016:COLINGDEMO,
####  author    = {Boudin, Florian},
####  title     = {pke: an open source python-based keyphrase extraction toolkit},
####  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational ####Linguistics: System Demonstrations},
####  month     = {December},
####  year      = {2016},
####  address   = {Osaka, Japan},
####  pages     = {69--73},
####  url       = {http://aclweb.org/anthology/C16-2015}
#### }
#### https://radimrehurek.com/gensim/install.html
#### https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html

import pke
import networkx as nx
import nltk
import gensim
from gensim import corpora


def gensim_topic_model(input_token_list):
    """
    Create a topic model
    """
    # Create dictionary based on the list of tokens
    dictionary = corpora.Dictionary(d.split() for d in input_token_list)
    # create word_matrix to input into the model
    list_2 = [d.split() for d in input_token_list]
    word_matrix = [dictionary.doc2bow(doc) for doc in list_2]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(word_matrix, num_topics=5, id2word=dictionary, passes=5)
    return ldamodel


def graph_word_connection(input_text):
    """
    format input text and plot connection for chosen part of speech
    """
    extractor = pke.unsupervised.TextRank()
    text = nltk.word_tokenize(input_text)
    with_tag_text = nltk.pos_tag(text)

    # create list of part of speech to be annotated
    pos = {"CC", "JJ", "NNP"}
    # load the document into the TextRank model
    extractor.load_document(input=[with_tag_text], language="en", normalization=None)
    extractor.candidate_weighting(window=5, pos=pos, top_percent=0.33)
    # built the graph
    extractor.build_word_graph(window=2, pos="ADJ")
    # create list of words from the graph nodes
    k = list(extractor.graph.nodes)
    # choose only nodes from 0 to 15
    H = extractor.graph.subgraph(nodes=k[0:15])
    # create the graph
    graph_ = nx.draw(H, with_labels=True, font_weight="bold", alpha=0.7, font_size=12)

    return graph_
