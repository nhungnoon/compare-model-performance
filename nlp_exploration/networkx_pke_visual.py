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


import pke
import string
import networkx as nx
stoplist = list(string.punctuation)

def graph_word_connection(input_text):
   extractor = pke.unsupervised.TextRank()
   text = nltk.word_tokenize(input_text)
   with_tag_text = nltk.pos_tag(text)

  # create list of part of speech to be annotated
  pos = {'CC', 'JJ', 'NNP'}
  # load the document into the TextRank model
  extractor.load_document(input = [with_tag_text],
    language='en',
    normalization=None)
  extractor.candidate_weighting(window=5,
                              pos=pos,
                              top_percent=0.33)
  # built the graph
  extractor.build_word_graph(window=2, pos='ADJ')
  # create list of words from the graph nodes 
  k = list(extractor.graph.nodes)
  # choose only nodes from 0 to 15
  H = extractor.graph.subgraph(nodes=k[0:15])
  # create the graph
  graph_ = nx.draw(H,with_labels=True, font_weight='bold',
                 alpha=0.9, font_size=12)

  
