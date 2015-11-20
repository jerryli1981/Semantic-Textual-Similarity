import networkx as nx
import copy
import numpy as np

class Relation:
    def __init__(self, mention, index):
        self.mention = mention
        self.index = index

class Node:

    def __init__(self, word, index):
        self.word = word
        self.kids = []
        self.index = index
        self.finished = False

class DTree:

    def __init__(self, toks, parents, labels,vocab,rel_vocab):

        deps = []
        for i, (govIdx, rel) in enumerate(zip(parents, labels)):
            deps.append((rel, int(govIdx)-1, i))

        self.nodes = []
        for tok in toks:
            self.nodes.append(Node(tok, vocab[tok])) 

        # add dependency edges between nodes
        rootIdx = None
        self.dependencies = []
        for rel, govIdx, depIdx in deps:
            if govIdx == -1:
                rootIdx = depIdx
                continue
            self.nodes[govIdx].kids.append((depIdx, Relation(rel,rel_vocab[rel])))
            self.dependencies.append((govIdx, depIdx))

        self.root = self.nodes[rootIdx]
        self.rootIdx = rootIdx

        G = nx.DiGraph()
        G.add_edges_from(self.dependencies)

        self.is_dag = nx.is_directed_acyclic_graph(G)

    def resetFinished(self):
        for node in self.nodes:
            node.finished = False