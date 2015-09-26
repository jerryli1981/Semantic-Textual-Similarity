# - an individual node contains the word associated with the node along with 
#   pointers to its kids and parents. 
class node:

    def __init__(self, word):
        if word != None:
            self.word = word
            self.kids = []
            self.parent = []
            self.finished = 0
            self.is_word = 1

            # the "ind" variable stores the look-up index of the word in the 
            # word embedding matrix We. set this value when the vocabulary is finalized
            self.ind = -1

        else:
            self.is_word = 0

class dtree:

    def __init__(self, word_list):
        self.nodes = []
        for word in word_list:
            self.nodes.append(node(word))

        self.label = -1


    def add_edge(self, par, child, rel):
        self.nodes[par].kids.append( (child, rel ) )
        self.nodes[child].parent.append( (par, rel) )


    # return all non-None nodes
    def get_nodes(self):
        return [node for node in self.nodes if node.is_word]


    def get_node_inds(self):
        return [(ind, node) for ind, node in enumerate(self.nodes) if node.is_word]


    # get a node from the raw node list
    def get(self, ind):
        return self.nodes[ind]


    # return the raw text of the sentence
    def get_words(self):
        return ' '.join([node.word for node in self.get_nodes()[1:]])


    # return raw text of phrase associated with the given node
    def get_phrase(self, ind):

        node = self.get(ind)
        words = [(ind, node.word), ]
        to_do = []
        for ind, rel in node.kids:
            to_do.append(self.get(ind))
            words.append((ind, self.get(ind).word))

        while to_do:
            curr = to_do.pop()

            # add this kid's kids to to_do
            if len(curr.kids) > 0:
                for ind, rel in curr.kids:
                    words.append((ind, self.get(ind).word))
                    to_do.insert(0, self.get(ind))  


        return ' '.join([word for ind, word in sorted(words, key=itemgetter(0) ) ]).strip()  


    def reset_finished(self):
        for node in self.get_nodes():
            node.finished = 0


    # one tree's error is the sum of the error at all nodes of the tree
    def error(self):
        sum = 0.0
        for node in self.get_nodes():
            sum += node.ans_error

        return sum


    # - this function is not used in the paper, but it enables corrective weighting ala
    #   Richard's 2014 TACL paper to be implemented
    # - it counts the size of the subtree rooted at every node in the tree
    def count_kids(self):
        for node in self.get_nodes():

            # all nodes include themselves
            node.count = 1.0

            # no kids, count is just 1
            if len(node.kids) == 0:
                continue

            to_do = [self.get(ind) for ind, rel in node.kids]

            # otherwise, traverse subtrees
            while to_do:
                curr = to_do.pop()
                node.count += 1

                # add this kid's kids to to_do
                if len(curr.kids) > 0:
                    for ind, rel in curr.kids:
                        to_do.append(self.get(ind))


# - function that parses the resulting stanford parses
#   e.g., "nsubj(finalized-5, john-1)"
def split_relation(text):
    rel_split = text.split('(')
    rel = rel_split[0]
    deps = rel_split[1][:-1]
    if len(rel_split) != 2:
        print 'error ', rel_split
        sys.exit(0)

    else:
        dep_split = deps.split(',')

        # more than one comma (e.g. 75,000-19)
        if len(dep_split) > 2:

            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

            print 'fixed: ', fixed
            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )

        return rel, final_deps

# - given a list of all the split relations in a particular sentence,
#   create a dtree object from that list
def make_tree(plist):

    # identify number of tokens
    max_ind = -1
    for rel, deps in plist:
        for ind, word in deps:
            if ind > max_ind:
                max_ind = ind

    # load words into nodes, then make a dependency tree
    nodes = [None for i in range(0, max_ind + 1)]
    for rel, deps in plist:
        for ind, word in deps:
            nodes[ind] = word

    tree = dtree(nodes)

    # add dependency edges between nodes
    for rel, deps in plist:
        par_ind, par_word = deps[0]
        kid_ind, kid_word = deps[1]
        tree.add_edge(par_ind, kid_ind, rel)

    return tree  
       