from math import log
from collections import Counter
from collections import defaultdict


# This function returns list of counters: documents. Each counter is a like-dictionary item that count times each word appears in the i'th document.
# It also returns list of sets: true_labels. Each set contain the topics of the i'th document.
def parse_file(filename):
    documents = []  # each place at the list is a dictionary of words and their frequency in the relevant document (implement as counter)
    true_labels = []    # each place at the list is a set of all true labels (topics) of the relevant document
    exclude = '<TRAIN'
    with open(filename) as fd:
        for line in fd:
            if exclude in line:
                topics = line.split()[2:]
                topics[-1] = topics[-1][:-1]
                true_labels.append(set(topics))
            elif line != '\n' and exclude not in line:
                cnt = Counter()
                for word in line.split():
                    cnt[word] += 1
                documents.append(cnt)
    return documents, true_labels


def em_initialization():
    pass


def e_step():
    pass


def m_step():
    pass


# Using the parameters (prob_clus and prob_word_clus) this function returns the likelihood of the observed documents
def likelihood():
    pass


# Using the hard_assignment and the documents_true_labels calculate the confusion matrix
def get_confusion_matrix():
    pass



if __name__ == '__main__':
    dev_filename = 'develop.txt'
    clusters_num = 9
    threshold = ...
    likelihood_list = []    # The likelihood of the observed documents after each iteration of em algorithm.
    perplexity_list = []    # The perplexity of the observed documents after each iteration of em algorithm.

    documents_words_counters, documents_true_labels = parse_file(dev_filename)
    total_num_of_words_in_documents = sum([sum(cnt_values) for cnt_values in [cnt.values() for cnt in documents_words_counters]])

    weights_doc_clus = [dict.fromkeys(range(clusters_num), 0) for i in range(len(documents_words_counters))]    # w_ti : probability that document t belongs to cluster i
    prob_clus = dict.fromkeys(range(clusters_num), 0)   # P(C_i) : aprior probability that some document belongs to cluster i
    prob_word_clus = defaultdict(dict)  # P_ik: probability for word k given cluster i

    # Initialize weights and parameters
    em_initialization()

    delta = float("inf")    # The difference between the current likelihood and the previous one. Algorithm will stop when delta < threshold
    last_likelihood = 0

    while delta >= threshold:
        e_step()
        m_step()
        current_likelihood = likelihood()
        likelihood_list.append(current_likelihood)
        perplexity_list.append(2**((-1/total_num_of_words_in_documents)*current_likelihood))

        delta = current_likelihood-last_likelihood
        last_likelihood = current_likelihood

    # End of em algorithm

    hard_assignment = ...   # list: each document assigned to its prefferd cluster.

    confusion_matrix = get_confusion_matrix()



