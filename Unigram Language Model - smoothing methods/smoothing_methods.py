import sys
from math import log
from collections import defaultdict


# This class contains arguments given to the program and some generic method for parsing and create histogram of words
class Init:
    def __init__(self):
        self.dev_filename = sys.argv[1]
        self.test_filename = sys.argv[2]
        self.input_word = sys.argv[3]
        self.output_filename = sys.argv[4]
        self.voc_size = 300000
        self.output6 = 1/self.voc_size    # uniform probability for event = input_word
        self.total_num_of_events_dev = 0

        self.dev_dict = self.parse_file(self.dev_filename, '<TRAIN')
        self.test_dict = self.parse_file(self.test_filename, '<TEST')
        self.dev_list = self.parse_file_to_list(self.dev_filename, '<TRAIN')
        self.test_list = self.parse_file_to_list(self.test_filename, '<TEST')
        self.total_num_of_events_test = sum(self.test_dict.values())

    def parse_file(self, file_name, exclude):
        events = {}
        # Reading the input file and insert new words from lines that are not
        # contain the word exclude. if the word is already in the events array,
        # the counter of the word is increased by 1.
        with open(file_name) as fd:
            for line in fd:
                if exclude not in line:
                    for word in line.split():
                        if word in events:
                            events[word] += 1
                        else:
                            events[word] = 1
        fd.close()
        return events

    def parse_file_to_list(self, file_name, exclude):
        events = []
        # Reading the input file and insert new words from lines that are not
        # contain the word exclude.
        # Also add 1 to the counter "total_num_of_events_dev" if exclude == '<TRAIN' for each word added
        with open(file_name) as fd:
            for line in fd:
                if exclude not in line:
                    for word in line.split():
                        events.append(word)
                        if exclude == '<TRAIN':
                            self.total_num_of_events_dev += 1
        fd.close()
        return events

    def create_histogram(self, my_list):
        # get a list of words (events), and return dictionary of words (events) and their number of appearance
        events = {}
        for word in my_list:
            if word in events:
                events[word] += 1
            else:
                events[word] = 1
        return events


# This class handles the Lidstone model
class Lidstone(Init):
    def __init__(self):
        Init.__init__(self)
        self.total_num_of_events_training = round(0.9 * self.total_num_of_events_dev)
        self.total_num_of_events_val = self.total_num_of_events_dev - self.total_num_of_events_training
        self.training_list = self.dev_list[:self.total_num_of_events_training]
        self.val_list = self.dev_list[self.total_num_of_events_training:]
        self.training_dict = self.create_histogram(self.training_list)
        self.val_dict = self.create_histogram(self.val_list)
        self.observed_voc_size = len(self.training_dict)
        self.input_word_count = int(self.training_dict.get(self.input_word, 0))  # number of times event input_word appears in the training set
        self.lambdas = [x / 100 for x in range(0, 201)] # the possible lambdas according to the exercise requirements.
        self.best_lambda = self.get_best_lambda()  # best lambda after perplexity test on validation set.
        self.best_perp = get_perplexity(self, self.__class__.__name__, self.best_lambda) # The perplexity acoording to the best lambda we have chosen.
        self.perplexity_test_set = get_perplexity(self, self.__class__.__name__, self.best_lambda, True)
        self.check_sum_est_eq_1 = sum(self.train(x, self.best_lambda) for x in self.training_dict) + \
                                  (self.voc_size-self.observed_voc_size) * \
                                  (self.train('no_way_this_word_appears_in_train_set_xd1234', self.best_lambda))

    # this function get an word and a lambda and return the p(event = word) as estimated by lidstone model with lambda
    def train(self, word, lam):
        word_count = int(self.training_dict.get(word, 0))
        # (count of a given word + lambda)/(size of train set + lambda*vocabulary size)
        return ((word_count + lam) /
                (self.total_num_of_events_training + (lam * self.voc_size)))

    def get_best_lambda(self):
        # returns the lambda that gets us the lower perplexity rate, among the defined lambdas.
        min_perp = sys.float_info.max
        for lam in self.lambdas:
            if lam != 0:
                new_perp = get_perplexity(self, self.__class__.__name__, lam)
                if new_perp < min_perp:
                    min_perp = new_perp
                    self.best_lambda = lam
        return self.best_lambda


# This class handles the HeldOut model
class HeldOut(Init):
    def __init__(self):
        Init.__init__(self)
        self.total_num_of_events_training = round(self.total_num_of_events_dev/2)
        self.total_num_of_events_held_out = self.total_num_of_events_dev - self.total_num_of_events_training
        self.training_list = self.dev_list[:self.total_num_of_events_training]
        self.held_out_list = self.dev_list[self.total_num_of_events_training:]
        self.training_dict = self.create_histogram(self.training_list)
        self.held_out_dict = self.create_histogram(self.held_out_list)
        self.reverse_training_dict = defaultdict(list)  # keys: frequencies, values: list of word with that frequency
        self.parameters = {}    # keys: events, values: estimated probability for the event by this model
        self.probability_for_unseen_word = 0    # calculate in 'train()'
        self.final_table = {}   # used for Output29. key: r(fmle), value: [f_h, n_r, t_r]

        self.train()
        self.p_event_input_word = self.parameters.get(self.input_word, self.probability_for_unseen_word)
        self.p_event_unseenword = self.parameters.get('unseen-word', self.probability_for_unseen_word)
        self.check_sum_est_eq_1 = self.probability_for_unseen_word*(self.voc_size-len(self.training_dict))+sum(self.parameters.values())   # this parameter should be equal 1
        self.perplexity_test_set = get_perplexity(self, self.__class__.__name__)

    def train(self):
        # calculate probabilities based on held out system and write to 'parameters'
        for event, num in self.training_dict.items():
            self.reverse_training_dict[num].append(event)
        for num, list_events in self.reverse_training_dict.items():
            freq_ho = 0     # frequency in the held-out set of all events in list_events
            for event in list_events:
                freq_ho += self.held_out_dict.get(event, 0)
            total_mass_probability_for_list_events = freq_ho/self.total_num_of_events_held_out
            est_probability_for_each_event = total_mass_probability_for_list_events/len(list_events)
            for event in list_events:
                self.parameters[event] = est_probability_for_each_event
            # fill the final_table
            if num in range(10):
                self.final_table[num] = [round(est_probability_for_each_event*self.total_num_of_events_training, 5),
                                         len(list_events), freq_ho]
        total_mass_prob_for_unseen_events = 1-sum(self.parameters.values())
        self.probability_for_unseen_word = total_mass_prob_for_unseen_events/(self.voc_size-len(self.training_dict))
        # fill final_table for r=0
        final_table_t0 = sum(self.held_out_dict[x] for x in (set(self.held_out_list)-set(self.training_list)))
        self.final_table[0] = [round(self.probability_for_unseen_word*self.total_num_of_events_training, 5),
                               self.voc_size-len(self.training_dict), final_table_t0]
        return


# This function used for multiple objectives. It can calculate the perplexity gets by specific model
# on the validation set (for choosing lambda in Lidstone model) or on the test set for both models.
def get_perplexity(model, model_name, lam=-1.0, is_testset_for_lidstone=False):
    sum_of_prob = 0
    # calculate the perplexity on validation set for choosing the best lambda in Lidstone model.
    if model_name == "Lidstone" and not is_testset_for_lidstone:
        val_list = model.val_list   # all events in the validation set (list)
        length = len(val_list)      # length of the validation set (list)
        for word in val_list:
            sum_of_prob += log(model.train(word, lam), 2)  # add to sum_of_prob the log of the probability of each word.
    elif model_name == "Lidstone" and is_testset_for_lidstone:
        test_list = model.test_list     # all events in the test set (list)
        length = len(test_list)         # length of the test set (list)
        for word in test_list:
            sum_of_prob += log(model.train(word, lam), 2)  # add to sum_of_prob the log of the probability of each word.
    elif model_name == "HeldOut":
        test_list = model.test_list
        length = len(test_list)
        for word in test_list:
            sum_of_prob += log(model.parameters.get(word, model.probability_for_unseen_word), 2)
    else:   # An unknown model
        return -1
    return 2 ** ((-1 / length) * sum_of_prob)    # The perplexity method we saw in class.


# This function get a histogram of words in the training set and an event, and return the Maximum Likelihood Estimate
# for that event consider the histogram 'my_dict'
def get_mle(my_dict, event):
    total_num_of_events_in_dict = sum(my_dict.values())
    count_event_in_dict = my_dict.get(event, 0)
    return count_event_in_dict/total_num_of_events_in_dict


# comparing both perplexity by lidstone and by heldout and return the model with lowest perplexity on test set.
def compare_perp(lidstone, heldOut):
    if lidstone.perplexity_test_set < heldOut.perplexity_test_set:
        return "L"
    return "H"


# This function write the desired output to 'file_name'
def write_output(file_name):
    with open(file_name, 'w') as fd:
        fd.write('#Students\tTidhar Suchard\tKfir Cohen\t205888209\t203485487\n')
        fd.write('#Output1\t' + my_init.dev_filename + '\n')
        fd.write('#Output2\t' + my_init.test_filename + '\n')
        fd.write('#Output3\t' + my_init.input_word + '\n')
        fd.write('#Output4\t' + my_init.output_filename + '\n')
        fd.write('#Output5\t' + str(my_init.voc_size) + '\n')
        fd.write('#Output6\t' + str(my_init.output6) + '\n')
        fd.write('#Output7\t' + str(my_init.total_num_of_events_dev) + '\n')
        fd.write('#Output8\t' + str(lidstone_model.total_num_of_events_val) + '\n')
        fd.write('#Output9\t' + str(lidstone_model.total_num_of_events_training) + '\n')
        fd.write('#Output10\t' + str(lidstone_model.observed_voc_size) + '\n')
        fd.write('#Output11\t' + str(lidstone_model.input_word_count) + '\n')
        fd.write('#Output12\t' + str(get_mle(lidstone_model.training_dict, lidstone_model.input_word)) + '\n')
        fd.write('#Output13\t' + str(get_mle(lidstone_model.training_dict, "unseen-word")) + '\n')
        fd.write('#Output14\t' + str(lidstone_model.train(my_init.input_word, 0.10)) + '\n')
        fd.write('#Output15\t' + str(lidstone_model.train("unseen-word", 0.10)) + '\n')
        fd.write('#Output16\t' + str(get_perplexity(lidstone_model, lidstone_model.__class__.__name__, 0.01)) + '\n')
        fd.write('#Output17\t' + str(get_perplexity(lidstone_model, lidstone_model.__class__.__name__, 0.10)) + '\n')
        fd.write('#Output18\t' + str(get_perplexity(lidstone_model, lidstone_model.__class__.__name__, 1.00)) + '\n')
        fd.write('#Output19\t' + str(lidstone_model.best_lambda) + '\n')
        fd.write('#Output20\t' + str(lidstone_model.best_perp) + '\n')

        fd.write('#Output21\t' + str(heldOut_model.total_num_of_events_training) + '\n')
        fd.write('#Output22\t' + str(heldOut_model.total_num_of_events_held_out) + '\n')
        fd.write('#Output23\t' + str(heldOut_model.p_event_input_word) + '\n')
        fd.write('#Output24\t' + str(heldOut_model.p_event_unseenword) + '\n')
        fd.write('#Output25\t' + str(my_init.total_num_of_events_test) + '\n')

        fd.write('#Output26\t' + str(lidstone_model.perplexity_test_set) + '\n')
        fd.write('#Output27\t' + str(heldOut_model.perplexity_test_set) + '\n')
        fd.write('#Output28\t' + str(compare_perp(lidstone_model, heldOut_model)) + '\n')

        # output 29:
        # handling case where there isn't an event that happened r times in the training corpus
        null_list = ['Null', 'Null', 'Null']
        fd.write('#Output29\n')
        for r in range(10):
            # flambda is the expected frequency of a word that happened r times in the training corpus according
            # to the estimated probability by lidstone model with best lambda found.
            # it is calculated: size_of_training_corpus * discount_probability_for_event_that_happend_r_times_originally
            flambda = lidstone_model.total_num_of_events_training * \
                      ((r+lidstone_model.best_lambda) /
                       (lidstone_model.total_num_of_events_training+lidstone_model.best_lambda*lidstone_model.voc_size))
            flambda = round(flambda, 5)
            fd.write(str(r) + '\t' + str(flambda) + '\t' + str(heldOut_model.final_table.get(r, null_list)[0]) + '\t'
                     + str(heldOut_model.final_table.get(r, null_list)[1]) + '\t'
                     + str(heldOut_model.final_table.get(r, null_list)[2]) + '\n')


if __name__ == '__main__':
    my_init = Init()
    lidstone_model = Lidstone()
    heldOut_model = HeldOut()
    write_output(my_init.output_filename)
