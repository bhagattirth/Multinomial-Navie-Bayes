from utils import load_test_set, load_training_set
import numpy as np

def preprocessing(dataset):
    data_entry = np.empty(len(dataset), dtype=object)
    for i in range(len(dataset)):
        data = {}
        for word in dataset[i]:
            data[word] = 1 if word not in data else data[word] + 1
        data_entry[i] = data
    return data_entry

def train_and_use_model(training, testing, training_wc, vocab):
    
    p_train, n_train = training
    p_test, n_test = testing
    p_train_wc, n_train_wc = training_wc

    p_prior = len(p_train)/(len(p_train) + len(n_train))
    n_prior = 1 - p_prior


def naive_runner(P_TRAIN, N_TRAIN, 
                 P_TEST, N_TEST):
    
    pos_train, neg_train, vocab = load_training_set(P_TRAIN, N_TRAIN)
    pos_test,  neg_test = load_test_set(P_TEST, N_TEST)
    vocab = np.array(vocab)
    
    p_train_d = preprocessing(pos_train)
    p_train_wc = sum(len(doc) for doc in pos_train)
    
    n_train_d = preprocessing(neg_train)
    n_train_wc = sum(len(doc) for doc in neg_train)

    train_package = (p_train_d, n_train_d)
    train_wc_package = (p_train_wc, n_train_wc)

    train_and_use_model(train_package, train_wc_package, train_package, vocab)

    # p_test_d = preprocessing(pos_test)
    # n_test_d = preprocessing(neg_test)





if __name__=="__main__":
    PER_POS_INST_TRAIN = 0.004
    PER_NEG_INST_TRAIN = 0.004
    PER_POS_INST_TEST  = 0.004
    PER_NEG_INST_TEST  = 0.004

    naive_runner(PER_POS_INST_TRAIN, PER_NEG_INST_TRAIN,
                PER_POS_INST_TEST, PER_NEG_INST_TEST)




