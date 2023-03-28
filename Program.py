from utils import load_test_set, load_training_set
from helper import indexing, preprocess, plot
from math import log

def pred_class_log(training, pos_prior, wc_package, uniqueWords, datapoint, alpha):
    p_train, n_train = training
    p_term_count, n_term_count = wc_package
    pos_prob = log(pos_prior)
    neg_prob = log(1 - pos_prior)
    
    for word in datapoint:
        term_feq = datapoint[word]
        p_doc_freq = p_train.get(word, 0)
        n_doc_freq = n_train.get(word, 0)
        
        pos_prob += log((p_doc_freq + alpha)/(p_term_count + alpha * (uniqueWords))) 
        neg_prob += log((n_doc_freq + alpha)/(n_term_count + alpha * (uniqueWords))) 

    return 0 if pos_prob > neg_prob else 1


def pred_class(training, pos_prior, wc_package, uniqueWord, datapoint, alpha):
    p_train, n_train = training
    p_term_count, n_term_count = wc_package
    pos_prob = pos_prior
    neg_prob = 1 - pos_prior
    
    for word in datapoint:
        term_feq = datapoint[word]
        p_doc_freq = p_train.get(word, 0)
        n_doc_freq = n_train.get(word, 0)
        
        pos_prob *= ((p_doc_freq + alpha)/(p_term_count + alpha * (uniqueWord)))
        neg_prob *= ((n_doc_freq + alpha)/(n_term_count + alpha * (uniqueWord)))


    return 0 if pos_prob > neg_prob else 1


def train_and_use_model(training, testing, num_of_doc, alpha=.00001, IS_LOG = False):  
    print("calculting accuracy")
    p_test, n_test = testing
    pos_prior = num_of_doc[0]/(num_of_doc[1] + num_of_doc[0])
    p_train_wc = sum(training[0][word] for word in training[0])
    n_train_wc = sum(training[1][word] for word in training[1])
    wc_package = (p_train_wc, n_train_wc)

    calc = pred_class_log if IS_LOG else pred_class

    TP, FN, TN, FP = 0, 0, 0, 0
    for entry in p_test:
        label = calc(training, pos_prior, wc_package, num_of_doc[2], entry, alpha)
        if label == 0:
            TP += 1
        else:
            FN+= 1

    for entry in n_test:
       label = calc(training, pos_prior, wc_package, num_of_doc[2], entry, alpha)
       if label == 1:
           TN += 1
       else:
           FP +=1
    
    acc = (TP+TN)/(len(p_test) + len(n_test))
    recall, per, matrix = compute_Re_Per(TP, TN, FP, FN)
    return acc, recall, per, matrix

def compute_Re_Per(TP, TN, FP, FN):
    recall = TP/(TP + FN)
    per = TP/(TP + FP)
    matrix = [[TP, FN], [FP, TN]]
    return recall, per, matrix


def naive_runner(P_TRAIN, N_TRAIN, P_TEST, N_TEST, ALPHA, IS_LOG):
    # Load Training and Test Data
    print("Loading Data...")
    pos_train, neg_train, vocab = load_training_set(P_TRAIN, N_TRAIN)
    pos_test,  neg_test = load_test_set(P_TEST, N_TEST)
    print("Loaded")
    
    # Creates a class dictionary of term freq
    p_train_d = preprocess(pos_train)
    n_train_d = preprocess(neg_train)
    print("preprocessing complete")
    train_package = (p_train_d, n_train_d)
    
    # Indexes the Training docs
    p_test_index = indexing(pos_test)
    n_test_index = indexing(neg_test)
    print("Indexing complete")
    test_package=(p_test_index, n_test_index)

    # Number of docs in each set
    doc_class_info = (len(pos_train), len(neg_train), len(vocab))

    return train_and_use_model(train_package, test_package, doc_class_info, ALPHA, IS_LOG)

if __name__=="__main__":
    # PER_POS_INST_TRAIN = 0.2
    # PER_NEG_INST_TRAIN =  0.2
    # PER_POS_INST_TEST  =  0.2
    # PER_NEG_INST_TEST  =  0.2
    # IS_LOG = True
    # ALPHA = 1

    PER_POS_INST_TRAIN = .1
    PER_NEG_INST_TRAIN =  .5
    PER_POS_INST_TEST  =  1
    PER_NEG_INST_TEST  =  1
    IS_LOG = True
    ALPHA = 10

    acc, recall, per, matrix = naive_runner(PER_POS_INST_TRAIN, PER_NEG_INST_TRAIN,
                                            PER_POS_INST_TEST, PER_NEG_INST_TEST, ALPHA, IS_LOG)


    # x_val, y_val = [], []
    # for alpha in [.0001, .001, .01, .1, 1, 10, 100, 1000]:
    #     acc, recall, per, matrix = naive_runner(PER_POS_INST_TRAIN, PER_NEG_INST_TRAIN,
    #                                         PER_POS_INST_TEST, PER_NEG_INST_TEST, alpha, IS_LOG)
    #     x_val.append(alpha)
    #     y_val.append(acc)
    # plot(x_val, y_val, "Different Alpha Values", "Alpha Value", "Accuracies", "graph.png")


    print("Accuracy:", acc, "Recall:", recall, "Percesion:", per)
    print("Confusion Matrix:", matrix)