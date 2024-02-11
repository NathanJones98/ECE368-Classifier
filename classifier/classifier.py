import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import copy

#finding if the probability of p[w|spam] and p[w|ham] 
# essentially finding the probablility of the word given it's spam
def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    #print(file_lists_by_category)

    #W1, W2
    spam_files, ham_files = file_lists_by_category

    spam_word_count = dict()
    ham_word_count = dict()

    #[wd : xd]
    #For both spam and ham emails, 
    #get_word_freq -> For a specific word, count the use of the word across 
    # all spam emails
    spam_word_count = util.get_word_freq(spam_files)
    ham_word_count = util.get_word_freq(ham_files)

    spam_emails_total_words = 0
    for word in spam_word_count:
        spam_emails_total_words += spam_word_count[word]

    ham_emails_total_words = 0
    for word in ham_word_count:
        ham_emails_total_words += ham_word_count[word]
    
    combined_dict = copy.deepcopy(spam_word_count)
    assert(combined_dict == spam_word_count)
    for key in ham_word_count:
        if key in combined_dict:
            combined_dict[key] += ham_word_count[key]
        else:
            combined_dict[key] = ham_word_count[key]
    
    _v_ = len(combined_dict)

    p_d = dict()
    q_d = dict()

    for word in spam_word_count:
        p_d[word] = (spam_word_count[word] + 1) / (spam_emails_total_words + _v_)
    
    for word in ham_word_count:
        q_d[word] = (ham_word_count[word] + 1) / (ham_emails_total_words + _v_)

    for word in combined_dict:
        if not (word in spam_word_count):
            p_d[word] = 1 / (spam_emails_total_words + _v_)

        if not (word in ham_word_count):
            q_d[word] = 1 / (ham_emails_total_words + _v_)

    #for item in p_d:
        #print(item, p_d[item])
        #input("Test")

    probabilities_by_category = (p_d, q_d)
    
    return probabilities_by_category


def classify_new_email(filename,probabilities_by_category,prior_by_category,tradeOff):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified

    probabilities_by_category: output of function learn_distributions

    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    SPAM_STRING = 'spam'
    HAM_STRING = 'ham'

    spam_ham_string = ''
    posterior_probabilities = []
    
    p_d = probabilities_by_category[0] #spam
    q_d = probabilities_by_category[1] #ham
    
    # Split the email contents into individual words
    words = util.get_words_in_file(filename)

    # Initialize the log probabilities
    log_p_spam = np.log(prior_by_category[0])
    log_p_ham = np.log(prior_by_category[1])

    # Calculate the log probabilities of the email belonging to spam and ham categories
    # log_p_spam => p[spam|Xn] = p[word|spam] * p[y] 
    # p[x] is same for both (word occurance / total word occurances)
    for word in words:
        if word in p_d:
            log_p_spam += np.log(p_d[word])
        else:
            log_p_spam += np.log(1 / len(p_d))

        if word in q_d:
            log_p_ham += np.log(q_d[word])
        else:
            log_p_ham += np.log(1 / len(q_d))

    # Determine the classification result
    posterior_probabilities = [log_p_spam, log_p_ham]

    if log_p_spam - np.log(tradeOff) >= log_p_ham:
        spam_ham_string = SPAM_STRING
    else:
        spam_ham_string = HAM_STRING

    classify_result = (spam_ham_string, posterior_probabilities)

    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    tradeOffs = [1e-5, 1e-4, 1e-3, 1e-2, 1, 1e3, 1e4, 1e5, 1e10, 1e22]
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    spam_emails_missed = []
    ham_emails_missed = []

    # Classify emails from testing set and measure the performance
    for trades in tradeOffs:

        performance_measures = np.zeros([2,2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                    probabilities_by_category,
                                                    priors_by_category,
                                                    trades)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0],totals[0],correct[1],totals[1]))
        spam_emails_missed.append(totals[0] - correct[0])
        ham_emails_missed.append(totals[1] - correct[1])

    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    
    #plt.plot(tradeOffs, spam_emails_missed, 'r', label='Spam')
    #plt.plot(tradeOffs, ham_emails_missed, 'b', label='Ham')
    
    plt.plot(spam_emails_missed, ham_emails_missed, 'r', label='Spam')

    plt.xlabel('Type 1 (Spam classified as Ham) Errors')
    plt.ylabel('Type 2 (Ham classified as Spam) Errors')
    plt.title('Spam Vs Ham Tradeoff')
    plt.legend()

    plt.savefig("nbc.pdf")
    #plt.show()

 