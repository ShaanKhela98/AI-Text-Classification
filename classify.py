#Shanpreet Singh
#April 26th, 2019
#PS7

import argparse
import collections
import csv
import math
import numbers
import os
import sys
from collections import defaultdict

one_freqlabel = {}
two_freqlabel = {}
global_dict = {}
temp_dict = []
lab_1 = 0
lab_2 = 0


onedf_freqlabel = {}
twodf_freqlabel = {}
df_temp1= []
df_temp2 = []
df_list = []
df_dict = {}

DEBUG = False

#***** Problem 8 *****
def tfmine(test_data, stream=False):
    lab_1 = 0
    lab_2 = 0
    if DEBUG:
        print("***** Entering TF *****")
        print("Data: ", test_data)

    file = open('tfmine.csv', 'w+')
    test = open(test_data, 'r+')    

    for lines in test:
        first_label = lines[0]
        break     
    
    for lines in test:
        if lines[0] == first_label:
            lab_1 += 1
            words = lines.split()
            for word in words:
                if word != first_label:
                    count = one_freqlabel.get(word, 0)
                    one_freqlabel[word] = count + 1
                    temp_dict.append(word)

        else:
            words = lines.split()
            lab_2 += 1
            for word in words: 
                if word != "-" or word != '-1':
                    count = two_freqlabel.get(word, 0)
                    two_freqlabel[word] = count + 1
                    temp_dict.append(word)
        

    golden_dict = sorted(temp_dict)
    
    for word in golden_dict:
        if word == first_label or word == "-1":
            continue
    
        try:
            if one_freqlabel[word] == two_freqlabel[word]:
                continue
            entry = str(word) + " " + str(one_freqlabel[word]) + " " + str(two_freqlabel[word]) + "\n"
            global_dict[word] = entry           
        except KeyError or TypeError:
            try: 
                entry = str(word) + " " + str(one_freqlabel[word]) + " " + str(0) + "\n"
                global_dict[word] = entry

            except KeyError or TabError:
                entry = str(word) + " " + str(0) + " " + str(two_freqlabel[word]) + "\n"
                global_dict[word] = entry
    if stream:
        print("===== Results =====")
        print("Data: "+ test_data)
        print("Class Label: 0")
        print (dict(collections.Counter(one_freqlabel).most_common(5)), "\n")

        print("\nClass Label: 1")
        print (dict(collections.Counter(two_freqlabel).most_common(5)), "\n")

    for word in global_dict:
        file.write(global_dict[word])

def mine(test_data):
    tfmine(test_data)
    if DEBUG:
        print("***** Entering MINE *****")
        print("Data: ", test_data)
    #Calculate probability of each word based on class
    tf_data = open('tfmine.csv', "r+")
    
    #Alpha value for smoothing
    Laplace_Smooth = 0.001
    prob1_c = 0.0
    prob2_c = 0.0

    probability_dict = {}

    tf_data.seek(0)
    total_size = sum(1 for line in tf_data)

    counter_1 = 0
    counter_2 = 0

    tf_data.seek(0)
    
    for line in tf_data:
        entry = line.split()
        counter_1 += float(entry[1])
        counter_2 += float(entry[2])
    
    tf_data.seek(0)

    for line in tf_data:
        entry = line.split()
        word = entry[0]
        clabel_1 = float(entry[1])
        clabel_2 = float(entry[2])

        if clabel_1 == 0.0:
            clabel_1 == Laplace_Smooth

        if clabel_2 == 0.0:
            clabel_2 == Laplace_Smooth

        #Exponent of probabilities is amount of appearances in doc 
        # (use log and add) 
        
        prob1_c = float((1+clabel_1)/(1+counter_1))

        prob2_c = float((1+clabel_2)/(1+counter_2))

        entry = (prob1_c, prob2_c)

        probability_dict[word] = entry
        prob1_c = 0
        prob2_c = 0        
        #probability_dict[word] = probability of 1 and probaility of -1
        #probabliity_dict[word] = entry
           
    if DEBUG:
        print ("Amount of words in file: " + str(total_size))
        bet=counter_1+counter_2
        print("REAL AMOUNT: " + str(bet) )

    #Parse each line and see if best word is in. If it is then guess 1 else -1
    TP = 0 # TP = predicted in class 0, is in class 0 
    FN = 0 # FN = predicted in class 0, is in class 1
    FP = 0 # FP = predicted in class 1, is in class 0
    TN = 0 # TN = predicted in class 1, is in class 1

    data_train = open(test_data, "r+")
    
    c1 = 0
    c2 = 0

    for lines in data_train:
        first_label = lines[0]
        break
        
    data_train.seek(0)
    i=0

    for document in data_train:
        line = document.split()
        expon_list = word_count(line)
        for word in line:
            if word in probability_dict:
                temp = list(probability_dict[word])
                c1 += (expon_list[word] * math.log(temp[0]))
                c2 += (expon_list[word] * math.log(temp[1]))

        if c1 >= c2: #If actual word in line
                if document[0] == first_label: 
                    TP += 1 #guessed right
                else: 
                    FN += 1 #Guessed wrong
        else:
            if document[0] != first_label: #if actual not in line
                TN += 1 #guessed right
            else:
                FP += 1 #Guess wrongS
        c1 = 0
        c2 = 0
        expon_list.clear()        


    print("===== MNB Result ======")
    print("Data: ", test_data)
    print("*** Confusion Matrices ***")
    print("TP   FN  | " + str(TP) + " " + str(FN))
    print("FP   TN  | " + str(FP) + " " + str(TN)) 
    print("")


#***** Problem 7 *****
def nb(test_data):
    if DEBUG:
        print("***** Entering NB *****")
        print("Data: ", test_data)
    #Calculate probability of each word based on class
    tf_data = open('df.csv', "r+")
    
    #Alpha value for smoothing
    Laplace_Smooth = 0.001
    prob1_c = 0.0
    prob2_c = 0.0

    probability_dict = {}

    tf_data.seek(0)
    total_size = sum(1 for line in tf_data)

    counter_1 = 0
    counter_2 = 0

    tf_data.seek(0)
    
    for line in tf_data:
        entry = line.split()
        counter_1 += float(entry[1])
        counter_2 += float(entry[2])
    
    tf_data.seek(0)

    for line in tf_data:
        entry = line.split()
        word = entry[0]
        clabel_1 = float(entry[1])
        clabel_2 = float(entry[2])

        if clabel_1 == 0.0:
            clabel_1 == Laplace_Smooth

        if clabel_2 == 0.0:
            clabel_2 == Laplace_Smooth

        #Exponent of probabilities is amount of appearances in doc 
        # (use log and add) 
        
        prob1_c = float((1+clabel_1)/(2+counter_1))
        prob2_c = float((1+clabel_2)/(2+counter_2))

        not_prob1 = 1 - prob1_c
        not_prob2 = 1 - prob2_c

        entry = (prob1_c, not_prob1, prob2_c, not_prob2)

        probability_dict[word] = entry
        prob1_c = 0
        prob2_c = 0        
        #probability_dict[word] = probability of 1 and probaility of -1
        #probabliity_dict[word] = entry
           
    #Parse each line and see if best word is in. If it is then guess 1 else -1
    TP = 0 # TP = predicted in class 0, is in class 0 
    FN = 0 # FN = predicted in class 0, is in class 1
    FP = 0 # FP = predicted in class 1, is in class 0
    TN = 0 # TN = predicted in class 1, is in class 1

    data_train = open(test_data, "r+")
    
    c1 = 0
    c2 = 0
    not1 = 0
    not2 = 0

    for lines in data_train:
        first_label = lines[0]
        break
        
    data_train.seek(0)

    for document in data_train:
        line = document.split()
        expon_list = word_count(line)
        for word in line:
            if word in probability_dict:
                temp = list(probability_dict[word])
                c1 += (expon_list[word] * math.log(temp[0]))
                c2 += (expon_list[word] * math.log(temp[2]))
                not1 += (expon_list[word] * math.log(temp[1]))
                not2 += (expon_list[word] * math.log(temp[3]))

        if c1 >= c2: #If actual word in line
            if document[0] == first_label: 
                TP += 1 #guessed right
            else: 
              FN += 1 #Guessed wrong
        else:
            if document[0] != first_label: #if actual not in line
                TN += 1 #guessed right
            else:
                FP += 1 #Guess wrongS
        c1 = 0
        c2 = 0
        not1 = 0
        not2 = 0
        expon_list.clear()        


    print("===== NB Result ======")
    print("Data: ", test_data)
    print("*** Confusion Matrices ***")
    print("TP   FN  | " + str(TP) + " " + str(FN))
    print("FP   TN  | " + str(FP) + " " + str(TN)) 
    print("")



#***** Problem 6 *****

# so lets say there are 1000 documents of class 1
# you check all of those documents and in 400 of those documents, list appears at least once
# the df for list in class 1 is 400
def word_countdf(str): #Breaks word according 
    counts = dict()
    words = str

    for word in words:
        if word in counts:
            counts[word] = 1
        else:
            counts[word] = 1
    return counts

def df(test_data, stream=True):
    #Make a list of all words
    
    #for word in list, if word in doc: find first label then split the count of how many documents have that word 
    
    if DEBUG:
        print("***** Entering DF *****")
        print("Data: ", test_data)

    file = open('df.csv', 'w+')
    test = open(test_data, 'r+')    

    for lines in test:
        first_label = lines[0]
        break     
    
    test.seek(0)
    df_temp = []

    for lines in test:
        sentence = lines.split()
        words_list = word_countdf(sentence)
        if lines[0] == first_label:
            for word in words_list:
                if word != first_label:
                    if word in df_temp1:
                        onedf_freqlabel[word] += int(words_list[word])
                    else:
                        onedf_freqlabel[word] = 1
                        df_temp1.append(word)
                        df_list.append(word)
                    
        else:
            for word in words_list: 
                if word == "-" or word == "-1":
                    continue
                else:
                    if word in df_temp2:
                        twodf_freqlabel[word] += int(words_list[word])
                    else:
                        twodf_freqlabel[word] = 1
                        df_temp2.append(word)
                        df_list.append(word)
 

    golden_dict = sorted(df_list)

    for word in golden_dict:
        if word == first_label or word == "-1":
            continue
    
        try:
            entry = str(word) + " " + str(onedf_freqlabel[word]) + " " + str(twodf_freqlabel[word]) + "\n"
            df_dict[word] = entry           
        except KeyError or TypeError:
            try: 
                entry = str(word) + " " + str(onedf_freqlabel[word]) + " " + str(0) + "\n"
                df_dict[word] = entry

            except KeyError or TabError:
                entry = str(word) + " " + str(0) + " " + str(twodf_freqlabel[word]) + "\n"
                df_dict[word] = entry
    temp = ""
    for word in golden_dict: 
        if word != temp:
            file.write(df_dict[word])
        temp = word


    if stream:
        print("===== Results =====")
        print("Data: "+ test_data)
        print("Class Label: 0")
        print (dict(collections.Counter(onedf_freqlabel).most_common(5)), "\n")

        print("\nClass Label: 1")
        print (dict(collections.Counter(twodf_freqlabel).most_common(5)), "\n")


#***** Problem 5 *****
def word_count(str): #Breaks word according 
    counts = dict()
    words = str

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

def mnb(test_data): 
    if DEBUG:
        print("\n***** Entering MNB *****")
        print("Data: ", test_data)
    
    #Calculate probability of each word based on class
    tf_data = open('tf.csv', "r+")
    
    #Alpha value for smoothing
    Laplace_Smooth = 0.001
    prob1_c = 0.0
    prob2_c = 0.0

    probability_dict = {}

    tf_data.seek(0)
    total_size = sum(1 for line in tf_data)

    counter_1 = 0
    counter_2 = 0

    tf_data.seek(0)
    
    for line in tf_data:
        entry = line.split()
        counter_1 += float(entry[1])
        counter_2 += float(entry[2])
    
    tf_data.seek(0)

    for line in tf_data:
        entry = line.split()
        word = entry[0]
        clabel_1 = float(entry[1])
        clabel_2 = float(entry[2])

        if clabel_1 == 0.0:
            clabel_1 == Laplace_Smooth

        if clabel_2 == 0.0:
            clabel_2 == Laplace_Smooth

        #Exponent of probabilities is amount of appearances in doc 
        # (use log and add) 
        
        prob1_c = float((1+clabel_1)/(1+counter_1))

        prob2_c = float((1+clabel_2)/(1+counter_2))

        entry = (prob1_c, prob2_c)

        probability_dict[word] = entry
        prob1_c = 0
        prob2_c = 0        
        #probability_dict[word] = probability of 1 and probaility of -1
        #probabliity_dict[word] = entry
           
    if DEBUG:
        print ("Amount of words in file: " + str(total_size))
        bet=counter_1+counter_2
        print("REAL AMOUNT: " + str(bet) )

    #Parse each line and see if best word is in. If it is then guess 1 else -1
    TP = 0 # TP = predicted in class 0, is in class 0 
    FN = 0 # FN = predicted in class 0, is in class 1
    FP = 0 # FP = predicted in class 1, is in class 0
    TN = 0 # TN = predicted in class 1, is in class 1

    data_train = open(test_data, "r+")
    
    c1 = 0
    c2 = 0

    for lines in data_train:
        first_label = lines[0]
        break
        
    data_train.seek(0)
    
    for document in data_train:
        line = document.split()
        expon_list = word_count(line)
        for word in line:
            if word in probability_dict:
                temp = list(probability_dict[word])
                c1 += (expon_list[word] * math.log(temp[0]))
                c2 += (expon_list[word] * math.log(temp[1]))

        if c1 >= c2: #If actual word in line
                if document[0] == first_label: 
                    TP += 1 #guessed right
                else: 
                    FN += 1 #Guessed wrong
        else:
            if document[0] != first_label: #if actual not in line
                TN += 1 #guessed right
            else:
                FP += 1 #Guess wrongS
        c1 = 0
        c2 = 0
        expon_list.clear()        


    print("===== MNB Result ======")
    print("Data: ", test_data)
    print("*** Confusion Matrices ***")
    print("TP   FN  | " + str(TP) + " " + str(FN))
    print("FP   TN  | " + str(FP) + " " + str(TN)) 
    print("")


#****** Problem 4 ******

# The class prior is the percentage of documents that are of the given class
# If there are 10 total documents: 7 -> class 0 && 3 -> class 1
# Class 0 = 7/10 
# 

def priors(test_data):
    if DEBUG:    
        print("\n***** Entering PRIORS *****")
        print("Data: ", test_data)

    TP = 0 # TP = predicted in class 0, is in class 0 
    FN = 0 # FN = predicted in class 0, is in class 1
    FP = 0 # FP = predicted in class 1, is in class 0
    TN = 0 # TN = predicted in class 1, is in class 1

    total_size = sum(1 for line in open(test_data))
    
    probc_1 = lab_1/total_size
    probc_2 = lab_2/total_size
    

    data_test = open(test_data, "r+")

    for document in data_test:
        entry = document.split()
        if probc_1 >= probc_2: #If actual word in line
                if document[0] == "1": 
                    TP += 1 #guessed right
                else: 
                    FN += 1 #Guessed wrong
        else:
            if document[0] != "1": #if actual not in line
                TN += 1 #guessed right
            else:
                FP += 1 #Guess wrong
 
    
    print("===== TFGREP Result ======")
    print("Data: ", test_data)
    print("*** Confusion Matrices ***")
    print("TP   FN  | " + str(TP) + " " + str(FN))
    print("FP   TN  | " + str(FP) + " " + str(TN)) 
    print("")
     


#***** Problem 3 tfgrep *****
def tfgrep(test_data):
    if DEBUG:    
        print("\n***** Entering TFGREP *****")
        print("Data: ", test_data)

    best_diff = 0
    data = open('tf.csv', 'r+')
     
    for word in data:
        entry = word.split(' ')
        list_1 = float(entry[1])
        list_2 = float(entry[2])
        challenge_diff = abs(list_1 - list_2)

        if challenge_diff > best_diff:
            best_diff = challenge_diff
            best_word = entry[0]
                

    #Two confusion matrices: First indicating performance on train and second being test 

    #Parse each line and see if best word is in. If it is then guess 1 else -1
    TP = 0 # TP = predicted in class 0, is in class 0 
    FN = 0 # FN = predicted in class 0, is in class 1
    FP = 0 # FP = predicted in class 1, is in class 0
    TN = 0 # TN = predicted in class 1, is in class 1

    data_test = open(test_data, "r+")

    for document in data_test:
        if best_word in document: #If actual word in line
            if document[0] == "1": 
                TP += 1 #guessed right
            else: 
                FN += 1 #Guessed wrong
        else:
            if document[0] != "1": #if actual not in line
                TN += 1 #guessed right
            else:
                FP += 1 #Guess wrong
 
    
    print("===== tfgrep result ======")
    print("Data: ", test_data)
    print("Most discriminating term: " + str(best_word) + "\n")
    print("*** Confusion Matrices ***")
    print("TP   FN  | " + str(TP) + " " + str(FN))
    print("FP   TN  | " + str(FP) + " " + str(TN)) 
    print("")


    
#***** Problem 2 Term Frequency *****
def tf(test_data, stream=True):
    lab_1 = 0
    lab_2 = 0
    if DEBUG:
        print("***** Entering TF *****")
        print("Data: ", test_data)

    file = open('tf.csv', 'w+')
    test = open(test_data, 'r+')    

    for lines in test:
        first_label = lines[0]
        break     
    
    for lines in test:
        if lines[0] == first_label:
            lab_1 += 1
            words = lines.split()
            for word in words:
                if word != first_label:
                    count = one_freqlabel.get(word, 0)
                    one_freqlabel[word] = count + 1
                    temp_dict.append(word)

        else:
            words = lines.split()
            lab_2 += 1
            for word in words: 
                if word != "-" or word != '-1':
                    count = two_freqlabel.get(word, 0)
                    two_freqlabel[word] = count + 1
                    temp_dict.append(word)
        

    golden_dict = sorted(temp_dict)
    
    for word in golden_dict:
        if word == first_label or word == "-1":
            continue
        try:
            entry = str(word) + " " + str(one_freqlabel[word]) + " " + str(two_freqlabel[word]) + "\n"
            global_dict[word] = entry           
        except KeyError or TypeError:
            try: 
                entry = str(word) + " " + str(one_freqlabel[word]) + " " + str(0) + "\n"
                global_dict[word] = entry

            except KeyError or TabError:
                entry = str(word) + " " + str(0) + " " + str(two_freqlabel[word]) + "\n"
                global_dict[word] = entry
    if stream:
        print("===== Results =====")
        print("Data: "+ test_data)
        print("Class Label: 0")
        print (dict(collections.Counter(one_freqlabel).most_common(5)), "\n")

        print("\nClass Label: 1")
        print (dict(collections.Counter(two_freqlabel).most_common(5)), "\n")

    for word in global_dict:
        file.write(global_dict[word])

# ******* Problem 1 Main ************ #               
def textfile(value):
    # m=magic.open(magic.MAGIC_NONE)
    # if not m.load(value) == 'text/plain':
    #     print ("FAIL")
    #     return -1
    return 1

def parse_text(document, label):
    test = open('pos.csv', 'w+')
    temp = []
    with open(document)as doc:
        for line in doc:
            if line[0] == label:
                temp += line #Get every line
    return temp

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("*** ERROR ***")
        print("\nPlease enter data in correct format")
        print("python3 classify.py (Training Data) (Testing Data) (Execution function)\n")

    else:
        train_data = sys.argv[1]
        test_data = sys.argv[2]
        execute_func = sys.argv[3]

        check_train = textfile(train_data)
        check_test = textfile(test_data)

        if check_train == -1 or check_test == -1:
            print("*** ERROR ***")
            print("\nPlease enter data in correct format")
            print("python3 classify.py (Training Data MUST BE PLAIN TEXT) (Testing Data MUST BE PLAIN TEXT) (Execution function)\n")

        else:
            if DEBUG:
                print ("Training Data: ", train_data)
                print ("Testing Data: ", test_data)
                print ("Execution function:", execute_func, "\n")
            
            #pos_list = parse_text(test_data, '1')
            #sub_list = parse_text(test_data, '-1')

            if sys.argv[3] == "tf":
                tf(train_data, True)

            if sys.argv[3] == "tfgrep":
                tf(train_data, False)
                tfgrep(train_data)
                tfgrep(test_data)

            if sys.argv[3] == "priors":
                tf(train_data, False)
                priors(train_data)
                priors(test_data)
            
            if sys.argv[3] == "mnb":
                tf(train_data, False)
                mnb(train_data)
                mnb(test_data)

            if sys.argv[3] == "df":
                df(train_data, True)

            if sys.argv[3] == "nb":
                df(train_data, False)
                nb(train_data)
                nb(test_data)
            
            if sys.argv[3] == "mine":
                print("")
                print("The changes in my default implementation has to deal with the minimization of False Negative and False Positive")
                print("I am using MNB and creating simpler complexity in which the matrices produce more TP and TN")
                print("I think by raising our floor and eliminating words that appear in both class labels have some impact in our probabilites and taking these terms out will help increase chances of getting a better result due to less sample size")
                print("Taking out words that appear in both classes with the same tf")
    
                mine(train_data)
                mine(test_data)
