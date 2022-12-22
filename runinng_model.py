import pickle 
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB

def storeModel(model):
    
    classifier1, classifier2, classifier3 = model
    # database
    db = {}
    db['p_array'] = classifier1
    db['p_class'] = classifier2
    db['target'] = classifier3  
    # Its important to use binary mode
    dbfile = open('Model_pickle', 'ab') #Opens a file for appending in binary mode.
      
    # source, destination
    pickle.dump(db, dbfile)                     
    dbfile.close()
  
def loadModel():
    # for reading also binary mode is important
    dbfile = open('Model_pickle', 'rb')  
    #Opens the file as read-only in binary format and starts reading from the beginning of the file. While binary format can be used for different purposes, it is usually used when dealing with things like images, videos, etc.   
    db = pickle.load(dbfile)
    lst = list()
    for keys in db:
        lst.append(db[keys])
        # print(keys, '=>', db[keys])
    model = tuple(lst)
    dbfile.close()
    return model


def storeVocabulary(Vocabulary):
     
    # Its important to use binary mode
    dbfile = open('Vocab_pickle', 'ab') #Opens a file for appending in binary mode.
      
    # source, destination
    pickle.dump(Vocabulary, dbfile)                     
    dbfile.close()
  
def loadVocabulary():
    # for reading also binary mode is important
    dbfile = open('Vocab_pickle', 'rb')  
    #Opens the file as read-only in binary format and starts reading from the beginning of the file. While binary format can be used for different purposes, it is usually used when dealing with things like images, videos, etc.   
    db = pickle.load(dbfile)
    # print(db)
    # lst = list()
    # for keys in db:
    #     lst.append(db[keys])
    #     print(keys, '=>', db[keys])
    
    dbfile.close()
    return db    

def preprocess(data, vocabulary = None):
    count_vectorizer = CountVectorizer(vocabulary = vocabulary)
    data = count_vectorizer.fit_transform(data)
    #tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)

    return data, count_vectorizer.vocabulary_ #this returns the word vectors and the  vocabulary

def classify(classifier, testdata):
    
    predicted_val=[]
    #Your code to classify test data using the learned model will go here
    # print(f"I am a hyprocriet {type(testdata)}, and {testdata}")
    test_array = testdata.toarray() 
    test_array_dimensions = test_array.shape
    ta_no_of_rows, ta_no_of_columns = test_array_dimensions[0], test_array_dimensions[1]

    p = classifier[0]
    p_class = classifier[1]
    classes = classifier[2]
    classes_length = len(classes)
    
    for x in range(ta_no_of_rows): 
        p_array = [0] * classes_length
        for y in range(classes_length):
            var = p[y][test_array[x].astype(bool)].prod()
            p_array[y] = p_class[classes[y]] * var 
        final_array = pd.Series(p_array,classes)
        predicted_val.append(final_array.idxmax())

    return predicted_val

def learn_model(data,target, classifier = None):
    
    # classifier = None
    #Your custom implementation of NaiveBayes classifier will go here.
    
    # data_array = data.toarray() # converting the data to an array  

    data_array = data.toarray()

    dimensions = data_array.shape # extracting matrix dimensions using the .shape() function
    data_no_of_rows, data_no_of_columns = dimensions[0], dimensions[1]

    target_new = np.unique(target) # formating target data 
    target_new = list(target_new) 
    target_new_length = len(target_new) 

    p_array_1, p_array_2 = np.zeros((target_new_length,data_no_of_columns)), [0]*target_new_length # empty arrays for probabilities

    for x in range(target_new_length): 
        temp = data_array[target == target_new[x]]

        temp_dimensions = temp.shape
        temp_no_of_rows, temp_no_of_columns = temp_dimensions[0], temp_dimensions[1]
        
        p_array_1[x] = np.sum(temp,axis=0)   
        p_array_1[x] += 1 
        p_array_1[x] = p_array_1[x] / (np.sum(p_array_1[x]))

        p_array_2[x] = temp_no_of_rows / data_no_of_rows

    p_class = pd.Series(p_array_2, target_new)
    if classifier == None:
        classifier = (p_array_1,p_class, target_new)
        # print("This runs")
    else:
        print(f"ust checking smthing ({classifier[0]}\n)")
        print(p_array_1,p_class, target_new, sep = '\n')
        classifier = (np.add(classifier[0],p_array_1)/2,np.add(p_class,classifier[1])/2, np.add(classifier[2],target_new)/2) #maybe implement intepolation with a normal standard guassian pdf
        # classifier = (p_array_1,p_class, target_new)
        # print(f'P_aray = \n{p_array_1}, P_class \n{p_class}, target \n {target_new} ')
    return classifier

def feed_back(txt_lst,target):
    new_vocab = loadVocabulary()
    testing_data, _  = preprocess(txt_lst, new_vocab)
    classifier = loadModel()
    # print(f"I am a hyprocriet {type(testing_data)}, and {testing_data}")
    new_classifier = learn_model(testing_data,target, classifier)
    storeModel(new_classifier)

def make_prediction(text_lst):
    new_vocab = loadVocabulary()
    # print(f'New vocab = \n{new_vocab}')
    loaded_model = loadModel()
    testing_data, _  = preprocess(text_lst, new_vocab)
    predicted = classify(loaded_model, testing_data)
    if (predicted[0] == 0.0):
        answer = 'negative'
    else:
        answer = 'positive'    
    print(f"The pridicted delimna is = {answer}")

from tqdm import tqdm
inp = input("Please inpput the text you want to have checked \n")
txt_lst = [inp]
txt1 = txt_lst.copy()
# print(type(txt_lst))
make_prediction(txt_lst)    
print(loadModel())
# learn_model([txt_lst], [[4]],loadModel() )

for i in tqdm(range(10)): 
    feed_back(txt_lst, [4] )
print(loadModel())