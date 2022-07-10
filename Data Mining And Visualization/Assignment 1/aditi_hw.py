import pandas as pd
import numpy as np
import yaml

def load_data(df):
    '''In this function the training or testing Dataframe is passed after which the all the elements of the Dataframe is read and divided into 3 Dataframes, each consisting of the data of the 
    respective Classes.'''
    df_class1 = pd.DataFrame()
    df_class2 = pd.DataFrame()
    df_class3 = pd.DataFrame()
    temp_list=[]
    for index, row in df.iterrows():                   #All the elements of the Dataframe are traversed
        temp_list=[]
        for i in range(0,len(row)):
            if(str(df.iloc[index,i]).find("class")==0):          #The elements of the Dataframe are segregated into Different classes and their corresponding data is stored.
                if(df.iloc[index,i].find("class-1")==0):
                    temp_list.append(1)
                    df_class1=df_class1.append(pd.DataFrame([temp_list]), ignore_index=True)
                elif(df.iloc[index,i].find("class-2")==0):
                    temp_list.append(2)
                    df_class2=df_class2.append(pd.DataFrame([temp_list]), ignore_index=True)
                elif(df.iloc[index,i].find("class-3")==0):
                    temp_list.append(3)
                    df_class3=df_class3.append(pd.DataFrame([temp_list]), ignore_index=True)
            else:    
                temp_list.append(df.iloc[index,i])
    return df_class1,df_class2,df_class3

import numpy as np
#
# Perceptron implementation
#
class Perceptron(object):
     
    def __init__(self, n_iterations=20, random_state=1, learning_rate=0.0001):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
 
    ''' The algorithm used is Stochastic Gradient Descent (SGD). In this algorithm the weights are updated after each training iteration and based on the learning eate and the accuracy of the  
    prediction the weights and biases are upated. '''
     
    
    def fit(self, X, y,class_1,class_2):
        rand = np.random.RandomState(self.random_state)
        list_predicted=[]
        list_acc=[]
        wrong_classification=0
        self.params = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.weights=self.params[1:]
        self.bias=self.params[0]
        for _ in range(self.n_iterations):
            wrong_classification=0
            for xi, actual_value in zip(X, y):
                predicted_value = self.predict(xi,class_1,class_2)
                if(predicted_value!=actual_value):
                    wrong_classification=wrong_classification+1
                self.weights = self.weights + self.learning_rate * (actual_value - predicted_value) * xi
                self.bias = self.bias + self.learning_rate * (actual_value - predicted_value) * 1
            acc=(len(y)-wrong_classification)/len(y)
            list_acc.append(acc)
        avg_acc=sum(list_acc)/len(list_acc)
        print("Average Training Accuracy "+str(avg_acc))
        return avg_acc
    '''The regularization term is added in the weight update equation where the Term for L2 regularization (1-2*(regularization_coeff * learning_rate)) is multiplied with the weight'''
    def fit_regularizer(self, X, y,class_1,class_2,reg_coeff):
        rand = np.random.RandomState(self.random_state)
        list_predicted=[]
        list_acc=[]
        wrong_classification=0
        self.params = rand.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.weights=self.params[1:]
        self.bias=self.params[0]
       
        for _ in range(self.n_iterations):
            wrong_classification=0
            for xi, actual_value in zip(X, y):
                predicted_value = self.predict(xi,class_1,class_2)
                if(predicted_value!=actual_value):
                    wrong_classification=wrong_classification+1
                
                self.weights = (1-(2*reg_coeff*self.learning_rate))*self.weights + self.learning_rate * (actual_value - predicted_value) * xi
                self.bias = self.bias + self.learning_rate * (actual_value - predicted_value) * 1
            acc=(len(y)-wrong_classification)/len(y)
            list_acc.append(acc)
        avg_acc=sum(list_acc)/len(list_acc)
        print("Average Training Accuracy "+str(avg_acc))
        return avg_acc
        
    def predict(self, X,class_1,class_2):                     #This funtion computes the forward propagation of the perceptron algorithm. Where the weights are multiplied with the input data and then a bias is added. Based on it a step function is used to classify if it is label 0 or 1. If the weighted_sum is greater than 0 then it will be 1 otherwise 0.
        weighted_sum = np.dot(X, self.weights) + self.bias
        return np.where(weighted_sum >= 0.0, class_2, class_1)
     
    '''The test function evaluated how good the perceptron model has been trained to classify the kind of data. The function uses the data and calculates a weighted sum to know the label of the 
    data. On a set of test it determines how many of them have been wrong classifications. Based on which it returns a score.'''
    def test(self, X, y,class_1,class_2):
        wrong_classification=0
        for xi, label in zip(X, y):
            output = self.predict(xi,class_1,class_2)
            if(label != output):
                wrong_classification=wrong_classification+1
        self.score_data = (len(y) - wrong_classification)/len(y)
        print("Test accuracy "+str(self.score_data))
        return self.score_data





def binary_classifier(df_train,df_test,cfg):

    '''In this function Question 3 solution has been implemented. For each class of data the features and labels are separated into different variables for each class.
    For each kind of classification the features and labels are separated and the first class is labeled as 0 and the second class is labeled as 1. The features and labels of both the classes 
    are combined. This is done for both training and testing data. After this the training and testing features and labels are passed on the respective training and testing functions.'''
    
    df_class1,df_class2,df_class3=load_data(df_train)
    df_class1_t,df_class2_t,df_class3_t=load_data(df_test)
    X_1=df_class1.iloc[:,:4]
    y_1=df_class1.iloc[:,4]
    X_2=df_class2.iloc[:,:4]
    y_2=df_class2.iloc[:,4]
    X_3=df_class3.iloc[:,:4]
    y_3=df_class3.iloc[:,4]
    
    X_1_t=df_class1.iloc[:,:4]
    y_1_t=df_class1.iloc[:,4]
    X_2_t=df_class2.iloc[:,:4]
    y_2_t=df_class2.iloc[:,4]
    X_3_t=df_class3.iloc[:,:4]
    y_3_t=df_class3.iloc[:,4]
    
    if(cfg['part3']['execute_classification_1']==1):                #This part of the code performs Class 1 vs Class 2 binary classification
        print("Executing Classification 1 : -")
        print("-------------------------------------------------------- \n\n\n")
        train_labels_X1=[]
        train_labels_X2=[]
        test_labels_X1=[]
        test_labels_X2=[]
        for i in range(0,len(y_1)):                       #Class 1 training is labeled as 0
            train_labels_X1.append(0)
        for i in range(0,len(y_2)):                       #Class 2 training is labeled as 1
            train_labels_X2.append(1)
        for i in range(0,len(y_1_t)):                     #Class 1 testing is labeled as 0 
            test_labels_X1.append(0)
        for i in range(0,len(y_2_t)):                     #Class 2 testing is labeled as 1
            test_labels_X2.append(1)
        train_labels_X1=np.array(train_labels_X1)         #The lists are converted to numpy arrays
        train_labels_X2=np.array(train_labels_X2)
        train_labels_X1=pd.DataFrame(train_labels_X1)     #The numpy arrays are converted to dataframes
        train_labels_X2=pd.DataFrame(train_labels_X2)
        test_labels_X1=pd.DataFrame(test_labels_X1)
        test_labels_X2=pd.DataFrame(test_labels_X2)
        df_train_X1_X2=X_1.append(X_2, ignore_index=True)  #The training data of Class 1 and Class 2 are combined 
        #df_train_labels_X1_X2=y_1.append(y_2, ignore_index=True)
        df_train_labels_X1_X2=pd.concat([train_labels_X1,train_labels_X2],ignore_index = True,sort = False) #The training labels of Class 1 and Class 2 are combined
        df_test_X1_X2=X_1_t.append(X_2_t, ignore_index=True)                       #The testing data of Class 1 and Class 2 are combined 
        #df_test_labels_X1_X2=y_1_t.append(y_2_t, ignore_index=True)
        df_test_labels_X1_X2=pd.concat([test_labels_X1,test_labels_X2],ignore_index = True,sort = False)         #The testing labels of Class 1 and Class 2 are combined
        
        df_train_X1_X2=df_train_X1_X2.to_numpy()                       # The dataframes are converted to numpy arrays
        df_train_labels_X1_X2=df_train_labels_X1_X2.to_numpy()
        df_test_X1_X2=df_test_X1_X2.to_numpy()
        df_test_labels_X1_X2=df_test_labels_X1_X2.to_numpy()
        perceptron = Perceptron()                             #Creating the object of the Perceptron class
        perceptron.fit(df_train_X1_X2, df_train_labels_X1_X2,0,1)    #Training the perceptron mode
        perceptron.test(df_test_X1_X2, df_test_labels_X1_X2,0,1)    #Testing the perceptron mode
        print("Classification 1 Completed")
        print("---------------------------------------------------------------------------- \n\n\n")
        
    if(cfg['part3']['execute_classification_2']==1):
        print("Executing Classification 2 : -")
        print("-------------------------------------------------------- \n\n\n")
        train_labels_X2=[]
        train_labels_X3=[]
        test_labels_X2=[]
        test_labels_X3=[]
        for i in range(0,len(y_2)):
            train_labels_X2.append(0)
        for i in range(0,len(y_3)):
            train_labels_X3.append(1)
        for i in range(0,len(y_2_t)):
            test_labels_X2.append(0)
        for i in range(0,len(y_3_t)):
            test_labels_X3.append(1)
        train_labels_X2=np.array(train_labels_X2)
        train_labels_X3=np.array(train_labels_X3)
        train_labels_X2=pd.DataFrame(train_labels_X2)
        train_labels_X3=pd.DataFrame(train_labels_X3)
        test_labels_X2=pd.DataFrame(test_labels_X2)
        test_labels_X3=pd.DataFrame(test_labels_X3)
        df_train_X2_X3=X_2.append(X_3, ignore_index=True)
        #df_train_labels_X1_X2=y_1.append(y_2, ignore_index=True)
        df_train_labels_X2_X3=pd.concat([train_labels_X2,train_labels_X3],ignore_index = True,sort = False)
        df_test_X2_X3=X_2_t.append(X_3_t, ignore_index=True)
        #df_test_labels_X1_X2=y_1_t.append(y_2_t, ignore_index=True)
        df_test_labels_X2_X3=pd.concat([test_labels_X2,test_labels_X3],ignore_index = True,sort = False)
        
        df_train_X2_X3=df_train_X2_X3.to_numpy()
        df_train_labels_X2_X3=df_train_labels_X2_X3.to_numpy()
        df_test_X2_X3=df_test_X2_X3.to_numpy()
        df_test_labels_X2_X3=df_test_labels_X2_X3.to_numpy()
        
        perceptron = Perceptron()                                     #Creating the object of the Perceptron class
        perceptron.fit(df_train_X2_X3, df_train_labels_X2_X3,0,1)           #Training the perceptron mode
        perceptron.test(df_test_X2_X3, df_test_labels_X2_X3,0,1)           #Testing the perceptron mode
        print("Classification 2 Completed")
        print("-------------------------------------------------------- \n\n\n")
        
    if(cfg['part3']['execute_classification_3']==1):
        print("Executing Classification 3 : -")
        print("-------------------------------------------------------- \n\n\n")
        train_labels_X1=[]
        train_labels_X3=[]
        test_labels_X1=[]
        test_labels_X3=[]
        for i in range(0,len(y_1)):
            train_labels_X1.append(0)
        for i in range(0,len(y_3)):
            train_labels_X3.append(1)
        for i in range(0,len(y_1_t)):
            test_labels_X1.append(0)
        for i in range(0,len(y_3_t)):
            test_labels_X3.append(1)
        train_labels_X1=np.array(train_labels_X1)
        train_labels_X3=np.array(train_labels_X3)
        train_labels_X1=pd.DataFrame(train_labels_X1)
        train_labels_X3=pd.DataFrame(train_labels_X3)
        test_labels_X1=pd.DataFrame(test_labels_X1)
        test_labels_X3=pd.DataFrame(test_labels_X3)
        df_train_X1_X3=X_1.append(X_3, ignore_index=True)
        #df_train_labels_X1_X2=y_1.append(y_2, ignore_index=True)
        df_train_labels_X1_X3=pd.concat([train_labels_X1,train_labels_X3],ignore_index = True,sort = False)
        df_test_X1_X3=X_1_t.append(X_3_t, ignore_index=True)
        #df_test_labels_X1_X2=y_1_t.append(y_2_t, ignore_index=True)
        df_test_labels_X1_X3=pd.concat([test_labels_X1,test_labels_X3],ignore_index = True,sort = False)
        
        df_train_X1_X3=df_train_X1_X3.to_numpy()
        df_train_labels_X1_X3=df_train_labels_X1_X3.to_numpy()
        df_test_X1_X3=df_test_X1_X3.to_numpy()
        df_test_labels_X1_X3=df_test_labels_X1_X3.to_numpy()
        
        perceptron = Perceptron()                                                  #Creating the object of the Perceptron class
        perceptron.fit(df_train_X1_X3, df_train_labels_X1_X3,0,1)                        #Training the perceptron model
        perceptron.test(df_test_X1_X3, df_test_labels_X1_X3,0,1)                        #Testing the perceptron model
        print("Classification 3 Completed")
        print("-------------------------------------------------------- \n\n\n")

def one_vs_all_classifier(df_train,df_test,cfg):

    '''In this function we perform Multiclass classification using One vs Rest approach for that the class under observation is labeled as 0 and the Rest are labeled as 1. The data is processed 
    and the Perceptron algorithm is applied.       '''
    
    df_class1,df_class2,df_class3=load_data(df_train)
    df_class1_t,df_class2_t,df_class3_t=load_data(df_test)
    X_1=df_class1.iloc[:,:4]                              #The data and labels of various classes are extracted from the dataframes and stored separately
    y_1=df_class1.iloc[:,4]
    X_2=df_class2.iloc[:,:4]
    y_2=df_class2.iloc[:,4]
    X_3=df_class3.iloc[:,:4]
    y_3=df_class3.iloc[:,4]
    
    X_1_t=df_class1.iloc[:,:4]
    y_1_t=df_class1.iloc[:,4]
    X_2_t=df_class2.iloc[:,:4]
    y_2_t=df_class2.iloc[:,4]
    X_3_t=df_class3.iloc[:,:4]
    y_3_t=df_class3.iloc[:,4]

    y1_all=[]
    y2_all=[]
    y3_all=[]
    y1_all_t=[]
    y2_all_t=[]
    y3_all_t=[]
    avg_one_vs_rest_train=[]
    avg_one_vs_rest_test=[]
    for i in range(0,len(y_1)):           #If Class 1 is part of the Rest then Class 1 is labeled as 1 and the labels are stored in a list.
        y1_all.append(1)
    for i in range(0,len(y_2)):           #If Class 2 is part of the Rest then Class 2 is labeled as 1 and the labels are stored in a list.
        y2_all.append(1)
    for i in range(0,len(y_3)):           #If Class 3 is part of the Rest then Class 3 is labeled as 1 and the labels are stored in a list. 
        y3_all.append(1)

    for i in range(0,len(y_1_t)):         #If Class 1 testing is part of the Rest then Class 1 is labeled as 1 and the labels are stored in a list.
        y1_all_t.append(1)
    for i in range(0,len(y_2_t)):         #If Class 2 testing is part of the Rest then Class 2 is labeled as 1 and the labels are stored in a list.
        y2_all_t.append(1)
    for i in range(0,len(y_3_t)):         #If Class 3 testing is part of the Rest then Class 3 is labeled as 1 and the labels are stored in a list.
        y3_all_t.append(1)
  
    y1_all=pd.DataFrame(y1_all)           #The lists are converted to Dataframes
    y2_all=pd.DataFrame(y2_all)
    y3_all=pd.DataFrame(y3_all)
    y1_all_t=pd.DataFrame(y1_all_t)
    y2_all_t=pd.DataFrame(y2_all_t)
    y3_all_t=pd.DataFrame(y3_all_t)
    
    if(cfg['part4']['execute_classification_1']==1):                                     #This part of the code computes Class 1 vs Rest classification 
        print("Executing Classification 1 : -")
        print("-------------------------------------------------------- \n\n\n")
        y1_mod=[]
        y1_mod_t=[]
        for i in range(0,len(y_1)):                                   #The training labels of Class 1 is set to 0 for One vs Rest classification
            y1_mod.append(0)
        for i in range(0,len(y_1)):                                   #The testing labels of Class 1 is set to 0 for One vs Rest classification 
            y1_mod_t.append(0)
        y1_mod=pd.DataFrame(y1_mod)                                   #The labels are converted to Dataframe
        y1_mod_t=pd.DataFrame(y1_mod_t)
        df_train_X1_all=pd.concat([X_1,X_2,X_3],ignore_index = True,sort = False)                       #The training data is of Class 1 and Rest is combined to form one dataframe
        df_train_labels_X1_all=pd.concat([y1_mod,y2_all,y3_all],ignore_index = True,sort = False)       #The training labels of Class 1 and Rest is combined
        df_test_X1_all=pd.concat([X_1_t,X_2_t,X_3_t],ignore_index = True,sort = False)                  #The testing data is of Class 1 and Rest is combined to form one dataframe
        df_test_labels_X1_all=pd.concat([y1_mod_t,y2_all_t,y3_all_t],ignore_index = True,sort = False)  #The testing labels of Class 1 and Rest is combined

        df_train_X1_all=df_train_X1_all.to_numpy()                                                      #The Dataframes are converted to numpy array
        df_train_labels_X1_all=df_train_labels_X1_all.to_numpy()
        df_test_X1_all=df_test_X1_all.to_numpy()
        df_test_labels_X1_all=df_test_labels_X1_all.to_numpy()

        perceptron = Perceptron()                                              #Creating the object of the Perceptron class
        train_acc=perceptron.fit(df_train_X1_all, df_train_labels_X1_all,0,1)        #Training the perceptron mode
        test_acc=perceptron.test(df_test_X1_all, df_test_labels_X1_all,0,1)         #Testing the perceptron mode
        avg_one_vs_rest_train.append(train_acc)                                                 #To calculate the average over all One vs Rest classifications the accuracy of Class 1 vs Rest is 
                                                                                                #stored
        avg_one_vs_rest_test.append(test_acc)
        print("Classification 1 Completed")
        print("-------------------------------------------------------- \n\n\n")

    if(cfg['part4']['execute_classification_2']==1):                                                          #This part of the code computes Class 2 vs Rest classification 
        print("Executing Classification 2 : -")
        print("-------------------------------------------------------- \n\n\n")
        y2_mod=[]
        y2_mod_t=[]
        for i in range(0,len(y_2)):
            y2_mod.append(0)
        for i in range(0,len(y_2)):
            y2_mod_t.append(0)
        y2_mod=pd.DataFrame(y2_mod)
        y2_mod_t=pd.DataFrame(y2_mod_t)
        df_train_X2_all=pd.concat([X_2,X_1,X_3],ignore_index = True,sort = False)
        df_train_labels_X2_all=pd.concat([y2_mod,y1_all,y3_all],ignore_index = True,sort = False)
        df_test_X2_all=pd.concat([X_2_t,X_1_t,X_3_t],ignore_index = True,sort = False)
        df_test_labels_X2_all=pd.concat([y2_mod_t,y1_all_t,y3_all_t],ignore_index = True,sort = False)   

        df_train_X2_all=df_train_X2_all.to_numpy()
        df_train_labels_X2_all=df_train_labels_X2_all.to_numpy()
        df_test_X2_all=df_test_X2_all.to_numpy()
        df_test_labels_X2_all=df_test_labels_X2_all.to_numpy()

        perceptron = Perceptron()
        train_acc=perceptron.fit(df_train_X2_all, df_train_labels_X2_all,0,1)
        test_acc=perceptron.test(df_test_X2_all, df_test_labels_X2_all,0,1)
        avg_one_vs_rest_train.append(train_acc)
        avg_one_vs_rest_test.append(test_acc)
        print("Classification 2 Completed")
        print("-------------------------------------------------------- \n\n\n")
    if(cfg['part4']['execute_classification_3']==1):                                                     #This part of the code computes Class 3 vs Rest classification 
        print("Executing Classification 3 : -")
        print("-------------------------------------------------------- \n\n\n")
        y3_mod=[]
        y3_mod_t=[]
        for i in range(0,len(y_3)):
            y3_mod.append(0)
        for i in range(0,len(y_2)):
            y3_mod_t.append(0)
        y3_mod=pd.DataFrame(y3_mod)
        y3_mod_t=pd.DataFrame(y3_mod_t)
        df_train_X3_all=pd.concat([X_3,X_1,X_2],ignore_index = True,sort = False)
        df_train_labels_X3_all=pd.concat([y3_mod,y1_all,y2_all],ignore_index = True,sort = False)
        df_test_X3_all=pd.concat([X_3_t,X_1_t,X_2_t],ignore_index = True,sort = False)
        df_test_labels_X3_all=pd.concat([y3_mod_t,y1_all_t,y2_all_t],ignore_index = True,sort = False)   

        df_train_X3_all=df_train_X3_all.to_numpy()
        df_train_labels_X3_all=df_train_labels_X3_all.to_numpy()
        df_test_X3_all=df_test_X3_all.to_numpy()
        df_test_labels_X3_all=df_test_labels_X3_all.to_numpy()

        perceptron = Perceptron()
        train_acc=perceptron.fit(df_train_X3_all, df_train_labels_X3_all,0,1)
        test_acc=perceptron.test(df_test_X3_all, df_test_labels_X3_all,0,1)
        avg_one_vs_rest_train.append(train_acc)
        avg_one_vs_rest_test.append(test_acc)
        print("Classification 3 Completed")
        print("-------------------------------------------------------- \n\n\n")
        print("Average One vs Rest Training accuracy: "+str((sum(avg_one_vs_rest_train)/len(avg_one_vs_rest_train))))                  #Average One vs Rest classification accuracy is calculated 
        print("Average One vs Rest Testing accuracy: "+str((sum(avg_one_vs_rest_test)/len(avg_one_vs_rest_test))))

def one_vs_all_classifier_regularized(df_train,df_test,cfg):
    '''In this funtion we perform One vs Rest classification with L2 Regularization. The function is same as the function above, only the function to train the perceptron is different'''
    df_class1,df_class2,df_class3=load_data(df_train)
    df_class1_t,df_class2_t,df_class3_t=load_data(df_test)
    X_1=df_class1.iloc[:,:4]
    y_1=df_class1.iloc[:,4]
    X_2=df_class2.iloc[:,:4]
    y_2=df_class2.iloc[:,4]
    X_3=df_class3.iloc[:,:4]
    y_3=df_class3.iloc[:,4]
    
    X_1_t=df_class1.iloc[:,:4]
    y_1_t=df_class1.iloc[:,4]
    X_2_t=df_class2.iloc[:,:4]
    y_2_t=df_class2.iloc[:,4]
    X_3_t=df_class3.iloc[:,:4]
    y_3_t=df_class3.iloc[:,4]

    y1_all=[]
    y2_all=[]
    y3_all=[]
    y1_all_t=[]
    y2_all_t=[]
    y3_all_t=[]
    avg_one_vs_rest_train=[]
    avg_one_vs_rest_test=[]
    for i in range(0,len(y_1)):
        y1_all.append(1)
    for i in range(0,len(y_2)):
        y2_all.append(1)
    for i in range(0,len(y_3)):
        y3_all.append(1)

    for i in range(0,len(y_1_t)):
        y1_all_t.append(1)
    for i in range(0,len(y_2_t)):
        y2_all_t.append(1)
    for i in range(0,len(y_3_t)):
        y3_all_t.append(1)
  
    y1_all=pd.DataFrame(y1_all)
    y2_all=pd.DataFrame(y2_all)
    y3_all=pd.DataFrame(y3_all)
    y1_all_t=pd.DataFrame(y1_all_t)
    y2_all_t=pd.DataFrame(y2_all_t)
    y3_all_t=pd.DataFrame(y3_all_t)
    reg_coeff=cfg['part5']['regularization_coefficient']
    if(cfg['part5']['execute_classification_1']==1):
        print("Executing Classification 1 : -")
        print("-------------------------------------------------------- \n\n\n")
        y1_mod=[]
        y1_mod_t=[]
        for i in range(0,len(y_1)):
            y1_mod.append(0)
        for i in range(0,len(y_1)):
            y1_mod_t.append(0)
        y1_mod=pd.DataFrame(y1_mod)
        y1_mod_t=pd.DataFrame(y1_mod_t)
        df_train_X1_all=pd.concat([X_1,X_2,X_3],ignore_index = True,sort = False)
        df_train_labels_X1_all=pd.concat([y1_mod,y2_all,y3_all],ignore_index = True,sort = False)
        df_test_X1_all=pd.concat([X_1_t,X_2_t,X_3_t],ignore_index = True,sort = False)
        df_test_labels_X1_all=pd.concat([y1_mod_t,y2_all_t,y3_all_t],ignore_index = True,sort = False)   

        df_train_X1_all=df_train_X1_all.to_numpy()
        df_train_labels_X1_all=df_train_labels_X1_all.to_numpy()
        df_test_X1_all=df_test_X1_all.to_numpy()
        df_test_labels_X1_all=df_test_labels_X1_all.to_numpy()

        perceptron = Perceptron()
        train_acc=perceptron.fit_regularizer(df_train_X1_all, df_train_labels_X1_all,0,1,reg_coeff)                      #The L2 regularized training of the perceptron
        test_acc=perceptron.test(df_test_X1_all, df_test_labels_X1_all,0,1)
        avg_one_vs_rest_train.append(train_acc)
        avg_one_vs_rest_test.append(test_acc)
        print("Classification 1 Completed")
        print("-------------------------------------------------------- \n\n\n")
    if(cfg['part5']['execute_classification_2']==1):
        print("Executing Classification 2 : -")
        print("-------------------------------------------------------- \n\n\n")
        y2_mod=[]
        y2_mod_t=[]
        for i in range(0,len(y_2)):
            y2_mod.append(0)
        for i in range(0,len(y_2)):
            y2_mod_t.append(0)
        y2_mod=pd.DataFrame(y2_mod)
        y2_mod_t=pd.DataFrame(y2_mod_t)
        df_train_X2_all=pd.concat([X_2,X_1,X_3],ignore_index = True,sort = False)
        df_train_labels_X2_all=pd.concat([y2_mod,y1_all,y3_all],ignore_index = True,sort = False)
        df_test_X2_all=pd.concat([X_2_t,X_1_t,X_3_t],ignore_index = True,sort = False)
        df_test_labels_X2_all=pd.concat([y2_mod_t,y1_all_t,y3_all_t],ignore_index = True,sort = False)   

        df_train_X2_all=df_train_X2_all.to_numpy()
        df_train_labels_X2_all=df_train_labels_X2_all.to_numpy()
        df_test_X2_all=df_test_X2_all.to_numpy()
        df_test_labels_X2_all=df_test_labels_X2_all.to_numpy()

        perceptron = Perceptron()
        train_acc=perceptron.fit_regularizer(df_train_X2_all, df_train_labels_X2_all,0,1,reg_coeff)
        test_acc=perceptron.test(df_test_X2_all, df_test_labels_X2_all,0,1)
        avg_one_vs_rest_train.append(train_acc)
        avg_one_vs_rest_test.append(test_acc)
        print("Classification 2 Completed")
        print("-------------------------------------------------------- \n\n\n")
    if(cfg['part5']['execute_classification_3']==1):
        print("Executing Classification 3 : -")
        print("-------------------------------------------------------- \n\n\n")
        y3_mod=[]
        y3_mod_t=[]
        for i in range(0,len(y_3)):
            y3_mod.append(0)
        for i in range(0,len(y_2)):
            y3_mod_t.append(0)
        y3_mod=pd.DataFrame(y3_mod)
        y3_mod_t=pd.DataFrame(y3_mod_t)
        df_train_X3_all=pd.concat([X_3,X_1,X_2],ignore_index = True,sort = False)
        df_train_labels_X3_all=pd.concat([y3_mod,y1_all,y2_all],ignore_index = True,sort = False)
        df_test_X3_all=pd.concat([X_3_t,X_1_t,X_2_t],ignore_index = True,sort = False)
        df_test_labels_X3_all=pd.concat([y3_mod_t,y1_all_t,y2_all_t],ignore_index = True,sort = False)   

        df_train_X3_all=df_train_X3_all.to_numpy()
        df_train_labels_X3_all=df_train_labels_X3_all.to_numpy()
        df_test_X3_all=df_test_X3_all.to_numpy()
        df_test_labels_X3_all=df_test_labels_X3_all.to_numpy()

        perceptron = Perceptron()
        train_acc=perceptron.fit_regularizer(df_train_X3_all, df_train_labels_X3_all,0,1,reg_coeff)
        test_acc=perceptron.test(df_test_X3_all, df_test_labels_X3_all,0,1)
        avg_one_vs_rest_train.append(train_acc)
        avg_one_vs_rest_test.append(test_acc)
        print("Classification 3 Completed")
        print("-------------------------------------------------------- \n\n\n")
        print("Average One vs Rest Training accuracy: "+str((sum(avg_one_vs_rest_train)/len(avg_one_vs_rest_train))))
        print("Average One vs Rest Testing accuracy: "+str((sum(avg_one_vs_rest_test)/len(avg_one_vs_rest_test))))

if __name__ == "__main__":
    
    df_train=pd.read_csv('train.data',header=None)          #The training data is loaded using a pandas Dataframe
    df_test=pd.read_csv('test.data',header=None)            #The testing data is loaded using a pandas Dataframe
    with open("config_hw2.yaml", 'r') as ymlfile:           #The configuration file to run and instruct the program is loaded
        cfg = yaml.safe_load(ymlfile)
    part3_exec=cfg['part3']['execute']
    part4_exec=cfg['part4']['execute']
    part5_exec=cfg['part5']['execute']
    if(part3_exec==1):                                      #The program starts running following the values in the configuration file                    
        print("Executing Binary Classification: -")
        print("---------------------------------------------------------------------------- \n\n\n")
        binary_classifier(df_train,df_test,cfg)
        print("Binary Classification Completed")
        print("---------------------------------------------------------------------------- \n\n\n")
    if(part4_exec==1):
        print("Executing One vs Rest Classification: -")
        print("---------------------------------------------------------------------------- \n\n\n")
        one_vs_all_classifier(df_train,df_test,cfg)
        print("One vs Rest Classification Completed")
        print("---------------------------------------------------------------------------- \n\n\n")
    if(part5_exec==1):
        print("Executing One vs Rest Classification with L2 Regularization: -")
        print("---------------------------------------------------------------------------- \n\n\n")
        one_vs_all_classifier_regularized(df_train,df_test,cfg)
        print("One vs Rest Classification with L2 Regularization Completed")
        print("---------------------------------------------------------------------------- \n\n\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
