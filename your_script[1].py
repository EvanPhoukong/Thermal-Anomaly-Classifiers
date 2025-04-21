import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV  # Import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  # Import KNN classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Import accuracy score
from sklearn.preprocessing import LabelEncoder  # Encoding the target variable
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import glob 
import os
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import seaborn as sb
import matplotlib.pyplot as plt

def EDA():

    folder = os.getcwd()  #Get the current working directory/folder
    csv_paths = glob.glob(f'{folder}/*.csv') #Obtain all csv file paths and store them in a list

    #Below lines reads csv into dataframe, ignoring their headers(in case they are incorrect or missing)
    #and using the header I have defined.
    headers = "filename,xcenter,ycenter,area,eccentricity,angle,perimeter,Hue,Saturation,Class,Temperature".split(',')
    csv_files = [pd.read_csv(path, header=None, names=headers) for path in csv_paths]
    unified_data = pd.concat(csv_files, ignore_index = True) #Combine all csvs files into one dataframe, inoring indices
    
    unified_data.to_csv("unified_data.csv", index = False, header=True) #Convert dataframe to a csv
    data = pd.read_csv("unified_data.csv") #Read the csv into a dataframe
    
    numeric_cols = "xcenter,ycenter,area,eccentricity,angle,perimeter,Hue,Saturation,Temperature".split(',')
    for col in numeric_cols: #For columns containing numeric values, this ensures all values are numeric
        data[col] = pd.to_numeric(data[col], errors='coerce') #Non-numeric valeus replaced with NaN

    #Checks every value in class to make sure it is a proper lablel. If not, replace the value with NaN
    data['Class'] = data['Class'].apply(lambda x: x.lower() if x in ['car engine', 'head', 'arm', 'torso', 'leg', 'noise', 'tire'] else np.nan)

    data = data.replace("", np.nan) #Replace empty string with NaNs values so they can be dropped with dropna()

    data = data.dropna() #Drop NaNs

    data = data.drop('filename', axis=1) #Drop filename column

    data = data.drop_duplicates() #Drop duplicate rows

    target = data.columns[-2] #Get the target variable, in this case, "Class"
    
    #print(data['Class'].value_counts()) - This was used to show imbalances among the diff classes.

    #Crete Label Encoder
    lb = LabelEncoder()
    #Encode "Class" values so they are numerical
    data[target] = lb.fit_transform(data[target])
    #Make correlation matrix for the data
    corr_matrix = data.corr()
    plt.figure(figsize=(10,16))
    sb.heatmap(corr_matrix, annot=True, cmap="Blues", square=True)
    plt.title('Correlation Matrix')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.show()
    print("Correlation Matrix : \n", corr_matrix , "\n")
    #Select the independent variables that have a strong correlation to "Class"
    features = list(corr_matrix[abs(corr_matrix["Class"]) >= 0.65])
    #Remove "Class" from the features list
    features.remove("Class")

    return data, target, features, lb

def train_and_evaluate_knn(data, target, features, test_size, random_state, le):

    X = data[features] #Store data columns for features
    Y = data[target] #Store "Class" column

    #Resample the data and make the class labels evenly distributed.
    smote = SMOTE() 
    X, Y = smote.fit_resample(X, Y)
    #print(Y.value_counts())

    #Determines necessary statistcs to scale the independent variables, then scales X accordingly
    scaler = StandardScaler()
    scaler.fit(X)
    Xscale = scaler.transform(X)

    #Get the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(Xscale, Y, test_size=test_size, random_state=random_state)

    #Create the KNN classifier
    knn = KNeighborsClassifier()
    
    #Determine what hyperparameters we would like to test/use on KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9], # Number of neighbors around an observation point to consider
        'weights': ['uniform', 'distance'], # Whether distances are weighted differently or not
        'metric': ['euclidean', 'manhattan', 'minkowski'] # What distance metric to use
    }

    print("KNN Results: \n")
    print("Hyperparameters: \n", param_grid, "\n")
    performGridSearch(knn, param_grid, 5, X_train, X_test, y_train, y_test, le, "KNN")
    print()

def train_and_evaluate_SVM(data, target, features, test_size, random_state, le):

    X = data[features]
    Y = data[target]

    smote = SMOTE()
    X, Y = smote.fit_resample(X, Y)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    #Create SVM classifier object
    SVM = SVC()

    #Determine what hyperparameters we would like to test/use on SVM
    param_grid = {
        'C': [0.1, 1], # Regularization Paramter - margin maximization vs. misclassification tradeoff
        'kernel': ['linear', 'rbf'], # Specifies kernel type that ill be used as the hyperplane
        'gamma': ['scale', 0.1] # Controls how specific or general the the model will fit (for non-linear kernels)
    }

    print("SVM Results: \n")
    print("Hyperparameters: \n", param_grid, "\n")
    performGridSearch(SVM, param_grid, 2, X_train, X_test, y_train, y_test, le, "SVM")
    print()

def train_and_evaluate_DT(data, target, features, test_size, random_state, le):
    
    X = data[features]
    Y = data[target]

    smote = SMOTE()
    X, Y = smote.fit_resample(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    #Create Decision Tree Classifier Object
    DT = DecisionTreeClassifier()

    #Determine what hyperparameters we would like to test/use on DT
    param_grid = {
    'criterion': ['gini', 'entropy'],  #Metric used to determine how to split
    'max_depth': [5, 10, None],  # Max depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples necessary for a node to be able to split
    'min_samples_leaf': [1, 2, 5]  # Minimum samples to be at a leaf node
    }

    print("Decision Tree Results: \n")
    print("Hyperparameters: \n", param_grid, "\n")
    performGridSearch(DT, param_grid, 5, X_train, X_test, y_train, y_test, le, "Decision Tree")
    print()


def train_and_evaluate_NB(data, target, features, test_size, random_state, le):
    
    X = data[features]
    Y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    #Create Guassian Naive Bayes Classifier
    NB = GaussianNB()

    #Determine what hyperparameters we would like to test/use on NB
    param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5] #Number added to variances to increase numerical stability
    }

    print("Naive Bayes Results: \n")
    print("Hyperparameters: \n", param_grid, "\n")
    performGridSearch(NB, param_grid, 5, X_train, X_test, y_train, y_test, le, "Naive Bayes")
    print()

def train_and_evaluate_RF(data, target, features, test_size, random_state, le):
    
    X = data[features]
    Y = data[target]

    smote = SMOTE()
    X, Y = smote.fit_resample(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    #Determine what hyperparameters we would like to test/use on Random Forests
    RF = RandomForestClassifier()

    param_grid = {
    'n_estimators': [50, 100],  # Number of trees to make
    'max_depth': [None, 10],  # Max depth of a tree
    'min_samples_split': [2, 5]  # Mimimum samples needed for a node to split
    }

    print("Random Forests Results: \n")
    print("Hyperparameters: \n", param_grid, "\n")
    performGridSearch(RF, param_grid, 2, X_train, X_test, y_train, y_test, le, "Random Forest")
    print()

def performGridSearch(model, params, cv_val, X_train, X_test, y_train, y_test, le, name):

    #Perform Grid Search using hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv_val, scoring="accuracy")

    #Train grid_search model and pick the best one
    grid_search.fit(X_train, y_train)
    best = grid_search.best_estimator_

    #Use the best model to make predictions
    y_pred = best.predict(X_test)

    #Get the original, uncoded values of y_pred
    y_pred_original = le.inverse_transform(y_pred)

    # result_df = pd.DataFrame({'Predicted': y_pred_original, 'Actual': le.inverse_transform(y_test)})
    # for index, row in result_df.iterrows():
    #     match_status = "Match" if row['Predicted'] == row['Actual'] else "Mismatch"
    #     print(f"Row {index}: Predicted = {row['Predicted']}, Actual = {row['Actual']}, {match_status}")

    print(f"Best Parameters: {grid_search.best_params_}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Precision: {precision:.2f}")
    
    recall = recall_score(y_test, y_pred, average='weighted',  zero_division=0)
    print(f"Recall: {recall:.2f}")

    f1 = f1_score(y_test, y_pred, average='weighted',  zero_division=0)
    print(f"F1_Score: {f1:.2f}")

    y_test = le.inverse_transform(y_test)
    cm = confusion_matrix(y_test, y_pred_original)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['head', 'torso', 'arm', 'leg', 'car engine', 'tire', 'noise'])
    display.plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
    print("Confusion Matrix: \n", cm, "\n")
    

def main():
    data, target, features, le = EDA()
    train_and_evaluate_knn(data, target, features, test_size=0.3, random_state=42, le=le)
    train_and_evaluate_SVM(data, target, features, test_size=0.3, random_state=42, le=le)
    train_and_evaluate_DT(data, target, features, test_size=0.3, random_state=42, le=le)
    train_and_evaluate_NB(data, target, features, test_size=0.3, random_state=42, le=le)
    train_and_evaluate_RF(data, target, features, test_size=0.3, random_state=42, le=le)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()