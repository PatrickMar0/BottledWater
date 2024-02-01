# SVM using PCAs
# Patrick Marino


import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = ""

def main():
    # Principal Components Anaylsis
    print("\n---------------------------------")
    print("          PCA Program            ")
    print("---------------------------------")
    print("Please be sure that your training\ndata, testing data, and labels are\nwithin the same folder as this\nprogram.")
    print("---------------------------------")
    
    # Reads csv files and returns training data, testing data, and labels
    training, labels = find_data()
    testing = find_testing()

    # CHANGE THIS FILE TO ADD NEW POTENTIAL LABELS AND THEIR COLORS ON THE SCATTERPLOT!
    colordict = {
        "Zephyrhills":"blue",
        "Dasani":"red",
        "Evian":"green",
        "FIJI":"yellow",
        "Aquafina":"pink"
    }

    # Calculates pca and uses an svm to plot into clusters
    my_pca(training, labels, testing, colordict)



#Locates and loads the data from csv files from within its folder
def find_data():

    
    while True:

        print("\nEnter the file name of the training dataset (default is 'bottledwater.csv'): ")

        # Stores entry from user of potential csv training data 
        searchx = str(input())

        print("\nSearching for {}...".format(searchx))

        # This chunk of code attempts to load the given search term into a training variable.
        # If not user entry is tried again
        try:
            xlabel = np.loadtxt(searchx,delimiter=",",skiprows=1)
            print("{} found!".format(searchx))
            break
        except FileNotFoundError:
            print("File {} not found! Make sure to enter the file name correctly".format(searchx))
    

    
    while True:
        print("\nEnter the file name of the labels dataset (default is 'bottledwaterlabels.csv'): ")

        # Stores entry from user of potential csv label data 
        searchy = str(input())
        
        print("\nSearching for {}...".format(searchy))

        # This chunk of code attempts to load the given search term into a label variable.
        # If not user entry is tried again
        try:
            ylabel = np.loadtxt(searchy, delimiter=",",dtype=str)
            print("{} found!".format(searchy))
            break
        except FileNotFoundError:
            print("File {} not found! Make sure to enter the file name correctly".format(searchy))

    
    # Returns the training data and its labels back into main()
    return xlabel, ylabel

# This method works similarly to find_data(), in regards to finding and locating data for our testing dataset
def find_testing():

    while True:
        print("\nEnter the file name of the testing dataset (default is 'testsamples.csv'): ")

        # Stores entry from user of potential csv testing data 
        searchTest = str(input())

        print("\nSearching for {}...".format(searchTest))

        # This chunk of code attempts to load the given search term into a testing variable.
        # If not user entry is tried again
        try:
            test = np.loadtxt(searchTest,delimiter=",",skiprows=1)
            print("{} found!".format(searchTest))
            break
        except FileNotFoundError:
            print("File {} not found! Make sure to enter the file name correctly".format(searchTest))

    # Returns testing data into main()
    return test


def my_pca(training, labels, testing, colordict):

    
    print("\nCalculating Scatter Matrix...")

    # mu is calculated
    mu = training.mean(axis=0)

    # Scatter matrix is calculated below
    diff = training-mu
    scatter = np.matmul(diff, diff.T)

    print("Scatter Matrix: \n{}".format(scatter))

    print("\nCalculating eigenvalues and eigenvectors (this may take a few seconds)...")

    # Eigenvalues and eigenvectors are calculated and stored 
    eigVal, eigVec = np.linalg.eigh(scatter)
    ind = np.argsort(eigVal)[::-1]
    eigVal = eigVal[ind]
    eigVec = eigVec[:, ind]

    print("Eigenvalues: \n{}".format(eigVal))
    print("Eigenvectors: \n{}".format(eigVec))

    print("\nCalculating the number of components...")

    # Threshold is used to represent our covariance
    threshold = 0.9

    # The number of principal components are calculated using threshold and our eigenvalues below
    n = np.where(np.cumsum(eigVal) / np.sum(eigVal) >= threshold)[0][0] + 1
    
    print("There is {} principal components!".format(n))
    

    print("\nRunning PCAs on training and testing data...")
    
    # We design and fit a pca using our testing and training data
    pca = PCA(n_components=n, random_state=4000)
    pca.fit(training)
    xtr = pca.transform(training)
    xte = pca.transform(testing)

    print("Fitting data into SVM...")

    # SVM is created to fit our principal components into
    clf = SVC(kernel="linear")
    clf.fit(xtr,labels)

    # Below the training data is tested and accuracy is calculated
    print("\n------------------------------------")
    print("Testing accuracy of training dataset")
    print("Accuracy: {}%".format(clf.score(xtr, labels)*100))
    print("\nPredicted results of the test:")
    print(clf.predict(xte))

    # Returns the number of support vectors for each class
    print("\nSupport Vectors")
    print("---------------")
    print("\nNumber of SVMs per class: \n")
    for i in range(len(clf.classes_)):
        print("| ",clf.classes_[i],": ",clf.n_support_[i])
    
    # Figure 1: Returns the data points, the support vectors, and the testing data points 
    plt.figure(1)
    plt.scatter(xtr[:,0],xtr[:,1],s=30,c="blue",edgecolors="black",label="Training Data")
    plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1], color="yellow",s=10,edgecolors="black", label="Support Vectors")
    plt.scatter(xte[:,0],xte[:,1],s=40,c="red",edgecolors="black", label="Testing Data")
    plt.title("Support Vectors")

    # Legend handles the key of the graph
    plt.legend()
    plt.show(block=False)

    # custom_points returns the color and label in a presentable form for figure 2
    patches = custom_points(colordict)

    # Figure 2: Returns the data points and their clustering classifications
    plt.figure(2)

    # For each dataset we individually plot the points and assign them their given color to match their classification
    for i in range(len(labels)):
        plt.plot(xtr[i,0],xtr[i,1],'bo',c=colordict[labels[i]],label=labels[i],mec="black",alpha=0.5,linewidth=1)

    # Legend handles the key of the graph
    plt.legend(handles=patches)
    plt.title("SVM Clusters of PCA")
    plt.show(block=False)


    pdb.set_trace()


# This method takes our color dictionary and creates them into patches to be used by matplotlib
def custom_points(colordict):

    # Classes and colors store the dictionaries values
    classes = list(colordict.keys())
    colors = list(colordict.values())
    patches = []

    # Appends each color and class into an array of patches
    for i in range(len(classes)):
        patches.append(mpatches.Patch(color=colors[i],label=classes[i]))

    # Returns array of patches back into my_pca()
    return patches
    
    



if __name__ == '__main__':
    main()