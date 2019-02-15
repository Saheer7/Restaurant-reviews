import matplotlib
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
#nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from pandas import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

root=Tk()
global f1
def browsefunc():
    root.fileName = filedialog.askopenfilename(filetypes=(("how code files", ".hc"), ("All files", "*.*")))
    con = sqlite3.connect(root.fileName)
    f1 = root.fileName
    pathlabel.config(text=f1)
    print(f1)
    messages = pd.read_sql_query("""SELECT Score, Summary FROM Reviews WHERE Score != 3""", con)

    def partition(x):
        if x < 3:
           return 'negative'
        return 'positive'

    Score = messages['Score']
    Score = Score.map(partition)
    Summary = messages['Summary']
    X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=0.2, random_state=50)



    stemmer = PorterStemmer()



    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed


    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        # tokens = [word for word in tokens if word not in stopwords.words('english')]
        stems = stem_tokens(tokens, stemmer)
        return ' '.join(stems)


    intab = string.punctuation
    outtab = "                                "
    trantab = str.maketrans(intab, outtab)

    # --- Training set

    corpus = []
    for text in X_train:
        text = text.lower()
        text = text.translate(trantab)
        text = tokenize(text)
        corpus.append(text)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(corpus)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # --- Test set

    test_set = []
    for text in X_test:
        text = text.lower()
        text = text.translate(trantab)
        text = tokenize(text)
        test_set.append(text)

    X_new_counts = count_vect.transform(test_set)
    X_test_tfidf = tfidf_transformer.transform(X_new_counts)



    df = DataFrame({'Before': X_train, 'After': corpus})
    print(df.head(20))

    prediction = dict()


    model = MultinomialNB().fit(X_train_tfidf, y_train)
    prediction['Multinomial'] = model.predict(X_test_tfidf)


    model = BernoulliNB().fit(X_train_tfidf, y_train)
    prediction['Bernoulli'] = model.predict(X_test_tfidf)


    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train_tfidf, y_train)
    prediction['Logistic'] = logreg.predict(X_test_tfidf)

    def formatt(x):
        if x == 'negative':
            return 0
        return 1
    vfunc = np.vectorize(formatt)

    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k']
    for model, predicted in prediction.items():
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
        cmp += 1

    plt.title('Classifiers comparaison')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print(metrics.classification_report(y_test, prediction['Logistic'], target_names = ["negative", "positive"]))
    #print(metrics.f1_score(y_test,prediction['Logistic'], average=None))
    #print(metrics.precision_recall_fscore_support(y_test, prediction['Logistic'], average='weighted'))
    a=confusion_matrix(y_test,prediction['Logistic'])

    print(confusion_matrix(y_test, prediction['Logistic']))
    global s
    global c
    global d
    global e
    s=a[0][0]
    c=a[0][1]
    d=a[1][0]
    e=a[1][1]
    print(a[0][0])
    print(a[0][1])
    print(a[1][0])
    print(a[1][1])

def callback():
    plt.show()





root.title("Data Analysis")
image = Image.open('D:\\Project\\new1.png')
photo_image = ImageTk.PhotoImage(image)
label = tk.Label(root, image = photo_image)
label.config(height=5000, width=2000)
label.pack()

#theLabel=Label(root,text='Data Analysis on Restaurant Reviews',fg='green')
#theLabel.config(font=("Geometric Sans Serif", 30))
#theLabel.pack()




def popupBonus():
    toplevel = Toplevel()
    f1=root.fileName
    print(f1)
    if f1 == 'D:/Project/Suraj.sqlite':
        x = (((s + e) * 10) / (s + c + d + e)) / 2
        v = round(x, 2)
        var = StringVar()
        var.set(v)

    elif f1 == 'D:/Project/dvr.sqlite':
        x = (((s+e) * 10) / (s+c+d+e)) / 2
        v = round(x, 2)
        var = StringVar()
        var.set(v)

    elif f1 == 'D:/Project/food world.sqlite':
        x = (((s + e) * 10) / (s + c + d + e)) / 2
        v = round(x, 2)
        var = StringVar()
        var.set(v)

    elif f1 == 'D:/Project/Krithunga.sqlite':
        x = (((s + e) * 10) / (s + c + d + e)) / 2
        v = round(x, 2)
        var = StringVar()
        var.set(v)

    elif f1 == 'D:/Project/mouryainn.sqlite':
        x = (((s + e) * 10) / (s + c + d + e)) / 2
        v = round(x, 2)
        var = StringVar()
        var.set(v)

    elif f1 == 'D:/Project/sasya.sqlite':
        x = (((s + e) * 10) / (s + c + d + e)) / 2
        v = round(x, 2)
        var = StringVar()
        var.set(v)
    rating = '''Rating'''
    b = ''''''

    label1 = Label(toplevel,text=rating)
    label1.config(font=("Geometric Sans Serif", 40))
    label1.place(x=200, y=79)
    label1.pack()
    label2 = Label(toplevel, textvariable=var)
    label2.config(font=("Geometric Sans Serif", 40))
    label2.place(x=220, y=79)
    label2.pack()
    label3 = Label(toplevel, text=b, height=0, width=210)
    label3.config(font=("Geometric Sans Serif", 40))
    label3.pack()


    #popupBonusWindow = tk.Tk()
    #popupBonusWindow.wm_title("Window")
    #labelBonus = Label(root, text=rating)


    #labelBonus.grid(row=0, column=0)

browsebutton = Button(root, text="Browse", command=browsefunc,bd=5,background='wheat4')
browsebutton.place(x=60,y=190)
browsebutton.config(font=("comic sans ms",18,'italic bold'))

pathlabel = Label(root,background='wheat4')
pathlabel.place(x=200,y=200)
pathlabel.config(font=("comic sans ms",18,'italic bold'))

ratingbutton= Button(root, text="Rating", command=popupBonus,background='wheat4',bd=5)
ratingbutton.place(x=60,y=320)
ratingbutton.config(font=("comic sans ms",18,'italic bold'))

accuracybutton=Button(root, text="Accuracy", command=callback,background='wheat4',bd=5)
accuracybutton.place(x=250,y=320)
accuracybutton.config(font=("comic sans ms",18,'italic bold'))

root.mainloop()




#def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
 #   plt.imshow(cm, interpolation='nearest', cmap=cmap)
  #  plt.title(title)
   # plt.colorbar()
    #tick_marks = np.arange(len(set(Score)))
    #plt.xticks(tick_marks, set(Score), rotation=45)
    #plt.yticks(tick_marks, set(Score))
    #plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')


# Compute confusion matrix

#cm = confusion_matrix(y_test, prediction['Logistic'])
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(cm)

#cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#plt.figure()
#plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
#plt.show()