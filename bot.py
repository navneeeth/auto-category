from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
import csv
from sklearn.datasets import load_iris
#filepath = input("Enter the path of the file")
#categorical=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
#scalable=['Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
#
filepath="C:\\Users\\Navneeth\\Desktop\\loan_prediction\\"
filez =input("Enter your file path")
#filez=load_iris()
df =read_csv(filez)
j=0
lines = csv.reader(open(filez, 'r', errors='ignore'), delimiter=',')
num_cols = len(next(lines))
#print(num_cols)
categorical=[]
scalable=[]
with open(filez, 'r') as f: #using this to access only the headings of the csv file
    for row in f:
        print(row)
        while(j<num_cols):
            a=0
            b=0
            c=0 #initialising inside the while so that it repeats becoming 0 after moving to a different column each time
            d=0
            if(j==num_cols-1):
                second_data = row.split(',')[j]
                second_data=second_data[:len(second_data)-1]
                #print(second_data)
                for i in df[second_data]:
                    if(i==''):
                        continue
                    if(type(i)==str and i.replace(" ", "").isalpha() or (str(i).isnumeric() and int(i)<5)):
                        c=c+1
                        if(c==50):
                            categorical.append(second_data)

                    elif(str(i).replace(".", "").isnumeric()): #replacing the point since isnumeric returns false for floating point
                        a=a+1
                        if(a==1):
                            scalable.append(second_data)
                    else:
                        continue
                #print(X_test[second_data])
            else:
                second_data = row.split(',')[j]
                #print(second_data)
                for i in df[second_data]:
                    if(i==''):
                        continue
                    if(type(i)==str and i.replace(" ", "").isalpha() or (str(i).isnumeric() and int(i)<5)):
                        d=d+1
                        if(d==50):
                            categorical.append(second_data)
                    elif(str(i).replace(".", "").isnumeric()):
                        b=b+1
                        if(b==1):
                            scalable.append(second_data)
                    else:
                        continue

                #print(X_test[second_data])
            j=j+1
print(scalable)
print("\n")
print(categorical)
Y = df[categorical[len(categorical)-1]]
X = df.drop(categorical[len(categorical)-1],axis = 1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in X.columns:
    X[i]=le.fit_transform(X[i])
Y = le.fit_transform(Y)
print(X,Y)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dec_tree
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
# Decision Tree
dsc = dec_tree()
dsc.fit(X_train,Y_train)
Y_predicted = dsc.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test,Y_predicted)
print(acc)
x = input("enter the value you want to check:")
x = x.split()
print(dsc.predict([x]))
# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
print(lr.coef_)
print(len(lr.coef_),lr.intercept_)
X_plot = X_train.iloc[:, 1]
Y_plot = X_plot*lr.coef_[0] + lr.intercept_
Y_predicted = lr.predict(X_test)
for i,j in zip(Y_predicted,Y_test):
    print(i,j)
from sklearn.metrics import mean_squared_error as msq
print("MSQ="+str(msq(Y_test,Y_predicted)))
from matplotlib import pyplot as plt
plt.plot(X_plot,Y_plot)
plt.scatter(Y_test,Y_predicted)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
# knn
from sklearn.neighbors import KNeighborsClassifier as knn
knn_model = knn(n_neighbors=10)
knn_model.fit(X_train,Y_train)
Y_predicted = knn_model.predict(X_test)
correct_output=0
for i,j in zip(Y_test,Y_predicted):
    print(i,j)
    if i==j:
        correct_output+=1

print("number of correctly predicted output ="+ str(correct_output))
print("number of wrongly predicted output =" +str(len(Y_test)-correct_output))
acc = correct_output/len(Y_test)
print("accurate values="+str(acc))

# KMeans clustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
X,Y = make_blobs(n_samples=1000, n_features=2,centers=5)
kmeans = KMeans(n_clusters=5)
kmeans.fit(X_train)
Y_predicted = kmeans.predict(X_test)
print(kmeans.cluster_centers_)
print(Y_predicted)
print(Y_test)
from matplotlib import pyplot as plt
plt.scatter(X[:,0],X[:,1],edgecolors="black")
plt.show()

#SVC
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train,Y_train)
Y_prediction = svm_model.predict(X_test)
for i,j in zip(Y_prediction,Y_test):
    print(i,j)
svmaccuracy=str(accuracy_score(Y_test,Y_prediction))    
print("Accuracy ="+str(accuracy_score(Y_test,Y_prediction)))

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('KNN', knn()))
models.append(('DSC', dec_tree()))
models.append(('KMC',KMeans()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
from sklearn import model_selection
for name, model in models:
   kfold = model_selection.KFold(n_splits=10, random_state=seed)
   cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   if(model=='KNN'):
       knn_accuracy=cv_results.mean()
   elif(model=='DSC'):
        dsc_accuracy=cv_results.mean()
   elif(model=='SVM'):
        svm_accuracy=cv_results.mean()
   else:
       continue
   print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
if(knn_accuracy>dsc_accuracy):
    if(knn_accuracy>svm_accuracy):
        print("KNN is the best algorithm for this set of data.")
    else:
        print("SVM is the best algorithm for this set of data.")
elif(dsc_accuracy>svm_accuracy):
    print("DSC is the best algorithm for this set of data.")
else:
    print("SVM is the best algorithm for this set of data.")
        
