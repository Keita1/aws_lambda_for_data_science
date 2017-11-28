from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import json
import pickle

print("Load digits dataset.")
digits = datasets.load_digits()

print("Plotting sample digits")
f, axarr = plt.subplots(2, 5, figsize=(5,2))
for i in range(10):
    ax = axarr[int(i/5), i%5]
    ax.imshow(digits.images[i], cmap=plt.cm.gray_r)
    ax.set_title(i)
    ax.axis('off')
plt.tight_layout()
plt.show()

print("Performing test-train split")
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

print("Sample digit")
print(json.dumps(X_test[0].astype(int).tolist()))

print("Fitting a multi-class Logistic Regression classifier.")
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)

print("Train accuracy: {:.2%}".format(model.score(X_train, y_train)))
print("Test accuracy: {:.2%}".format(model.score(X_test, y_test)))

model.predict([X_test[0]])

print("Dumping the model as a Pickle file")
pickle.dump(model, open("model.pkl", 'wb'))
