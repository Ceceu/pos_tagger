from sklearn.metrics import classification_report

y_true = [3, 3, 1, 1, 2, 4]
y_pred = [3, 3, 1, 4, 2, 1]
target_names = ["c", "a", "b", "d"]
print(classification_report(y_true, y_pred))

import nltk
nltk.download('universal_tagset')
