# Lifecycle stage 5 — Model Building
 
from sklearn.svm import LinearSVC
 
 
def build_model():
 
    # C = regularization strength; one-vs-rest handles all categories
    return LinearSVC(C=1.0)
