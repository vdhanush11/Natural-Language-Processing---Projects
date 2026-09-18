# Lifecycle stage 5 — Model Building
 
from sklearn.naive_bayes import MultinomialNB
 
 
def build_model():
 
    # alpha = Laplace smoothing; keeps unseen words from zeroing a class
    return MultinomialNB(alpha=1.0)
