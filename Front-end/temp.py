import pickle

def sum(a, b):
    return a+b

pickle.dump(sum, open('sum.pkl', 'wb'))

