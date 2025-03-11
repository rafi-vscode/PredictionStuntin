import pickle
with open('model_rf_smote.pkl', 'rb') as f:
    model = pickle.load(f)
print(type(model))