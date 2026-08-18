import joblib

#Load the model for startup
pipeline = joblib.load("model.pkl")

# The model remembers how many features it was trained on 

print("Feature the model expects : ", pipeline.named_steps['model'].n_features_in_)
