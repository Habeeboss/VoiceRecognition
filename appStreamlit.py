import streamlit as st
import keras
import numpy as np
import librosa
import joblib
import random

def process_audio_file(y):
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    
    # Root Mean Square Energy
    rmse = librosa.feature.rms(y=y).mean()
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis = 1).flatten()
    return np.concatenate([np.array([zcr]), np.array([rmse]), np.array([spectral_centroid]), np.array([spectral_bandwidth]), np.array([spectral_rolloff]), mfccs], axis = 0).flatten()


# Sample models dictionary for illustration
modelsNames = {
    "model1": "NONE",
    "model2": "ANN",
    "model3": "CNN",
    "model4": "RNN",
    "model5": "KNN"
}

speaker_dict = {
    0: "AbdulHamid",
    1: "Abdurrafiu",
    2: "Abubakar",
    3: "Aisha",
    4: "Clement",
    5: "Clinton",
    6: "Comfort",
    7: "Faizat",
    8: "Faridah",
    9: "Faruq",
    10: "Feyi",
    11: "Gideon",
    12: "Happiness",
    13: "Ifeoluwa",
    14: "Jomiloju",
    15: "Kabirat",
    16: "Kenny",
    17: "Maliqah",
    18: "Masturoh",
    19: "Michopat",
    20: "Naimat",
    21: "Nife",
    22: "Nonso",
    23: "Olumide",
    24: "Precious",
    25: "Saheed",
    26: "Shukrah",
    27: "Sunday ",
    28: "Tekenah ",
    29: "Yessy",
    30: "Unknown"
}

modelAnn = keras.models.load_model('ANN MODEL.keras')
modelCnn = keras.models.load_model('CNN MODEL.keras')
modelRnn = keras.models.load_model('RNN MODEL.keras')
modelKnn = joblib.load('KNN MODEL.joblib')

scalerANN = joblib.load('ANN SCALER.joblib')
scalerCNN = joblib.load('CNN SCALER.joblib')
scalerRNN = joblib.load('RNN SCALER.joblib')
scalerKNN = joblib.load('KNN SCALER.joblib')


models ={
    'ANN': modelAnn,
    'CNN': modelCnn,
    'RNN': modelRnn,
    'KNN': modelKnn
    }

# Predict function (mock function to simulate prediction)
def predict_with_model(audio_data, model_name):
    X = process_audio_file(audio_data).reshape(1,-1)
    if model_name=='NONE':
        model_name = random.choice(list(models.keys()))
        if model_name=='ANN': return f"{speaker_dict[models[model_name].predict(scalerANN.transform(X)).argmax()]} by {model_name}"
        elif model_name=='CNN': return f"{speaker_dict[models[model_name].predict(scalerCNN.transform(X)).argmax()]} by {model_name}"
        elif model_name == 'KNN': return f"{speaker_dict[models['KNN'].predict(scalerKNN.transform(X)).tolist()[0]]} by KNN"
        else: return f"{speaker_dict[models['RNN'].predict(scalerRNN.transform(X).reshape(1,1,-1)).argmax()]} by RNN"
        # return output
    else:
        if model_name=='ANN': return models[model_name].predict(scalerANN.transform(X)).argmax()
        elif model_name=='CNN': return models[model_name].predict(scalerCNN.transform(X)).argmax()
        elif model_name == 'KNN': return models['KNN'].predict(scalerKNN.transform(X)).tolist()[0]
        else: return models['RNN'].predict(scalerRNN.transform(X).reshape(1,1,-1)).argmax()

# Streamlit App UI
st.title("Audio Classifier")

# File upload section
st.subheader("Upload an Audio File")
audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

# Model selection section
st.subheader("Select a Model")
model_name = st.radio("Choose a model for prediction", ['NONE', 'ANN', 'CNN', 'RNN', 'KNN'])

if audio_file is not None and model_name is not None:
    y, sr = librosa.load(audio_file, mono=True, duration=30)
    st.audio(y, format=audio_file.name.split('.')[-1], sample_rate = sr, autoplay = False, loop = False)


# Prediction section
if st.button("Classify Audio"):
    if audio_file is not None and model_name is not None:
        # Run prediction
        prediction = predict_with_model(y, model_name)

        # Display result
        if isinstance(prediction, str): st.success(f"Voice is recognized: {prediction}")
        else:  st.success(f"Voice is recognized: {speaker_dict[prediction]}")

        # Optionally play the audio file
    else:
        st.error("Please upload a file and select a model.")
