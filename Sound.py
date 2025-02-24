import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

audio_dataset_path = r"C:\Users\sohai\Downloads\Dataset\musan"
metadata_file = r"C:\Users\sohai\Downloads\Dataset\combined_dataset.csv"

metadata = pd.read_csv(metadata_file)

class_map = {
    'air_conditioner': 'Noise',
    'car_horn': 'Music',
    'children_playing': 'Speech',
    'dog_bark': 'Speech',
    'drilling': 'Noise',
    'engine_idling': 'Noise',
    'gun_shot': 'Noise',
    'jackhammer': 'Noise',
    'siren': 'Noise',
    'street_music': 'Music',
    'speech': 'Speech',
    'noise': 'Noise',
    'music': 'Music'
}

metadata['category'] = metadata['category'].map(class_map)

def apply_augmentation(audio, sample_rate):
    # Apply random time shifting
    if np.random.rand() < 0.5:
        shift = np.random.randint(sample_rate // 10)
        audio = np.roll(audio, shift)
    
    # Apply random pitch shifting
    if np.random.rand() < 0.5:
        pitch_shift = np.random.randint(-5, 5) 
        audio = librosa.effects.pitch_shift(audio, sample_rate, pitch_shift)
    
    # Apply random speed tuning
    if np.random.rand() < 0.5:
        speed_change = np.random.uniform(0.9, 1.1)  
        audio = librosa.effects.time_stretch(audio, speed_change)
    
    # Apply random noise addition
    if np.random.rand() < 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
        audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
    
    return audio
def feature_extractor(filename, max_pad_len=300):
    # Load the audio file
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast', duration=6.96)

    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100)
    
    pad_width = max_pad_len - mfccs_features.shape[1]
    if pad_width > 0:  
        mfccs_features = np.pad(mfccs_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return mfccs_features
extracted_features = []

for index_num, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    file_name = row['filepath'] 
    final_class_labels = row['category']  

    data = feature_extractor(file_name)
    
    extracted_features.append([data, final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data to fit the model
num_rows = 100
num_columns = 300
num_channels = 1

X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

# Convert the data type to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

def build_cnn_model(input_shape, num_labels):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(num_labels, activation='softmax')) 

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Define input shape and number of labels
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
num_labels = len(np.unique(y_train))

# Build the model
model = build_cnn_model(input_shape, num_labels)

# Summarize the model
model.summary()

from scipy.stats import rayleigh, nakagami

os.makedirs(r"C:\Users\sohai\Downloads\archives\saved_models\Model", exist_ok=True)

# Set up model checkpoint
checkpoint = ModelCheckpoint(r"C:\Users\sohai\Downloads\archives\saved_models\Model\best_model.keras", 
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')

def add_rayleigh_noise(audio, scale=0.1):
    noise = rayleigh.rvs(scale=scale, size=audio.shape)
    return audio + noise

def add_nakagami_noise(audio, mu=1.5, scale=0.1):
    noise = nakagami.rvs(mu, scale=scale, size=audio.shape)
    return audio + noise

def apply_noise_to_dataset(X, noise_func, noise_params):
    X_noisy = np.array([noise_func(audio, **noise_params) for audio in X])
    return X_noisy

X_test_noisy_rayleigh = apply_noise_to_dataset(X_test, add_rayleigh_noise, {'scale': 0.1})

X_test_noisy_nakagami = apply_noise_to_dataset(X_test, add_nakagami_noise, {'mu': 1.5, 'scale': 0.1})

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

model.save(r"C:\Users\sohai\Downloads\archives\saved_models\Model\aug_model.keras")

test_loss_rayleigh, test_acc_rayleigh = model.evaluate(X_test_noisy_rayleigh, y_test, verbose=2)
print(f"Test accuracy under Rayleigh noise: {test_acc_rayleigh}")

test_loss_nakagami, test_acc_nakagami = model.evaluate(X_test_noisy_nakagami, y_test, verbose=2)
print(f"Test accuracy under Nakagami noise: {test_acc_nakagami}")

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()


filetest = r"C:\Users\sohai\Desktop\Testing\Noise2.mp3"
#filetest = r"C:\Users\sohai\Downloads\saved_models\output.wav"
model_path = r"C:\Users\sohai\Downloads\archives\saved_models\Model\aug_model.keras"
# Extract features from the test file

model = load_model(model_path)
prediction_feature = feature_extractor(filetest)
prediction_feature = prediction_feature.reshape(1, 100, 300, 1)

# Get model predictions
predictions = model.predict(prediction_feature)

predicted_class_index = np.argmax(predictions, axis=1)

predicted_class_name = encoder.inverse_transform(predicted_class_index)

print(f"Predicted class: {predicted_class_name[0]}")
