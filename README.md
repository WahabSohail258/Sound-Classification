Project Overview: 

This project involves developing a system to differentiate between various sounds using signal processing techniques. Students will implement algorithms to classify audio signals into different categories (e.g., speech, music, noise) using neural networks. This project provides foundational experience in signal processing, including filtering, pitch detection, and time and frequency domain analysis.

Research Summary: 

To determine the most effective approach for classifying audio into speech, music, or noise, extensive research was conducted into existing methods and algorithms for audio processing and classification. The primary focus was on leveraging advanced machine learning and deep learning techniques to handle the complexity of audio data.

•	Selection of CNN Model:

Audio data exhibits hierarchical features in the time-frequency domain, making Convolutional Neural Networks (CNNs) a strong candidate for this task. CNN’s ability to automatically learn feature hierarchies, such as patterns in spectrograms, aligns well with the nature of audio classification. To validate this approach, we reviewed prior studies and techniques:

1.	Feature Extraction Compatibility: Research emphasized the importance of extracting features such as Mel-Frequency Cepstral Coefficients (MFCCs), which are robust representations of audio signals. CNNs are well-suited for processing these two-dimensional inputs.
2.	Robustness to Noise: Studies highlighted CNNs' ability to learn discriminative features even in noisy datasets, which was critical given our project's testing under various noise conditions.
3.	Proven Success in Related Applications: CNNs have demonstrated exceptional performance in speech recognition, music genre classification, and environmental sound classification, suggesting their potential for our task.

•	Why CNNs Are Helpful:

The choice of CNNs was guided by their numerous advantages:
1.	Automatic Feature Extraction: Unlike traditional machine learning methods, CNNs eliminate the need for manual feature engineering by automatically learning relevant features.
2.	Hierarchical Feature Learning: CNNs can capture both low-level patterns (e.g., pitch and tone) and high-level abstractions (e.g., sound type) through convolutional layers.
3.	Scalability and Efficiency: CNNs efficiently process large datasets and perform well in real-time applications, making them suitable for practical implementations.
4.	Noise Resilience: The model's robustness to noisy data was a pivotal factor, ensuring reliable classification even in challenging environments.

By selecting a CNN-based approach, we achieved a balance between model complexity and performance, ensuring the system's scalability and adaptability to various audio classification scenarios.

Data Collection and Preprocessing: 

The UrbanSound8K dataset alone did not provide sufficient coverage of speech sounds, so we supplemented it with additional datasets, such as MUSAN, to expand the variety of input data. This combination allowed us to train our model on a more diverse set of audio samples and test its performance under different conditions.

Since the UrbanSound8K dataset contains 10 different categories, and the MUSAN dataset contains three categories (Speech, Music, and Noise), we have implemented a class mapping to map each class from the UrbanSound8K and MUSAN dataset to one of the three categories.

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

We extracted features such as Mel-Frequency Cepstral Coefficients (MFCCs), RMS, Zero Crossing Rate, and signal energy. However, MFCCs were primarily used to classify the data into speech, music, and noise. MFCCs are derived from the frequency domain and represent the power spectrum of the audio, closely mimicking the human ear's response to different frequencies. This makes them highly effective for capturing the discriminative features of speech and music. Unlike RMS, Zero Crossing Rate, and energy, which capture basic characteristics, MFCCs provide a richer, more detailed spectral representation, making them ideal for our classification task.

The code extracts Mel-Frequency Cepstral Coefficients (MFCC) features from audio files for classification purposes. The feature_extractor function loads each audio file using librosa, extracts 100 MFCC features, and pads the feature matrix to a consistent length (max_pad_len=300). It then iterates through the metadata DataFrame, which contains file paths and labels, to extract and store the features along with their corresponding class labels. The features are stored in a DataFrame for easy use in machine learning models. The figure below shows the calculation for appropriate padding length which is a common practice in machine learning to ensure that all input data has the same shape.
 

To remove the noise from the audio signal, we designed a bandpass filter. First, we compute the fundamental frequency of the signal by taking its FFT. This fundamental frequency is then used to set the low and high cutoff frequencies for the bandpass filter. Here are the visuals after applying filter to the noisy signal:


The data augmentation functionality enhances audio data diversity by applying several random augmentations. It includes random time shifting, where the audio signal is shifted by a fraction of the sample rate, and random pitch shifting, which alters the pitch by -5 to 5 semitones. Additionally, it performs random speed tuning, adjusting the audio speed by a factor between 0.9 and 1.1, and adds random noise by incorporating a small amount of Gaussian noise. These augmentations help create varied training data for machine learning models.
addition

Pitch Detection Algorithm: 

The project successfully detects pitch using the YIN algorithm, a well-known method for pitch detection. In addition to pitch detection, the code also plots the Mel-frequency spectrogram of the audio file. The YIN algorithm accurately identifies the pitch over time, while the Mel-frequency spectrogram provides a detailed visualization of the spectral content of the audio.
 
The code defines a Convolutional Neural Network (CNN) for audio classification, consisting of four convolutional layers with ReLU activation, followed by max-pooling, dropout for regularization, and batch normalization. The final convolutional layer is followed by global average pooling and a dense output layer with softmax activation for classification. The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss. Input shape and number of labels are derived from the training data, and the model summary is displayed.

Model Training:

The model was successfully trained using CNN’s and epoch size of 30 and gave an accuracy of 90% with the validation accuracy of 90%.

I applied both Rayleigh and Nakagami-m noise models to the test data to evaluate predictions under noisy conditions. The code below demonstrates their application.The results achieved under each model's inference showed an accuracy of 90%.

Final Remarks:

•	This project provided an excellent opportunity to integrate signal processing concepts with machine learning techniques for practical implementation. Through rigorous research, we identified CNNs as the ideal model for audio classification due to their ability to learn hierarchical features effectively. By leveraging datasets like Urban Sound 8k and augmenting them for robustness, we ensured the model's applicability under diverse conditions.

•	The project achieved high prediction accuracy across speech, music, and noise categories, demonstrating the effectiveness of our approach. Testing under noisy conditions further validated the model's resilience and adaptability.

•	Key challenges included ensuring sufficient dataset diversity and optimizing feature extraction for real-time performance. These challenges were addressed through strategic data augmentation and careful model architecture design.

•	This project enhanced our understanding of audio signal processing, the application of machine learning algorithms, and the importance of robust testing in noisy environments. These learnings are instrumental for future projects involving signal processing and AI applications.

