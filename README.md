# Deepfake Audio Detection Using WavLM and OCSoftmax

## 1. Project Overview

Deepfake Audio Detection is a machine-learning application designed to classify audio recordings as **Bonafide (Real)** or **Spoof (Deepfake)**. The application uses a WavLM-based neural network for speech representation learning and an OCSoftmax-based scoring mechanism to distinguish genuine speech from potentially synthetic or manipulated audio.

The system is implemented in Python using Streamlit and provides an interactive interface for audio testing, model performance analysis, visualizations, and PDF report generation.

## 2. Objectives

- Detect potentially AI-generated or spoofed audio recordings.
- Preprocess audio into a consistent format for model inference.
- Extract speech representations using the WavLM architecture.
- Classify audio using a trained model checkpoint and decision threshold.
- Visualize audio waveforms, Mel-spectrograms, and model evaluation metrics.
- Generate downloadable PDF reports for individual audio tests and model evaluation.
- Provide a local AI assistant for questions about model performance.

## 3. Technologies Used

- **Programming language:** Python
- **User interface:** Streamlit
- **Deep learning:** PyTorch
- **Speech representation model:** WavLM Base Plus
- **Classification scoring:** OCSoftmax
- **Audio processing:** Torchaudio, SoundFile, SciPy
- **Data analysis:** Pandas, NumPy
- **Visualization:** Plotly
- **PDF generation:** ReportLab
- **Local AI assistant:** Ollama with the Llama 3.2 model

## 4. System Modules

### Module 1: Audio Input

The application accepts audio through live recording or uploaded audio files. Supported upload formats are WAV, FLAC, MP3, and OGG.

### Module 2: Audio Preprocessing

The input audio is converted to mono when necessary, resampled to 16 kHz, and cropped or padded to 64,000 samples, corresponding to four seconds. The signal is then normalized based on its root mean square (RMS) energy.

### Module 3: WavLM Feature Extraction

The WavLM encoder extracts hidden representations from multiple network layers. A learnable weighted layer sum combines these representations, and attentive statistical pooling calculates mean and standard-deviation features. A neural embedding layer produces the final audio representation.

### Module 4: Deepfake Classification

The extracted embedding is passed to the OCSoftmax scoring component. The resulting score is compared with the decision threshold loaded from the trained checkpoint.

- Score greater than or equal to the threshold: Bonafide (Real).
- Score below the threshold: Spoof (Deepfake).

### Module 5: Audio Visualization

The application displays the preprocessed waveform and Mel-spectrogram to support visual inspection of the audio signal.

### Module 6: Model Analytics

This module loads available training metrics and evaluation logs. It supports training-loss and equal error rate (EER) histories, ROC and DET curves, confusion matrices, and learned WavLM layer-weight visualizations when the corresponding data is available.

### Module 7: PDF Report Generation

The application generates downloadable PDF reports containing individual audio classification results or overall model evaluation metrics.

### Module 8: AI Assistant

The application includes a chat interface that sends user questions and available model-performance context to a local Ollama server running the Llama 3.2 model.

## 5. Project Structure

```text
Deepfake/
├── dashboard.py
├── model_artifacts/
│   ├── best_wavlm_asvspoof (1).pt
│   ├── training_metrics (1).pt
│   └── metrics.csv
└── temp_audio/
    └── input.wav
```

The `temp_audio` directory is created by the application when processing an audio sample. The metrics directory can also contain an optional Excel file or extensionless metrics file.

## 6. Installation and Execution

### Prerequisites

- Python 3.10 or a compatible Python environment
- pip package manager
- The trained model checkpoint
- Internet access for downloading model configuration files if they are not cached locally

### Step 1: Install Dependencies

Open a terminal in the project directory and install the dependencies listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

### Step 2: Verify Model Files

Ensure that the trained checkpoint is located at:

```text
model_artifacts/best_wavlm_asvspoof (1).pt
```

The application also looks for training metrics and evaluation logs in the `model_artifacts` directory.

### Step 3: Launch the Application

Run the following command:

```bash
streamlit run dashboard.py
```

### Step 4: Access the Application

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

### Step 5: Test Audio

Select the Real-Time Voice Tester page, record a voice sample or upload an audio file, and inspect the classification result, score, waveform, and Mel-spectrogram.

The Model Analytics & Reports page displays available evaluation information. The AI Assistant requires a locally running Ollama server with the `llama3.2` model available.

## 7. Main Processing Logic

The main processing pipeline is:

1. Receive an audio recording or uploaded file.
2. Save the audio temporarily.
3. Read the waveform and sampling rate.
4. Convert the waveform to mono and resample it to 16 kHz if necessary.
5. Crop or pad the waveform to 64,000 samples.
6. Normalize the audio signal.
7. Pass the waveform through the trained WavLM-based model.
8. Calculate the OCSoftmax score.
9. Compare the score against the checkpoint's decision threshold.
10. Display the classification verdict and supporting visualizations.
11. Generate a downloadable PDF report.

The classification rule implemented in the application is:

```python
is_bonafide = llr_score >= threshold
verdict_str = (
    "BONAFIDE (REAL)"
    if is_bonafide
    else "SPOOF (DEEPFAKE)"
)
```

## 8. Sample Input and Output

### Sample Input

- Input type: Uploaded WAV audio file
- Sampling rate: 16,000 Hz
- Channels: Mono
- Processed duration: 4 seconds
- Number of processed samples: 64,000

### Illustrative Output

The following values are examples of the output format and are not measured results from the trained model.

```text
Audio Analysis Result
---------------------
Classification: BONAFIDE (REAL)
Model Score: [score produced during inference]
Decision Threshold: [threshold loaded from checkpoint]

Additional Results:
- Processed waveform
- Mel-spectrogram
- Acoustic diagnostic summary
- Downloadable PDF report
```

If the score falls below the decision threshold, the application instead displays `SPOOF (DEEPFAKE)`.

Actual classification scores and predictions depend on the supplied audio and the trained checkpoint.

## 9. Expected Results

The application is designed to provide:

- A real-or-spoof classification for an audio sample when the trained checkpoint loads successfully.
- Visual representations of the processed audio.
- Model evaluation plots when the required metrics are available.
- Downloadable reports for audio analysis and model evaluation.
- Responses to model-performance questions when the local AI assistant is available.

The model's actual accuracy, EER, and ROC-AUC should be reported using measured evaluation results from the available training or evaluation artifacts rather than assumed values.

## 10. Limitations

- Predictions depend on the quality and characteristics of the input audio.
- The trained checkpoint must be available and compatible with the model architecture.
- The application processes a fixed four-second audio segment, so longer recordings are truncated.
- Model analytics depend on the availability and structure of the saved metrics.
- The AI assistant requires a running local Ollama service and the specified language model.
- A classification result is a model prediction and should not be treated as definitive proof that audio is genuine or manipulated.

## 11. Conclusion

The project integrates deep-learning-based speech representation, audio preprocessing, classification, interactive visualization, evaluation analytics, and report generation into a single application. It provides a practical interface for examining audio recordings and evaluating a trained deepfake audio detection model.

