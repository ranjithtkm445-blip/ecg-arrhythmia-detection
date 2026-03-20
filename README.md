---
title: ECG Arrhythmia Detection
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ECG Arrhythmia Detection System

**Live App:** https://ranjith445-ecg-arrhythmia-detection.hf.space

---

## Problem Statement

Every year thousands of people die from undetected heart arrhythmias.
The main reason is simple — there are not enough cardiologists to
manually read every ECG recording. A single 30-minute ECG has over
2000 heartbeats. A cardiologist needs 15 to 30 minutes to review one.
In rural areas patients wait days for a result.

This project was built to solve that. The idea is straightforward:
train a machine learning model on expert-labeled ECG data, then let
it automatically scan an ECG and flag any abnormal beats — in seconds.

---

## What I Built

A web application where a user uploads their ECG files and gets back:
- A verdict — whether arrhythmia is present or not
- A color coded ECG waveform showing exactly where abnormal beats occurred
- Charts showing beat distribution from two different AI models
- A clinical summary with heart rate, ST deviation and risk level
- A plain English explanation of what was detected and why
- Recommended next steps based on the risk level

---

## What is Arrhythmia

The heart has its own electrical system. Every beat starts with a
signal from the SA node, travels through the atria, pauses at the
AV node, then fires the ventricles. This produces the classic PQRST
wave pattern on an ECG.

Arrhythmia is when this system breaks down. The heart may beat too
fast, too slow, or fire from the wrong location. There are three
beat types this system detects:

Normal beat — the electrical signal follows the correct path.
The QRS spike is narrow and a clear P wave appears before it.

Atrial beat — a premature beat that fires early from the atria.
The QRS shape looks similar to normal but the P wave is absent
or has an unusual shape. The timing is off — the beat arrives
earlier than expected.

Ventricular beat — the most dangerous type. The beat fires from
the ventricles directly, bypassing the normal conduction pathway.
The QRS complex becomes wide and bizarre shaped. There is no P wave.
The T wave flips in the opposite direction. If these occur frequently
it can lead to ventricular tachycardia or even cardiac arrest.

---

## How to Read an ECG for Arrhythmia

An ECG records the electrical activity of the heart as a wave.
Each heartbeat looks like this:
```
         R
        /\
       /  \
  P   /    \     T
 /\  /      \   /\
/  \/        \ /  \
     Q        S
|← QRS →|
|←    RR Interval    →|
```

P wave — the atria contracting
QRS complex — the ventricles contracting, the tall sharp spike
T wave — the ventricles recovering
RR interval — time between two beats, used to calculate heart rate

To spot arrhythmia a doctor looks for:

Is the heart rate normal — between 60 and 100 BPM?
Are all RR intervals equal — or is the spacing irregular?
Is there a P wave before every QRS — or is it missing?
Is the QRS narrow or wide — wide means ventricular origin?
Is the ST segment flat or elevated — elevated suggests ischemia?

A normal beat has a narrow QRS, a visible P wave before it, and
regular spacing from the previous beat. A ventricular beat has a
wide messy QRS, no P wave, and is followed by a long pause.

---

## How the AI Detects It

Two models were trained and their results are averaged together.

**Random Forest** works on numbers. For each beat it calculates
six clinical measurements — RR interval, heart rate, QRS duration,
PR interval, ST deviation, and RR variability. These six numbers
are fed into 200 decision trees. Each tree votes on the beat type.
The majority vote wins.

The most important features turned out to be RR interval at 31.6%,
ST deviation at 29.8% and heart rate at 24.4%. These three features
alone capture most of the discriminating information between beat types.

**CNN** works on the raw waveform. A 650ms window of raw ECG signal
is cut around each R peak — 90 samples before and 144 samples after.
These 234 samples are fed directly into a convolutional neural network.
The network learns to recognize the shapes of normal, atrial and
ventricular beats from the waveform itself without any manual feature
engineering.

The CNN architecture has three convolutional blocks that learn
progressively complex patterns — from simple edge detection in the
first layer to full beat morphology recognition in the third.
It achieves 97.38% accuracy on the test set.

Both models give their prediction for every beat. The results are
averaged. If either abnormal beat type exceeds 5% of total beats
the system raises an arrhythmia verdict.

---

## Pipeline

**Step 1 — Preprocessing**
The raw ECG signal is loaded from the .dat file and passed through
a Butterworth bandpass filter set between 0.5 Hz and 40 Hz.
Everything below 0.5 Hz is baseline wander from breathing and
body movement. Everything above 40 Hz is powerline noise and
muscle artifact. The filter removes both and keeps only the
clinically relevant ECG signal.

**Step 2 — Segmentation**
The Pan-Tompkins algorithm is used to detect R peaks.
It works by differentiating the signal, squaring it, then
applying a moving window integrator to find the envelope.
Peaks in this envelope correspond to R peaks in the original signal.
Once R peaks are found, fixed windows are placed around each one
to identify the P wave, QRS complex and T wave boundaries.

**Step 3 — Feature Extraction**
Six features are extracted from each detected beat.
RR interval is calculated from the distance to the previous R peak.
Heart rate is derived from RR interval.
QRS duration comes from a fixed window around the R peak.
PR interval is the distance from P wave start to QRS start.
ST deviation is the mean signal amplitude 80 to 120ms after the R peak.
RR variability is the difference between consecutive RR intervals.
Eight records from the MIT-BIH database were used, giving 14902 beats.

**Step 4 — Classification**
The dataset is highly imbalanced — 13057 normal beats, 1026 atrial,
819 ventricular. Resampling brings each class to 700 beats.
Three models were trained: SVM at 83.33%, Random Forest at 85.00%
and CNN at 97.38%. The Random Forest and CNN are saved and used
in the web application.

**Step 5 — Dashboard**
The web app loads the trained models, processes the uploaded ECG,
classifies every beat and builds a full clinical report including
the verdict, waveform plot, charts, clinical summary and explanation.

---

## Dataset

MIT-BIH Arrhythmia Database from PhysioNet.
48 half-hour two-lead ECG recordings sampled at 360 Hz
with expert beat annotations.

Eight records were used for training:

| Record | Beats | Normal | Atrial | Ventricular | Notes |
|---|---|---|---|---|---|
| 100 | 2270 | 2236 | 33 | 1 | Clean normal baseline |
| 101 | 1858 | 1855 | 3 | 0 | Normal rhythm |
| 105 | 1285 | 1251 | 0 | 34 | Arrhythmia variety |
| 200 | 1595 | 852 | 9 | 734 | Heavy PVC |
| 208 | 1581 | 1268 | 311 | 2 | Atrial rich |
| 209 | 2124 | 1864 | 260 | 0 | Atrial rich |
| 213 | 2439 | 2162 | 229 | 48 | Atrial rich |
| 222 | 1750 | 1569 | 181 | 0 | Atrial rich |
| Total | 14902 | 13057 | 1026 | 819 | |

---

## Results

| Model | Test Accuracy | CV Accuracy |
|---|---|---|
| SVM | 83.33% | 68.10% |
| Random Forest | 85.00% | 87.14% |
| CNN | 97.38% | 97.38% |

The CNN significantly outperforms the Random Forest because it
learns directly from the waveform shape rather than hand-crafted
features. It is especially strong at detecting ventricular beats
where the wide QRS morphology is a clear visual pattern.

The Random Forest is better at detecting atrial beats because
the timing features like RR interval and RR variability capture
the premature nature of atrial contractions more reliably than
the waveform alone.

---

## Tech Stack

Python, Flask, TensorFlow, Scikit-learn, SciPy, NumPy, WFDB,
Matplotlib, Chart.js, HTML, CSS, JavaScript, Docker,
Hugging Face Spaces, Git, GitHub

---

## Project Structure
```
ecg-arrhythmia-detection/
├── app.py                    Flask backend
├── Dockerfile                Container config
├── requirements.txt          Dependencies
├── templates/index.html      Frontend page
├── static/style.css          Dark theme CSS
├── static/script.js          Charts and API
├── best_model.pkl            Random Forest
├── cnn_model.keras           CNN model
├── scaler.pkl                Feature scaler
├── cnn_encoder.pkl           Label encoder
├── step1_preprocessing.py    Signal filtering
├── step2_segmentation.py     Beat segmentation
├── step3_features.py         Feature extraction
├── step4_classification.py   RF and SVM training
├── step4b_cnn.py             CNN training
└── step5_visualization.py    Dashboard output
```

---

## How to Run Locally
```bash
git clone https://github.com/ranjithtkm445-blip/ecg-arrhythmia-detection
cd ecg-arrhythmia-detection
python -m venv ecg_env
ecg_env\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 in your browser.

---

## Understanding the Output

**Verdict Banner**
The first thing shown after analysis. A green banner means no
arrhythmia was detected. Orange means atrial arrhythmia was found.
Red means ventricular or multiple arrhythmias were detected.
The banner also shows the percentage of abnormal beats that
triggered the verdict.

**ECG Waveform**
Shows the first 10 seconds of the filtered ECG signal. Each
detected beat is marked with a colored dot — green for normal,
orange for atrial, red for ventricular. The shaded region around
each dot shows the QRS boundary. Abnormal beats are immediately
visible because they appear in different colors at irregular positions.

**Beat Distribution Charts**
Two pie charts side by side — one from the Random Forest and one
from the CNN. Comparing them shows how much the two models agree.
A third chart shows the overall agreement percentage between both
models. High agreement means the result is reliable.

**Heart Rate Over Time**
A scatter plot showing heart rate at every beat across the full
recording. Yellow dashed line marks 60 BPM and red marks 100 BPM.
Clusters of red dots above 100 BPM indicate tachycardia episodes.
Sudden drops below 60 BPM indicate bradycardia episodes.

**Clinical Summary**
Shows average minimum and maximum heart rate, average RR interval,
ST deviation value and risk level from both models. Heart rate
status is labeled as NORMAL, BRADYCARDIA or TACHYCARDIA. ST
deviation is labeled NORMAL or ABNORMAL based on whether it
falls within the -0.1 to +0.1 mV clinical threshold.

**Model Accuracy Chart**
Bar chart comparing SVM, Random Forest and CNN side by side.
Per class accuracy bars below show how well each model performed
on normal, atrial and ventricular beats separately.

**Clinical Explanation**
This section explains the verdict in plain language. It lists
exactly which features triggered the detection, gives a beat by
beat breakdown with counts from both models, interprets the
findings clinically and recommends next steps. For high risk
results it recommends immediate cardiologist consultation and
a 12-lead ECG. For moderate risk it recommends scheduling an
appointment and considering a Holter monitor. For low risk it
recommends routine annual checkup.

---

## Disclaimer

This project was built for research and educational purposes.
The predictions are not medically certified and should not be
used for clinical diagnosis. Always consult a qualified
cardiologist for any cardiac concerns.

---

*Built as a final year project exploring the intersection of
biomedical signal processing and machine learning.*
