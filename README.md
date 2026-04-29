

---

# ECG Arrhythmia Detection System

**Live App:** [https://ranjith445-ecg-arrhythmia-detection.hf.space/](https://ranjith445-ecg-arrhythmia-detection.hf.space/)

---

## What is this project about?

The human heart beats in a regular rhythm. This rhythm can be seen using a test called an ECG (Electrocardiogram).

Sometimes, the heart does not beat properly. This condition is called **arrhythmia**.

* The heart may beat too fast
* Too slow
* Or irregularly

Detecting this early is very important. But:

* Doctors need time to analyze ECG reports
* A single ECG has thousands of heartbeats
* In some places, patients wait a long time for results

This project builds an **AI system that can analyze ECG signals automatically and quickly**.

---

## What does this system do?

This application allows a user to upload an ECG file and get:

* A decision: whether arrhythmia is present or not
* A waveform showing heartbeats
* Highlighted abnormal beats
* Charts showing different types of beats
* A summary of heart condition
* A simple explanation of the result
* Suggested next steps

---

## What is Arrhythmia? (Simple explanation)

The heart works using electrical signals.

Each heartbeat follows a pattern called **PQRST**.

* P wave → signal starts
* QRS complex → main heartbeat
* T wave → recovery

If this pattern changes, it means something is wrong.

### Types of beats detected

1. **Normal beat**

   * Regular timing
   * Proper signal flow

2. **Atrial beat**

   * Happens earlier than expected
   * Signal comes from a different place

3. **Ventricular beat**

   * More serious
   * Signal starts from the wrong chamber
   * Shape looks abnormal

---

## How does the AI detect arrhythmia?

The system uses two AI models working together.

### 1. Random Forest (works with numbers)

It measures important values like:

* Time between beats
* Heart rate
* Shape duration

Then it uses many decision trees to decide the beat type.

---

### 2. CNN (works with signal shape)

* Looks directly at the ECG waveform
* Learns patterns of normal and abnormal beats
* Detects shape differences automatically

---

### Final decision

Both models analyze each heartbeat.

* Their results are combined
* If abnormal beats cross a limit, the system reports arrhythmia

---

## How the system processes ECG (step-by-step)

1. **Cleaning the signal**
   Removes noise and unwanted signals

2. **Finding heartbeats**
   Detects important points (R peaks)

3. **Extracting features**
   Measures timing and shape details

4. **Classifying beats**
   Uses AI models to label each beat

5. **Showing results**
   Displays graphs, summary, and explanation

---

## Dataset used

* MIT-BIH Arrhythmia Dataset
* Contains real ECG recordings from patients
* Used around **14,902 heartbeats** for training

---

## Results

* CNN Model Accuracy: 97.38%
* Random Forest Accuracy: 85%

The CNN performs better because it learns directly from waveform shapes.

---

## Features of the application

* Upload ECG file
* Detect abnormal heartbeats
* Visual waveform display
* Color-coded beat types
* Charts for analysis
* Simple explanation of results

---

## Understanding the output (simple)

* Green → Normal
* Orange → Atrial issue
* Red → Ventricular issue (more serious)

The system also shows:

* Heart rate
* Risk level
* Beat distribution

---

## Important Note

* This system is built for **learning and demonstration**
* It is **not a medical tool**

---

## Disclaimer

This project is for educational purposes only.
Do not use it for medical decisions. Always consult a doctor.

---

