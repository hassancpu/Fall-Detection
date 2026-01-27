# 🚀 Fall-Detection: SVM-based Fall Event Detection  

**Detecting fall and not-fall events in videos using SVM**  

This repository provides a **simple pipeline** for detecting fall events in video sequences using a **Support Vector Machine (SVM)** classifier.

---

## 📌 Table of Contents
1. [Overview](#overview)  
2. [Data & Feature Extraction](#data--feature-extraction)  
3. [Training](#training)  
4. [Testing / Detection](#testing--detection)  
5. [Results](#results)  
6. [Acknowledgements](#acknowledgements)  

---

<a name="overview"></a>
## 🧠 Overview
This project detects **fall vs. not-fall events** in videos:  

- Features are extracted from **video frames**.  
- A **Support Vector Machine (SVM)** is trained on these features.  
- The trained SVM is used to classify new video sequences.  

---

<a name="data--feature-extraction"></a>
## 📂 Data & Feature Extraction

1. Define the path to save extracted features of **fall** and **not-fall** events from training videos.  
2. Define the path to the **training videos**.  
3. Define paths to **all extracted features** (new and old).  
4. To extract features for fall and not-fall events during runtime, press the **`z`** key.  

> ⚡ Feature extraction is required before training the SVM.

---

<a name="training"></a>
## 🛠 Training

1. Define the path where the **trained SVM** should be saved.  
2. Run the training script with your extracted features to train the classifier.  

> The SVM model will learn to distinguish between **fall** and **not-fall** events based on the features.

---

<a name="testing--detection"></a>
## 🖥 Testing / Detection

1. Define the path to the **trained SVM**.  
2. Define the path to a **test video**.  
3. Run the detection script to classify fall and not-fall events in the video.  

> The SVM predicts frame-wise or event-wise labels using the learned model.

---

<a name="results"></a>
## 📊 Results

Below is an example output of the fall-detection system:

[![Fall Detection Example](https://user-images.githubusercontent.com/62461020/122333347-3eebbc00-cf4d-11eb-95a9-a6d4cd8a05d8.mp4)](https://user-images.githubusercontent.com/62461020/122333347-3eebbc00-cf4d-11eb-95a9-a6d4cd8a05d8.mp4)  

- The dataset for training and testing was collected by **Mr. Farahnezhad**, who also contributed to code development.  
- This video shows a demonstration of **fall vs. not-fall detection** in action.

---

<a name="acknowledgements"></a>
## 🙏 Acknowledgements

We thank **Mr. Farahnezhad** for providing the dataset and supporting the development of this project.  
