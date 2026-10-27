# Acoustic Event Classification: DTW vs GMM

![Python](https://img.shields.io/badge/Python-3.x-blue)

Reducing event misclassification in acoustic signals by comparing a
**Dynamic Time Warping (DTW)** classifier against a **Gaussian Mixture Model (GMM)**
on noise-filtered **MFCC** features.

## Problem
Acoustic event classification needs accurate feature extraction and a classifier
that copes with temporal variation in sequential data. This project compares two
classic approaches to see which gives fewer misclassified events and better recall.

## Result (TL;DR)

| Model | Mean accuracy (10 iterations) |
|-------|-------------------------------|
| **GMM** | **99.3%** |
| DTW | 96.6% |

- GMM improved accuracy by **+2.7 points** over DTW.
- The difference was statistically significant (independent-samples t-test, p < 0.05; SPSS and ClinCalc, alpha = 0.05, 80% power).
- Recall: [add your recall numbers for GMM and DTW here]

![DTW vs GMM accuracy](results/accuracy_dtw_vs_gmm.png)

## Approach
1. **Acoustic data:** [source, number of recordings, number of event classes]
2. **Preprocessing and noise removal:** [describe the filter / method you used]
3. **Feature extraction:** MFCC (also compared wavelet and LPC features in a separate experiment)
4. **Classifiers:**
   - DTW: template matching that aligns sequences of different lengths
   - GMM: probabilistic model of the MFCC distribution per event class
5. **Evaluation:** accuracy and recall over 10 iterations, error analysis, independent-samples t-test

## Dataset
[Name, source link, number of samples, sampling rate, train/test split.
If the data is recorded by you or simulated, say so clearly.]

## Project structure
```
acoustic-event-classification/
├── README.md
├── requirements.txt
├── data/          # audio files or a download script
├── notebooks/     # exploration and modelling notebook
├── src/           # preprocessing, features, models
└── results/       # plots and summary tables
```

## How to run
```bash
git clone https://github.com/thrka33/acoustic-event-classification
cd acoustic-event-classification
pip install -r requirements.txt
jupyter notebook
```

## Limitations and next steps
- Results come from 10 iterations on [dataset size]; a larger dataset would give stronger evidence.
- Try a CNN on MFCC / spectrogram inputs and compare against GMM.
- Test on noisier real-world recordings.

## Author
**Tharika S**, B.E. Electronics and Communication Engineering, Saveetha School of Engineering
[GitHub](https://github.com/thrka33)
