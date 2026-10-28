# Activity recognition from wearable sensors (PAMAP2)

Classifying 12 everyday and sport activities (for example walking, running, cycling, stair climbing and rope jumping) from body-worn inertial sensors and heart rate, using the public **PAMAP2 Physical Activity Monitoring** dataset (Reiss & Stricker, 2012).

## Data
- 9 participants, each wearing three inertial measurement units (hand, chest and ankle; accelerometer and gyroscope at 100 Hz) and a heart-rate monitor (about 9 Hz).
- 12 protocol activities. Transient periods (activity ID 0) are removed.
- Source: UCI Machine Learning Repository (CC BY 4.0). The data are not stored in this repository; the notebook downloads them.

## Method (`pamap2_pipeline.py`, `pamap2_subject_wise.ipynb`)
1. **Windowing:** each continuous activity bout is cut into 5.12 s windows (512 samples) with 50% overlap.
2. **Features:** for each accelerometer and gyroscope axis, and for the acceleration magnitude at each sensor, the mean, SD, minimum, maximum and interquartile range, plus frequency-domain features (dominant frequency, spectral entropy and spectral energy from Welch's method). Heart rate mean and SD are also included.
3. **Model:** random forest (300 trees, balanced class weights).
4. **Evaluation:**
   - **Leave-one-subject-out (main result):** train on eight participants and test on the ninth, repeated for every participant. This shows how well the model works for a person it has never seen.
   - **Random split of windows (for comparison only):** this is optimistic, because overlapping windows from the same person appear in both training and test data.

## How to run
Open `pamap2_subject_wise.ipynb` in Google Colab and choose **Runtime → Run all** (about 10–20 minutes). Results are written to `results/`:
- `loso_results.csv`: accuracy and macro-F1 for each held-out participant
- `random_split_results.csv`: the optimistic comparison
- `loso_macro_f1.png`: per-participant performance

## A note on the original notebook
`phydat.ipynb` is my original 2024 coursework analysis. It split individual samples at random into training and test sets, which places almost identical neighbouring samples from the same person on both sides of the split. That is why it reports near-perfect accuracy, which would not hold for a new person. I have kept it for transparency; the subject-wise pipeline above replaces it.

## Limitations
- Nine participants is a small sample, and participant 109 performed few protocol activities.
- Laboratory protocol activities may not reflect free-living or sport-specific movement.
- Hand-crafted features; deep learning on raw signals is a natural next step.

## Reference
Reiss, A. & Stricker, D. (2012). Introducing a new benchmarked dataset for activity monitoring. *Proceedings of the 16th International Symposium on Wearable Computers (ISWC)*, 108–109.

## Author
Yashwanth Ajjilapuram · [ORCID 0009-0000-0198-8303](https://orcid.org/0009-0000-0198-8303)
