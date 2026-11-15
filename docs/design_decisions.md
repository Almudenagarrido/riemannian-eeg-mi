# Design decisions

Justification of every choice in the data pipeline (`src/riemannian_eeg/data.py`).
All methods (CSP+LDA, pyRiemann, SPDNet) use this same pipeline.

## 1. Dataset

**BCI Competition IV 2a** (MOABB: `BNCI2014_001`), a standard public benchmark for 4-class motor imagery.

- 9 subjects, 2 sessions per subject recorded on different days
- 288 trials per session, balanced classes: left hand, right hand, feet, tongue
-  25 channels, sampled at 250 Hz:
  - 22 **EEG** (electroencephalography): scalp electrodes that record the brain's electrical activity
  - 3 **EOG** (electrooculography): electrodes around the eyes that record eye movements and blinks
- License: CC-BY-ND 4.0. The data are not redistributed with our code; MOABB downloads them automatically into `data/` (ignored by Git).

Refs: Tangermann et al. (2012); Brunner et al. (2008).

## 2. Frequency band: 8-30 Hz

At rest, the motor cortex shows synchronized oscillations in two rhythms:

- **mu**: 8-13 Hz
- **beta**: 13-30 Hz

Imagining a movement activates the corresponding motor area, which makes these oscillations lose amplitude: **Event-Related Desynchronization (ERD)**. Hand imagery desynchronizes the mu rhythm over the contralateral hand area, while foot or tongue imagery can instead enhance it there (**Event-Related Synchronization, ERS**), which helps distinguish the 4 classes. Both effects are power changes within the mu and beta rhythms, so the 8-30 Hz band captures them, and it removes slow drifts (< 8 Hz) and muscle noise (> 30 Hz). It is the standard choice for CSP.

To isolate the effect of the classifier (Euclidean vs Riemannian) and not of the preprocessing, the same fixed band is used for all subjects and all methods.

**Possible extension:** subject-specific bands, e.g. a filter bank as in FBCSP (Ang et al., 2008). The 0-2 s fixation period of each trial could serve as the reference to find each subject's most reactive band, comparing its power spectrum with the imagery period. The bands must be selected on training data only, to avoid data leakage.

Refs: Pfurtscheller & Lopes da Silva (1999); Pfurtscheller et al. (2006); Guger, Ramoser & Pfurtscheller (2000); Ang et al. (2008).

## 3. Time window: t = 2-6 s

Each trial: a fixation cross and a warning tone appear at t = 0 s, the cue (arrow) appears at t = 2 s, and the subject performs motor imagery until the cross disappears at t = 6 s. We keep the 2-6 s imagery period (MOABB default for this dataset): 4 s at 250 Hz = 1001 samples, both ends included.

Ref: Brunner et al. (2008).

## 4. Channels: 22 EEG only

The 3 EOG channels record eye movements, not motor cortex activity. The dataset rules state they are provided for artifact processing and must not be used for classification. Using them could also let a classifier rely on eye movements instead of brain activity. The resulting covariance matrices are 22 x 22.

Ref: Brunner et al. (2008).

## 5. Evaluation protocol: cross-session

Train on session 1 (`0train`) and test on session 2 (`1test`), with one model per subject. This is the official split of the competition.

- The two sessions were recorded on different days: cap placement, electrode impedance and the subject's state change between them. This is the realistic BCI use case: calibrate one day, use it another.
- Randomly mixing trials from both days would make train and test too similar and give optimistic results.
- Riemannian methods are reported to be robust to this session-to-session variability, which makes this protocol especially relevant to compare them against the Euclidean baseline.
- It makes our results comparable with the competition and the literature.
- Validation data (only needed for SPDNet hyperparameters) must be taken from the training session. The test session is only used for the final evaluation.

Refs: Tangermann et al. (2012); Congedo, Barachant & Bhatia (2017).

## 6. Metric: accuracy (and Cohen's kappa)

Classes are balanced, so accuracy is a reliable metric. Chance level is 25 % with 4 classes, although with a finite number of trials the real chance threshold is somewhat higher (Müller-Putz et al., 2008). The competition used Cohen's kappa; with 4 balanced classes, kappa = (accuracy - 0.25) / 0.75, so both can be reported.

## References

- Ang, K. K., Chin, Z. Y., Zhang, H., Guan, C. (2008). Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface. *IJCNN 2008*.
- Brunner, C., Leeb, R., Müller-Putz, G. R., Schlögl, A., Pfurtscheller, G. (2008). BCI Competition 2008 – Graz data set A. https://www.bbci.de/competition/iv/
- Congedo, M., Barachant, A., Bhatia, R. (2017). Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. *Brain-Computer Interfaces*, 4(3).
- Guger, C., Ramoser, H., Pfurtscheller, G. (2000). Real-time EEG analysis with subject-specific spatial patterns for a brain-computer interface (BCI). *IEEE Trans. Rehabil. Eng.*, 8(4).
- Müller-Putz, G. R., Scherer, R., Brunner, C., Leeb, R., Pfurtscheller, G. (2008). Better than random? A closer look on BCI results. *International Journal of Bioelectromagnetism*, 10(1).
- Pfurtscheller, G., Brunner, C., Schlögl, A., Lopes da Silva, F. H. (2006). Mu rhythm (de)synchronization and EEG single-trial classification of different motor imagery tasks. *NeuroImage*, 31(1), 153–159.
- Pfurtscheller, G., Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11). doi:10.1016/S1388-2457(99)00141-8
- Tangermann, M. et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55. doi:10.3389/fnins.2012.00055