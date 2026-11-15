# From EEG trials to SPD matrices

All the methods in this project (CSP + LDA, MDM, tangent space and SPDNet) start from the same
representation: each EEG trial is summarized by its spatial covariance matrix. This document
explains, step by step, why that matrix contains the information we need to classify motor
imagery, and why it is a symmetric positive definite (SPD) matrix, the object our Riemannian
methods work with.

## 1. A trial as a matrix

Each trial is recorded by 22 EEG electrodes, sampled $T = 1001$ times (4 s at 250 Hz). We write it
as a matrix

$$
X = \begin{pmatrix} x_1(t) \\ \vdots \\ x_{22}(t) \end{pmatrix} \in \mathbb{R}^{22 \times T},
$$

where row $c$ is the signal of electrode $c$, and column $t$, written $x(t) \in \mathbb{R}^{22}$, is
the value of all electrodes at time $t$.

## 2. Zero mean after band-pass filtering

Any signal can be seen as a sum of oscillations at different frequencies, plus a constant offset
that doesn't oscillate at all (0 Hz): that offset is the signal's mean. The 8-30 Hz band-pass
filter only keeps oscillations between 8 and 30 Hz, so it removes the offset and leaves each
electrode oscillating around zero (zero mean). This is what makes the covariance formula below
so simple: there is no mean to subtract.

## 3. Trial covariance

Each trial has its own $22 \times 22$ covariance matrix, computed from its $T$ time samples. Since
the signal has zero mean,

$$
C = \frac{1}{T} X X^\top, \qquad C_{cd} = \frac{1}{T} \sum_t x_c(t)\, x_d(t).
$$

- The **diagonal** entries $C_{cc}$ are the average squared value of each electrode, i.e. its
  power in the mu/beta band.
- The **off-diagonal** entries $C_{cd}$ measure how much electrodes $c$ and $d$ vary together:
  large and positive if they go up and down at the same time, negative if one goes up when the
  other goes down, close to zero if they are unrelated.

## 4. Spatial filters as virtual electrodes

Each electrode sits on the scalp and records a mix of the activity of many brain areas, more
strongly the closer they are. No single electrode sees one area cleanly. But, as with several
microphones recording a room full of people, weighting and combining the electrodes can reinforce
the activity of one area and cancel out the rest.

A spatial filter is a vector of weights $w \in \mathbb{R}^{22}$, one per electrode. Applying it
gives one new signal, a "virtual electrode":

$$
s(t) = w^\top x(t) = \sum_{c=1}^{22} w_c \, x_c(t).
$$

For example, with three electrodes and $w = (1, -0.5, 0)$ on (C3, Cz, C4), the virtual electrode
is "C3 minus half of Cz, ignoring C4": it removes from C3 the part it shares with the central
area and keeps what is specific to the left motor cortex. Repeating this at every time $t$ gives
a full signal that oscillates like a real electrode.

## 5. The power of any virtual electrode comes from $C$

The power of the virtual electrode is the average of its squared values (its variance, since it
also has zero mean). It depends on $w$, because the weights decide how the electrodes are
combined, and it can be computed directly from $C$.

With two electrodes, squaring $s(t) = w_1 x_1(t) + w_2 x_2(t)$ gives

$$
s(t)^2 = w_1^2\, x_1(t)^2 + 2\, w_1 w_2\, x_1(t)\, x_2(t) + w_2^2\, x_2(t)^2,
$$

and averaging over time (the weights are constants, so they come out of the average), each
average is an entry of $C$:

$$
\operatorname{Var}(s) = w_1^2\, C_{11} + 2\, w_1 w_2\, C_{12} + w_2^2\, C_{22} = w^\top C \, w.
$$

With 22 electrodes the same expansion gives one term per pair of electrodes:

$$
\operatorname{Var}(s) = \sum_{c=1}^{22} \sum_{d=1}^{22} w_c\, w_d\, C_{cd} = w^\top C \, w.
$$

So once we have $C$, we can get the power of *any* combination of electrodes without going back
to the raw signal: $C$ describes the trial, and $w$ says how the electrodes are combined.

## 6. Why this is the information we need

Motor imagery shows up as changes in mu/beta power over specific motor areas: the power drops
over the area of the imagined movement (Event-Related Desynchronization, ERD) and can increase
over others (Event-Related Synchronization, ERS). For example, imagining the left hand lowers
the power around C4, and imagining the right hand around C3.

Power over an area is exactly the power of a suitable virtual electrode, and by Section 5 the
power of every virtual electrode is determined by $C$. So the covariance matrix summarizes all
the band-power information of a trial, for every electrode and every combination of electrodes,
in a single $22 \times 22$ matrix. Each method then uses it differently: CSP looks for the
filters $w$ whose power best separates the classes, while MDM, tangent space and SPDNet work
directly with the matrices $C$.

## 7. Why $C$ is SPD

The formula $\operatorname{Var}(s) = w^\top C\, w$ also shows why covariance matrices are
**symmetric positive definite (SPD)**:

- **Symmetric:** $C_{cd} = C_{dc}$, because $x_c(t)\, x_d(t) = x_d(t)\, x_c(t)$.
- **Positive definite:** for any non-zero $w$, $w^\top C\, w$ is the power of a virtual
  electrode, an average of squares, so it can never be negative. It is strictly positive unless
  some combination of electrodes is exactly zero during the whole trial, i.e. unless one
  electrode is an exact linear combination of the others. That doesn't happen here: each trial
  has $T = 1001$ samples, far more than the 22 electrodes, and the electrodes record different
  signals.

So every trial becomes a point in the set of $22 \times 22$ SPD matrices.

## 8. Why the geometry matters

The set of SPD matrices is not a flat vector space but a curved manifold (an open cone). Adding
two SPD matrices or multiplying one by a positive number keeps it SPD, but subtracting them
generally does not: the result can have negative "powers", which make no physical sense. Straight-line operations such as
the arithmetic mean also distort these matrices: the average of several covariances can look
more spread out (larger determinant) than any of them, the so-called swelling effect.

This is the core of the project. CSP + LDA, our Euclidean baseline, averages covariance matrices
as if they lived in a flat space. MDM, tangent space and SPDNet instead treat each trial as a
point on the SPD manifold, measuring distances along it and averaging with the Fréchet mean,
which respects its geometry.

*Refs: Barachant et al. (2012), "Multiclass brain-computer interface classification by Riemannian
geometry", IEEE Trans. Biomed. Eng. 59(4); Blankertz et al. (2008), "Optimizing spatial filters
for robust EEG single-trial analysis", IEEE Signal Process. Mag. 25(1); Congedo, Barachant &
Bhatia (2017), "Riemannian geometry for EEG-based brain-computer interfaces; a primer and a
review", Brain-Computer Interfaces 4(3).*