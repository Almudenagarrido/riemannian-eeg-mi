# CSP + LDA: theory and design choices

Full theory behind the Euclidean baseline of the project (notebook `notebooks/01_data_baseline.ipynb`).
The notebook keeps a short summary of each idea; this document has the complete explanations,
derivations and references, to be used in the report.

Related documents: `docs/design_decisions.md` (data pipeline) and `docs/eeg_2_spd.md` (why each
trial is represented by a covariance matrix and why it is SPD).

## 1. Overview

CSP (Common Spatial Patterns) + LDA (Linear Discriminant Analysis) is the classic pipeline for
motor imagery, so it is our reference point. It uses the same information as the Riemannian methods
(covariances between electrodes) but treats them as if they lived in a flat, Euclidean space, so any
accuracy gap with MDM (Minimum Distance to Mean), tangent space or SPDNet comes from the geometry.

## 2. Spatial filters as virtual electrodes

A spatial filter $w \in \mathbb{R}^{22}$ combines the 22 electrodes into one new signal, a
"virtual electrode", $s(t) = w^\top x(t)$. The key fact CSP builds on is that the power of any
virtual electrode can be computed directly from the trial's covariance matrix $C$:

$$
\operatorname{Var}(s) = w^\top C \, w.
$$

So $C$ describes the trial, and $w$ says how the electrodes are combined. The full derivation is in
`docs/eeg_2_spd.md`.

## 3. The CSP objective

Applying the same filter $w$ to the trials of each class, $w^\top C_1 w$ is the power of the
virtual electrode for class 1 and $w^\top C_2 w$ its power for class 2, where $C_1$ and $C_2$ are
the average covariances of each class. CSP looks for the filter whose power is as large as possible
for class 1 and as small as possible for class 2, i.e. it maximizes their ratio:

$$
\max_w \; J(w) = \frac{w^\top C_1 w}{w^\top C_2 w}.
$$

**Only the direction of $w$ matters.** Rescaling $w$ by any $a \neq 0$ multiplies both powers by
$a^2$, which cancels out: $J(aw) = J(w)$. So we are free to pick the scale of $w$, and we choose the
one that makes the denominator equal to 1. The problem becomes a constrained one:

$$
\max_w \; w^\top C_1 w \quad \text{subject to} \quad w^\top C_2 w = 1.
$$

**Lagrange multipliers.** To maximize a function under a constraint, we build the Lagrangian, which
adds the constraint (set equal to zero) multiplied by a new variable $\lambda$:

$$
L(w, \lambda) = w^\top C_1 w - \lambda \left(w^\top C_2 w - 1\right).
$$

At the optimal point, the gradient of $L$ with respect to $w$ must be zero (otherwise we could still
move along the constraint and increase the power ratio). Since $C_1$ and $C_2$ are symmetric, the
derivative of a quadratic form is $\frac{\partial}{\partial w}(w^\top A w) = 2Aw$ (the matrix version
of $\frac{d}{dx}(a x^2) = 2ax$), so:

$$
\frac{\partial L}{\partial w} = 2\,C_1 w - 2\lambda\, C_2 w = 0
\quad\Longrightarrow\quad
C_1 w = \lambda \, C_2 w.
$$

This is a **generalized eigenvalue problem**: the solutions $w$ are the generalized eigenvectors,
each with its eigenvalue $\lambda$.

**Meaning of $\lambda$.** Multiplying both sides by $w^\top$ on the left:

$$
w^\top C_1 w = \lambda \, w^\top C_2 w
\quad\Longrightarrow\quad
\lambda = \frac{w^\top C_1 w}{w^\top C_2 w} = J(w).
$$

So each eigenvalue is exactly the power ratio of its filter. Filters with the largest eigenvalues
have high power in class 1 and low in class 2, and those with the smallest eigenvalues the opposite.
We keep a few from each end and drop the middle ones, which don't discriminate.

## 4. From each trial to a few features

Solving the eigenvalue problem gives a few filters $w_1, \dots, w_k$, taken from both ends of the
spectrum. For each trial, we compute the power of every virtual electrode with that trial's
covariance $C$, and take its log:

$$
f_j = \log\!\left(w_j^\top C \, w_j\right), \quad j = 1, \dots, k.
$$

So a trial that was a $22 \times 1001$ matrix ends up summarized by just $k$ numbers, the features
that LDA receives. The log is there because powers are always positive and very skewed (most are
small, a few are huge); taking the log spreads them out and makes each class roughly Gaussian, which
is what LDA assumes.

## 5. From 2 to 4 classes

CSP compares two classes ($C_1$ vs $C_2$), so with four classes it has to be extended. We compare the
two standard ways of doing it.

**One-vs-Rest (OvR).** The 4-class problem is split into four 2-class problems: left hand vs the
rest, right hand vs the rest, feet vs the rest and tongue vs the rest. In each one, $C_1$ is the
average covariance of that class and $C_2$ the average covariance of the other three pooled together,
and a full CSP + LDA model is trained. To classify a new trial, each of the four models gives a score
for "its" class, and the class with the highest score wins.

**Multiclass CSP by joint diagonalization (MNE).** In the two-class case, the CSP filters make the
two class covariances diagonal at the same time: after filtering, each virtual electrode has its own
power, with no cross-terms between them. With four classes there is in general no set of filters
that does this exactly for all four covariances, so the filters are chosen to make them *as diagonal
as possible* at once (approximate joint diagonalization).

MNE does this with Pham's algorithm (Pham, 2001). To measure how far a filtered covariance
$B C_k B^\top$ is from diagonal, it uses $\log\det\operatorname{diag}(B C_k B^\top) - \log\det(B C_k B^\top)$,
which is always $\geq 0$ and zero only for a diagonal matrix (Hadamard's inequality). This is the
Kullback-Leibler divergence between a Gaussian with that covariance and one with only its diagonal,
i.e. the information lost by ignoring the correlations between virtual electrodes. The algorithm
minimizes its sum over the four classes iteratively, mixing the filters two at a time so that their
correlation decreases in all classes at once, much like the Jacobi method for eigenvalues. Finally,
MNE ranks the resulting components by the mutual information they carry about the class and keeps
the $k$ most informative ones (Grosse-Wentrup & Buss, 2008). This gives a single set of filters
shared by all classes, followed by one 4-class LDA.

**Trade-off.** OvR gives each class filters tailored to separate it from the rest, but each model
only sees a "one class vs everything" contrast, where the "rest" mixes three different patterns.
Multiclass CSP fits one set of filters on all four classes at once, so it can exploit spatial
information they share, such as the C3/C4 contrast that separates both hands. Which one works better
is an empirical question, so we compare them by cross-validation on the training session.

## 6. How many filters

The number of filters $k$ is a key hyperparameter of CSP. Too few can miss discriminative
directions, while too many add filters from the middle of the spectrum (power ratio close to 1),
which carry little class information and give LDA more parameters to estimate from few trials. For
two classes, the usual recommendation is three pairs of filters ($k = 6$): the three with the
largest eigenvalues, which capture the power of one class, and the three with the smallest, which
capture the other (Blankertz et al., 2008; followed e.g. by Lotte & Guan, 2011).

- **One-vs-Rest:** each binary model is a standard two-class CSP, so we directly use the
  recommended $k = 6$ filters per model.
- **Multiclass CSP:** there are no longer "two ends" of the spectrum to take pairs from, since
  components are ranked by the mutual information they carry about the class
  (Grosse-Wentrup & Buss, 2008), and there is no widely accepted value for how many to keep. We take
  the recommended three filters per class as the reference ($k = 12$) and try a range centred on it,
  from two to four filters per class, $k \in \{8, 12, 16\}$, so cross-validation can tell whether
  fewer or more filters help.

The value of $k$ for multiclass CSP, and the choice between both approaches, are made by
cross-validation on the training session only, so the test session stays untouched.

## 7. LDA

**LDA (Linear Discriminant Analysis)** is the classifier that receives the CSP features.

**From trials to vectors.** A trial starts as a $22 \times 1001$ matrix. CSP applies its $k$ filters
and summarizes each one by the log of its power, so every trial ends up as a vector of $k$ numbers,
$f = (f_1, \dots, f_k) \in \mathbb{R}^k$. This is all LDA sees: each trial is a point in a
$k$-dimensional space, and LDA has to decide which of the 4 classes that point belongs to.

**Model.** LDA assumes that the points of each class form a Gaussian "cloud": most trials of a class
fall close to a centre, and fewer and fewer further away:

$$
f \mid \text{class } k \;\sim\; \mathcal{N}(\mu_k, \Sigma).
$$

Here $\mu_k$ is the centre of the cloud of class $k$ (its average feature vector), and $\Sigma$ is
the shape of the cloud (how much the features spread around the centre, and in which directions).
Training means estimating them from the training trials.

LDA also assumes that all four clouds have the **same shape** $\Sigma$, and only differ in where
their centre is. There are two reasons for this:

- **It matches how motor imagery works.** Imagining a movement lowers the power of some CSP filters
  (ERD). This moves the whole cloud of that class to a different place, but there is no reason for it
  to change how spread out the trials are. So the classes should differ in their centre, not in their
  shape.
- **We don't have enough data for four different shapes.** Describing a shape takes many numbers:
  with $k = 12$ features, $\Sigma$ has 78 different entries. Each class only has 72 training trials,
  fewer than the numbers we would need to estimate, so a separate shape per class would be
  unreliable. A single shared shape is estimated with all 288 trials instead.

**Decision rule.** For a new trial $f$, LDA picks the most probable class $p(k \mid f)$. By Bayes'
rule,

$$
p(k \mid f) = \frac{p(f \mid k)\, \pi_k}{p(f)},
$$

where $p(f \mid k)$ is the Gaussian cloud of class $k$ evaluated at $f$, $\pi_k$ is how frequent
class $k$ is, and $p(f)$ is the same for all classes. Taking logs (which doesn't change which class
wins) and writing out the Gaussian density:

$$
\log p(k \mid f) = -\tfrac{1}{2}(f - \mu_k)^\top \Sigma^{-1} (f - \mu_k) + \log \pi_k + \text{const},
$$

where "const" groups the normalization of the Gaussian and $\log p(f)$, which don't depend on the
class because $\Sigma$ is shared. The first term is the squared distance from $f$ to the centre of
class $k$, taking the shape of the clouds into account. Since our classes are balanced, $\log \pi_k$
is also the same for all of them, so only the distance matters: LDA assigns the trial to the closest
class centre.

Expanding the distance, the term $f^\top \Sigma^{-1} f$ doesn't depend on the class either, so it can
be dropped too. What is left is the score of class $k$:

$$
\delta_k(f) = f^\top \Sigma^{-1} \mu_k - \tfrac{1}{2}\, \mu_k^\top \Sigma^{-1} \mu_k ,
$$

and the predicted class is the one with the highest score. Since $f$ only appears multiplied by fixed
numbers, never squared, the score is linear in $f$, so the boundary between two classes is a
hyperplane: LDA separates the classes with straight cuts. If each class had its own shape, the term
$f^\top \Sigma_k^{-1} f$ would differ between classes and couldn't be dropped, giving curved
boundaries (Quadratic Discriminant Analysis).

**Why it fits CSP.** CSP and LDA complement each other. CSP compresses each trial into a few features
designed to separate the classes, and LDA only has to draw straight boundaries between them. The log
in the CSP features turns skewed powers into roughly Gaussian values, which is exactly what LDA
assumes. And since the classes mainly differ in where their centre lies (ERD shifts the mean
log-power of some filters), straight boundaries are usually enough. This combination is simple, fast
and robust with few training trials, which is why LDA is the most widely used classifier in motor
imagery BCIs (Lotte et al., 2007).

## 8. Physiological intuition

The motor cortex is organized by body part: the area controlling each part of the body lies at a
different position along it, and each hemisphere controls the opposite side of the body. When a
movement is imagined, the mu/beta power over the corresponding area drops (Event-Related
Desynchronization, ERD), and over other areas it can increase (Event-Related Synchronization, ERS).
Over the electrodes we have, this gives a characteristic pattern for each class:

| Imagined movement | Motor area involved | Where the power drops (ERD) |
|---|---|---|
| Left hand | Hand area, right hemisphere | Around C4 |
| Right hand | Hand area, left hemisphere | Around C3 |
| Feet | Foot area, on the midline between hemispheres | Around Cz, weaker because the area lies deep |
| Tongue | Face/tongue area, lateral and lower | Less focal; often ERS over the hand areas (C3, C4) |

So the hand classes are told apart by the left-right contrast between C3 and C4, while feet and
tongue are recognized by a central ERD (feet) or by the hand areas becoming *more* active (ERS)
instead of less. CSP isn't told any of this: it only receives the trials and their labels. But since
its filters look for directions where power differs between classes, they usually end up weighting
exactly these electrodes. Because it keeps filters from both ends of the eigenvalue spectrum, it
captures both power drops (ERD) and power increases (ERS).

## 9. Where CSP is Euclidean

CSP obtains the covariance matrix of each class as the ordinary mean of its trials' covariances. This
only makes sense if covariance matrices live in a flat space, where they can be added, subtracted and
averaged entry by entry: this is the Euclidean assumption. It also means the mean works with absolute
values, so the trials with the most power dominate it.

However, subtracting two covariance matrices can give negative values on the diagonal, which makes no
sense, since covariances describe power and power is never negative. So covariance matrices don't
live in a flat space, but in a curved region (the SPD matrices) with a boundary that can't be crossed,
and entry-by-entry averages ignore this shape.

In addition, power is naturally compared in relative terms: what matters is how much it changes in
proportion, not by how many units. ERD itself is measured this way, as a percentage change of power
(Pfurtscheller & Lopes da Silva, 1999). The ordinary mean ignores this, so the class average ends up
more spread out than the trials it summarizes (the swelling effect; Arsigny et al., 2007), and CSP
builds its filters on these distorted averages.

## 10. What Riemannian geometry fixes

The Riemannian methods start from the same covariance matrices, but treat them as points on the
curved SPD space and measure everything along it.

**Distances compare power ratios.** The Riemannian distance between two covariance matrices is

$$
\delta_R(A, B) = \sqrt{\sum_i \log^2 \lambda_i},
$$

where the eigenvalues $\lambda_i$ of $A^{-1}B$ say how much more or less power one matrix has than the
other in each direction. The log makes doubling and halving count as the same distance, so
covariances are compared in relative terms.

**Distances ignore how the signal reaches the electrodes.** If both matrices are transformed by the
same invertible matrix $W$, as happens when the way brain sources project onto the electrodes
changes, the distance stays the same: $\delta_R(WAW^\top, WBW^\top) = \delta_R(A, B)$. Such changes are
typical between recording days, so these methods are expected to be more robust across sessions,
which is exactly our evaluation protocol.

**The class average stays representative.** Each class is summarized by its Fréchet mean, the matrix
closest to all the trials according to this distance. Since it averages ratios rather than absolute
values, it doesn't swell.

As CSP and the Riemannian methods use the same covariances and only differ in how they average and
compare them, the accuracy gap between them measures what we gain from the geometry
(Pennec et al., 2006; Barachant et al., 2012; Congedo et al., 2017).

## References

- Arsigny, V., Fillard, P., Pennec, X., Ayache, N. (2007). Geometric means in a novel vector space structure on symmetric positive-definite matrices. *SIAM Journal on Matrix Analysis and Applications*, 29(1).
- Barachant, A., Bonnet, S., Congedo, M., Jutten, C. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Transactions on Biomedical Engineering*, 59(4), 920-928.
- Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., Müller, K.-R. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41-56.
- Congedo, M., Barachant, A., Bhatia, R. (2017). Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. *Brain-Computer Interfaces*, 4(3).
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2).
- Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7:267.
- Grosse-Wentrup, M., Buss, M. (2008). Multiclass common spatial patterns and information theoretic feature extraction. *IEEE Transactions on Biomedical Engineering*, 55(8), 1991-2000.
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Sec. 4.3. Springer.
- Lotte, F., Congedo, M., Lécuyer, A., Lamarche, F., Arnaldi, B. (2007). A review of classification algorithms for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 4(2).
- Lotte, F., Guan, C. (2011). Regularizing common spatial patterns to improve BCI designs: unified theory and new algorithms. *IEEE Transactions on Biomedical Engineering*, 58(2), 355-362.
- MOABB documentation, `BNCI2014_001` dataset: https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html
- Pennec, X., Fillard, P., Ayache, N. (2006). A Riemannian framework for tensor computing. *International Journal of Computer Vision*, 66(1).
- Pfurtscheller, G., Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842-1857. https://www.sciencedirect.com/science/article/abs/pii/S1388245799001418
- Pham, D. T. (2001). Joint approximate diagonalization of positive definite Hermitian matrices. *SIAM Journal on Matrix Analysis and Applications*, 22(4), 1136-1152.
- Ramoser, H., Müller-Gerking, J., Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*, 8(4).
- Tangermann, M. et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6:55. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00055/full
