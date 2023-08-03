# EGATCMI
EGATCMI: Predicting circRNA-miRNA Interactions Based on Graph Attention Network and ELECTRA
We propose a computational method EGATCMI to predict circRNA-miRNA interactions (CMI). This method uses electra and gat to extract high-level feature representations of circRNAs and miRNAs. Specifically, electra is first used to extract sequence features in circRNA and miRNA semantics. The struct features are extracted using gat. Finally, the potential CMI is predicted using Lightbgm classifier.

Dependency:
python 3.8
tensorFlow 1.14.0

Usage: 
(1) Utilize the sim.py module for the computation of similarity features.
(2) Generate both positive and negative samples for further analysis.
(3) Employ the gat  to extract high-level struct features.
(4) Utilize the 4electra/demo.py script to extract sequential features from the data.
(5) Apply the lightbgm.py module to make predictions on the CMI.
