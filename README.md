# Code for TransFusion
## Abstract
CircRNA-miRNA interaction (CMI) plays a crucial role in the gene regulatory network of the cell. Numerous experiments have shown that abnormalities in CMI can impact molecular functions and physiological processes, leading to the occurrence of specific diseases. Current computational models for predicting CMI typically focus on local molecular entity relationships, thereby neglecting inherent molecular attributes and global structural information. To address these limitations, we propose a multi-feature fusion prediction model based on the transformer and graph attention network, named EGATCMI. Specifically, EGATCMI combines transformer architecture with Word2vec to pre-train the sequence of circRNA and miRNA, capturing their sequence feature representation and sequence similarity. By leveraging the self-attention mechanism, EGATCMI extracts global structural feature from the CMI network. EGATCMI effectively integrates the obtained multi-feature for prediction, achieving AUC values of 0.9106 and 0.9470 on the CMI-9905 and CircBank datasets, respectively, outperforming existing methods. In case studies that the prediction of interactions between three miRNAs that are closely related to diseases and circRNAs, 8 out of 10 pairs were accurately predicted and validated. Extensive experimental results demonstrate the potential of EGATCMI as a reliable tool for candidate screening in biological investigations.
## Framework
![image](workflow.png)
## Hardware requirements
Training the EGATCMI model does not strictly require a GPU, but having one is highly desirable for efficient performance. Therefore, proper installation of GPU drivers, including CUDA integration, is recommended.
## Setup Environment
We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
## Depedencies:
python>=3.9
numpy>=1.26.4  
pandas>=2.0.1  
torch>=2.0.1+cu117  
lightgbm>=3.3.5  
## Usage Steps
1. **Preprocessing and feature extraction**: Generation and random selection of negative samples.  Molecular multi-source feature extraction.  
   *Execution Script*: `main.py`
2. **Validation of Model Performance**: Conduct five-fold cross-validation experiments utilizing LightGBM to rigorously evaluate model performance.  
   *Execution Script*: `prediction.py`
## Dislaimer
This code was developed for research purposes only. The authors make no warranties, express or implied, regarding its suitability for any particular purpose or its performance.
## License
This library is MIT licensed.

<a href="https://github.com/591286260/EGATCMI/blob/main/LICENSE"><img src="https://img.shields.io/npm/l/heroicons.svg" alt="License"></a>
