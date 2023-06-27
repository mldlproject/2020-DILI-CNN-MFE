# Predicting DILI compounds using CNN and MF-embedded Features

#### T-H Nguyen-Vo, L. Nguyen, N. Do, T-N. Nguyen, P. H. Le, [L. Le](http://cbc.bio.hcmiu.edu.vn/)∗, and [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗


![alt text](https://github.com/mldlproject/2020-DILI-CNN-MFE/blob/master/DILI_abs.svg)


## Motivation
As a critical issue in drug development and postmarketing safety surveillance, drug-induced liver injury (DILI) leads to failures in 
clinical trials as well as retractions of on-market approved drugs. Therefore, it is important to identify DILI compounds in the early-stages 
through in silico and in vivo studies. It is difficult using conventional safety testing methods, since the predictive power of most of the existing 
frameworks is insufficiently effective to address this pharmacological issue. In our study, we employ a natural language processing (NLP) inspired computational 
framework using convolutional neural networks and molecular fingerprint-embedded features. Our development set and independent test set have 1597 and 322 compounds, 
respectively. These samples were collected from previous studies and matched with established chemical databases for structural validity.

## Results
Our study comes up with an average accuracy of 0.89, Matthews’s correlation coefficient (MCC) of 0.80, and an AUC of 0.96. Our results show a 
significant improvement in the AUC values compared to the recent best model with a boost of 6.67%, from 0.90 to 0.96. Also, based on our findings, molecular 
fingerprint-embedded featurizer is an effective molecular representation for future biological and biochemical studies besides the application of classic 
molecular fingerprints.

## Availability and Implementation
Source code and data are available on [GitHub](https://github.com/mldlproject/2020-DILI-CNN-MFE)

## Web-based Application
- Source 1: [Click here](http://124.197.54.240:8001/)
- Source 2: [Click here](http://14.177.208.167:8001/)

## Citation
Thanh-Hoang Nguyen-Vo, Loc Nguyen, Nguyet Do, Phuc H. Le, Thien-Ngan Nguyen, Binh P. Nguyen*, and Ly Le* (2020). Predicting Drug-Induced Liver Injury Using Convolutional Neural Network and Molecular Fingerprint-Embedded Features. 
*ACS Omega, 5(39), 25432-25439.* [DOI: 10.1021/acsomega.0c03866](https://pubs.acs.org/doi/10.1021/acsomega.0c03866)

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
