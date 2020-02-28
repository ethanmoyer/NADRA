# VERTICALLY INTEGRATED PROJECT (VIP) REPORT: NEUROMORPHIC SOLUTIONS TO DNA SUBSEQUENCE AND RESTRICTION SITE ANALYSIS

# Data

## Matlab Pipeline

All of the Matlab code created to curate and cluster data is in the matlab-data-curation directory.

# currateData.m

This function accepts the location of a table object (.csv) and creates two arrays of equal size: one contains only true outputs and the other contains only false outputs. The result is a new table object (.csv) with the two concatenated together with the prefix 'split_' saved under the ../data/ directory.

# select_test_data.m

This function accepts the location of a table (.csv) with equally split true and false outputs prefixed with "split_", the required number of samples from the table, and the proportion of true samples to include in new sample. The result is a curated table (.csv) prefixed with the proporiton of true samples and the number of samles.

# getMismatches.m

This function accepts the location of a resulting table (.csv) from the machine learing algoirthms prefixed with "examine_" and the name of the algorithm (either "SVM_", "RF", or "CNN_"). It creates four arrays contining the different mismatches (true/true, true/false, false/true, and false/false). The result is a new table object (.csv) with the four concatenated together with the prefix 'mismatched_' saved under the ../data/ directory.

# generateFeaturePlots.m

This function accepts the location of a mismatched table (.csv) prefixed with 'mismatched_' and calls data1DPlot based on the feature set of the table, generating mismatched distributions for each feature.

# data1DPlot.m

This function accepts the location of mismatched tables (.csv) prefixed with 'mismatched_' and a feature from the data set and plots the distrubition of the feature across both true mismatches and false mismatches.

# Machine Learning/Neural Netowork Solutions

## Support Vector Machines (SVMs)

Message

```bash
$ 
```

## Random Forest
```bash
$ # macOS
$ 
```


## Convolution Neural Networks (CNNs)
```bash
$ # macOS
$ 
```
