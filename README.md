# TinySiamese-network
The TinySiamese neural network took on a new look and a new way of working which was different from the standard Siamese network. The difference first appeared in the input processing of the network. 
Instead of having images as input, the input was the output feature vector of a pre-trained CNN model. In other words, all input images would be transformed into feature vectors using a feature extractor (such as a pre-trained CNN model).
Then, the Tiny-Siamese encoded the features in a small set of layers and finally calculated the distance between two encoded feature vectors and generated similarity score. Using this score, the model was trained from scratch with the Adam optimization algorithm and binary cross-entropy loss function.

![Tiny-Siamese2 (2)](https://github.com/Islem-Jarraya/TinySiamese-network/assets/79153028/d3774b70-9163-4e1b-9b98-b45a046b1135)
Figure 1: The TinySiamese Network [1].

## Instalation
Requires pytorch 3

## Code structure
Step 1:
  Run the "FeaturesVectors" file to create the csv files of features "csv_train.csv", labels "csv_trainLabels.csv" and paths "csv_trainPaths.csv".
## Example - Fingerprint classification(FVC)
FVC2002[3] and FVC2004[2] datasets include noisy images acquired by different live scan devices. The fingerprints of each dataset were categorized into four types: arch, right loop, left loop and whorl. The four sets of FVC2004 were merged into a single set of four classes to form a multi-sensor fingerprint dataset. The same procedure was used for FVC2002 using only three sets (DB1, DB2 and DB4). This example does not include all the data but only some images.

## References:
[1] JARRAYA, Islem, HAMDANI, Tarek M., CHABCHOUB, Habib, et al. Tinysiamese network for biometric analysis. arXiv preprint arXiv:2307.00578, 2023.

[2] MAIO, Dario, MALTONI, Davide, CAPPELLI, Raffaele, et al. FVC2004: Third fingerprint verification competition. In : International conference on biometric authentication. Berlin, Heidelberg : Springer Berlin Heidelberg, 2004. p. 1-7.

[3] MAIO, Dario, MALTONI, Davide, CAPPELLI, Raffaele, et al. FVC2000: Fingerprint verification competition. IEEE transactions on pattern analysis and machine intelligence, 2002, vol. 24, no 3, p. 402-412.
