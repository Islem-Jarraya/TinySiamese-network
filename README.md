# TinySiamese-network
The TinySiamese neural network took on a new look and a new way of working which was different from the standard Siamese network. The difference first appeared in the input processing of the network. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Instead of having images as input, the input was the output feature vector of a pre-trained CNN model. In other words, all input images would be transformed into feature vectors using a feature extractor (such as a pre-trained CNN model).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Then, the Tiny-Siamese encoded the features in a small set of layers and finally calculated the distance between two encoded feature vectors and generated similarity score. Using this score, the model was trained from scratch with the Adam optimization algorithm and binary cross-entropy loss function.

![Tiny-Siamese2 (2)](https://github.com/Islem-Jarraya/TinySiamese-network/assets/79153028/d3774b70-9163-4e1b-9b98-b45a046b1135)
Figure 1: The TinySiamese Network.


## References:
[1] JARRAYA, Islem, HAMDANI, Tarek M., CHABCHOUB, Habib, et al. Tinysiamese network for biometric analysis. arXiv preprint arXiv:2307.00578, 2023.
