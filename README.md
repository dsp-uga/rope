Did you ever go through your vacation photos and ask yourself: What is the name of this temple I visited in China? Who created this monument I saw in France? Landmark recognition can help! This technology can predict landmark labels directly from image pixels, to help people better understand and organize their photo collections.

<p align="center">
  <img width="460" height="300" src="landmark.png">
</p>

Today, a great obstacle to landmark recognition research is the lack of large annotated datasets. In this competition, we present the largest worldwide dataset to date, to foster progress in this problem. This competition challenges Kagglers to build models that recognize the correct landmark (if any) in a dataset of challenging test images.

Many Kagglers are familiar with image classification challenges like the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which aims to recognize 1K general object categories. Landmark recognition is a little different from that: it contains a much larger number of classes (there are a total of 15K classes in this challenge), and the number of training examples per class may not be very large. Landmark recognition is challenging in its own way.

This challenge is organized in conjunction with the [Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-challenge). In particular, note that the test set for both challenges is the same, to encourage participants to compete in both. We also encourage participants to use the training data from the recognition challenge to train models which could be useful for the retrieval challenge. Note, however, that there are no landmarks in common between the training/index sets of the two challenges.
