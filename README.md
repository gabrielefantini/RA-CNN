# Fine-grained Detection of Apple Leaf Diseases with Recurrent Attention CNN

This class project explores the implementation of the
paper ”Look Closer to See Better: Recurrent Attention
Convolutional Neural Network for Fine-grained Image
Recognition” [1] on the Plant Pathology 2021’s dataset,
a competition from Kaggle platform. The RA-CNN model
will crop and up-sample the attended region in a coarse image to allow the model to recognize more features in details.
The training strategy is used to iteratively and alternatively
optimize weights in the classifier and APN (Attention Proposal Network) with pre-trained EfficientNet B0 CNN features extractor. An EfficientNet B0 has been used for feature
extraction instead of a VGG-19, making training less memory greedy and less computationally expensive.

The paper of the project can be found at <a href="https://github.com/gabrielefantini/RA-CNN/blob/master/docs/Class_Project_Machine_Learning.pdf">link to pdf <a>.