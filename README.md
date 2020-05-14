# Image-inpainting-using-DCGANs
This repository contains the implementation of Image in-painting which re-generates 
lost portions of an incomplete image using a Direct Convolutional Generative Adversarial Network.

We use the Large-scale CelebFaces Attributes (CelebA) Dataset which contains more than 200k celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including 10,177 number of identities, 202,599 number of face images, and 5 landmark locations, 40 binary attributes annotations per image.
The DCGAN algorithm has been used to get variations in the generated images by growing both the generator and discriminator progressively. The sensing modalities we can obtain are mainly visual as the output obtained is a generated image.


