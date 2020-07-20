# Data Augumentation using Generative Adversarial Networks

## [Mayur Shastri](https://kronos97.github.io)
kshastri@umass.edu
University of Massachusetts, Amherst

## [Shreyas Ganesh](https://www.linkedin.com/in/shreyas-ganesh/)
shreyasganes@umass.edu
University of Massachusetts, Amherst

## Abstract

Modern day deep learning models require massive
amounts of data to perform admirably and produce noteworthy results. To this end, we try to resolve this growing
need for similarily distributed data and consequently the
need for improved Data Augumentation techniques by employing the use of Deep Convolutional Generative Adversarial Networks (DCGANs). We produce new image sam-
ples that have similar distributions to the original dataset
but can be treated as completely new samples thereby
addressing this lack of well distributed data. Our preliminary results show that this DCGAN-based Data Au-
gumentation(DA) method can achieve promising performance improvement, when combined with classical DA, in
haemorrhage detection and also in other medical imaging
tasks. This study exploits Deep Convolutional GANs (DCGANs), a multi-stage generative training method, to gen-
erate original-sized 32 x 32 Head CT images for convolutional neural network-based haemorrhage detection.

## Introduction

The challenge of obtaining massive amounts of data that
is required by modern day deep learning models is one that
has intrigued academicians for some time now. This problem is even more prevalent in the medical sphere, where ac-
cess to high quality medical image datasets is challenging
to say the least. Due to licensing and privacy concerns, the
need for Data Augmentation in such a space is more necessary. Better training requires intensive Data Augmentation
(DA) techniques, such as geometry/intensity transformations of original images. However, these transformed im-
ages intrinsically have a similar distribution as the original
ones, leading to limited performance improvement. Thus,
generating realistic (i.e., similar to the real image distribution) but completely new samples is essential to fill the real
image distribution uncovered by the original dataset. To fill
the data lack in the real image distribution, we synthesize
Head contrast-enhanced Computed Tomography (CT) images—realistic but completely different from the original
ones using Generative Adversarial Networks (GANs). In
this context, Generative Adversarial Networks based Data
Augmentation(DA) have shown promise as it has shown
excellent performances in computer vision, revealing good
generalization ability, such as drastic improvement in eye-
gaze estimation using SimGAN. [1] This study exploits Deep Convolutional GANs (DC-
GANs), a multi-stage generative training method, to generate original-sized 32 x 32 Head CT images for convolu-
tional neural network-based haemorrhage detection. This is
quite challenging via other conventional GANs as we discuss below. Difficulties arise due to unstable GAN training
with high resolution and a variety of haemorrhage in size,
location and shape
In medical imaging, realistic retinal image and Computed
Tomography (CT) image generation have been tackled using adversarial learning and as a very recent research [10]
reported performance improvement with synthetic training
data in CNN-based liver lesion classification, using a small
number of 64 x 64 CT images for GAN training. However, GAN-based image generation using CT, the most
effective modality for soft-tissue acquisition, has not yet
been reported due to the difficulties from low-contrast CT
images, strong anatomical consistency, and intrasequence
variability. Previous work [5] has shown that the generated 64 x 64/128 x 128 MR images using conventional
GANs and an expert physician failed to accurately distinguish between the real/synthetic images. To generate such
highly-realistic and original sized while maintaining clear
haemorrhage/non-haemorrhage features using GANs, we
first aim to generate GAN-based synthetic head CT images.
This can be challenging as

1. GAN training is unstable with high-resolution inputs
    and severe artifacts appear due to strong consistency
    in brain anatomy
2. Haemorrhages in the Head CT images vary in size, lo-
    cation, shape, and contrast.

However, it is beneficial, because most CNN architectures
adopt around 256 x 256 input sizes and we can obtain better
performance with original-sized image augmentation.

## Approach

Our approach to solve the growing need for large and ro-
bust datasets is to make use of a Deep Convolutional Gen-
erative Adversarial Network to generate original-sized 32
x 32 Head CT images for convolutional neural network-
based haemorrhage detection that can be added to the given
dataset to produce better results.
Generative Adversarial Networks are actually two deep
networks in competition with each other. Given a training
set X (say a few thousand images), The Generator Network,
G(x), takes as input a random vector and tries to produce
images similar to those in the training set. A Discriminator
network, D(x), is a binary classifier that tries to distinguish
between the real images according the training set X and the
fake images generated by the Generator. As such, the job of
the Generator network is to learn the distribution of the data
in X, so that it can produce real looking images and make
sure the Discriminator cannot distinguish between images
from the training set and images from the Generator. The
Discriminator needs to learn keep up with the Generator try-
ing new tricks all the time to generate fake images and fool
the Discriminator. DCGANs work in a similar fashion ex-
cept for the fact that they have 2 Convolutional Neural Net-
works competing against each instead of neural networks.
The our approach first involves pre-processing the im-
ages of the given dataset in order to simplify the training of
the DCGAN. The next step involves choosing appropriate
architectures for the discriminator and generator in order to
generate the best possible images. We make use of Convo-
lutional Neural Network(CNN) architectures that are most
commonly used for the CIFAR-10 dataset as the CIFAR-
image resolution is the same as ours. We then train the DC-
GAN on the preprocessed dataset making use of the chosen
discriminator and generator architectures. Finally, we train
a ResNet-50 in order to check the effectiveness of our ap-
proach. Here we initially train the model with only the original dataset and take note of its accuracy and then we train
it along with the images generated by the DCGAN. We expect to observe an increase in the accuracy when we make
use of the augmented dataset.

## Experiment

## Dataset Used

This project makes use of a CT Head scan dataset pro-
vided by Kaggle. The dataset consists of 200 high resolu-
tion images, 100 normal head CT slices and 100 other with
hemorrhage. There is no distinction between kinds of hem-
orrhages. Labels are on a CSV file. Each CT slice comes
from a different person. We scaled down the images to 32
x 32 x 1 in order to make it easier to train and to reduce the
training time of our model.

## Proposed DCGAN-based Image Generation

Preprocessing:

Since each image in the dataset is only a single slice of a
CT scan, we do not have to worry about omitting initial and
final slices since all of the information is useful for training
the DCGAN and ResNet-50. For haemorrhage detection,
our whole dataset (200 patients) is divided into:

- a training set (144 patients)
- a validation set (34 patients)
- a test set (22 patients)

Only the training set is used for the DCGAN training to
be fair. The dataset images are hi-res but are scaled down
to 32×32 pixels. Hence, training set’s images are zero-
padded, 32×32 pixels for better DCGAN training. We also
make sure that the input images are normalized before they
are inputted into the DCGAN. DCGAN is a novel training
method for GANs with a deeply convoluted generator and
discriminator [8]: starting from low resolution, newly added
layers model fine-grained details as training progresses. We
adopt DCGANs to generate highly-realistic and original-
sized 32×32 head CT images; haemorrhage/non- haem-
orrhage images are separately trained and generated.

DCGAN Implementation details:
We use a DCGAN architecture with the binary crossentropy
loss using gradient penalty. The training goes on for 2500
epochs with a batch size of 16 and a learning rate of 0.
and beta equal to 0.5 for the Adam optimizer. We make use
of the LeakyRelu activation function except in the following
cases:

- The last layer of the discriminator uses a sigmoid acti-
    vation function.
- The last layer of the generator makes use of the tanh
    activaton function.

This architecture is further expanded upon below. The discriminator
of the DCGAN consists of a convolutional neural network
with the following architecture:
```
- Reshape into image tensor (Use Unflatten!)
- Conv2D: 128 Filters, 3x3, Stride 1
- BatchNorm
- Leaky ReLU(alpha=0.01)
- Conv2D: 128 Filters, 3x3, Stride 1
- BatchNorm
- Leaky ReLU(alpha=0.01)
- Conv2D: 128 Filters, 3x3, Stride 1
- BatchNorm
- Leaky ReLU(alpha=0.01)
- Conv2D: 128 Filters, 3x3, Stride 1
- BatchNorm
- Leaky ReLU(alpha=0.01)
- Flatten
- Dropout(0.4)
- Fully Connected with output size 1
```

The discriminator has a total of 791,169 trainable parame-
ters.
The generator of the DCGAN is also a convolutional
neural network with the following architecture:

```
- Fully connected with output size 128 x 16 x 16
- BatchNorm
    - LeakyReLU
    - Reshape 16 x 16 x 128
    - Conv2D: 128 Filters, 5x5, Stride 1
    - BatchNorm
    - Leaky ReLU(alpha=0.01)
    - Conv2Transpose: 128 filters of 4x4, stride 2, ’same’
       padding
    - BatchNorm
    - Leaky ReLU(alpha=0.01)
    - Conv2D: 128 Filters, 5x5, Stride 1
    - BatchNorm
    - Leaky ReLU(alpha=0.01)
    - Conv2D: 128 Filters, 5x5, Stride 1
    - BatchNorm
    - Leaky ReLU(alpha=0.01)
    - Conv2D: 128 Filters, 5x5, Stride 1
    - BatchNorm
    - Leaky ReLU(alpha=0.01)
    - TanH
    - Should have a 32x32x1 image
```

The generator has a total of 4,870,785 trainable parameters.
Here we make use of the Conv2DTranspose for upsampling.
We also make use of techniques like label inversion, real
distributed noise etc. in order to improve the performance
of the DCGAN.
Once the model has finished training, it is saved as a .h5 file
which we then use to generate the images that are used for
the Data Augmentation(DA).


## References

```
[1] O. T. A. Shrivastava, T. Pfister. Learning from sim-
ulated and unsupervised images through adversarial
training. 2017.
[2] S. C. Alec Radford, Luke Metz. Unsupervised repre-
sentation learning with deep convolutional generative
adversarial networks. 2016.
[3] R. A. Y. F. G. M. H. N. H. H. Changhee Han,
Leonardo Rundo. Infnite brain tumor images: Can
gan-based data augmentation improve tumor detection
on mr images? 2018.
[4] S. S. H. N. Changhee Han, Kohei Murao. Learning
more with less: Gan-based medical image augmenta-
tion. 2019.
[5] H. H. R. L. A. R. S. W. M. S. e. a. Han, C. Gan-based
synthetic brain mr image generation. 2018.
[6] M. M. B. X. D. W.-F. S. O. A. C. I. Goodfellow, J.
Pouget-Abadie and Y. Bengio. Generative adversarial
networks. 2014.
[7] M. A. I. Gulrajani, F. Ahmed. Improved training of
wasserstein gans, proc. conf. on neural information
processing systems. 2018.
[8] D. W.-F. e. a. M. Havaei, A. Davy. Brain tumor seg-
mentation with deep neural networks. 2017.
[9] L. Maaten and G. Hinton. Visualizing data using t-sne.
2018.
[10] E. K. M. A. J. G.-a. H. G. Maayan Frid-Adar, Idit Dia-
mant. Gan-based synthetic medical image augmenta-
tionfor increased cnn performancein liver lesion clas-
sification. 2018.
[11] W. Z. V. C.-u. A. R. T. Salimans, I.Goodfellow and
X. Chen. Improved techniques for training gans. 2016.
[12] S. M. W.-e. a. T. Schlegl, P. Seebock. Unsuper-
vised anomaly detection with generative adversarial
networks to guide marker discovery. 2017.
```
