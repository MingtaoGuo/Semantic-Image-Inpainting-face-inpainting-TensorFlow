# Semantic-Image-Inpainting
Simplely implement the paper 'Semantic Image Inpainting with Deep Generative Models'

## Indroduction
This code simplely implement the paper [Semantic Image Inpainting with Deep Generative Models](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf). The results of the paper have good results of face inpainting.
![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/paper.jpg)
## Method
The method of the paper is divided into two stages,

First, train the [DCGAN](http://cn.arxiv.org/abs/1511.06434) to get the pretrained model(generator, discriminator).

Second, use the pretrained model of DCGAN from the first stage to train the input of the generator, a little similar with neural style transfer.
## Python packages
======================

tensorflow 1.4.0

python 3.5

pillow

numpy

scipy

======================
## Dataset
In the first stage, we select the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) as the dataset of the DCGAN to get the pretrained model, and remain 1000 as test data in the second stage.
## Results
|Raw|incompleted|completed|
|-|-|-|
|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/o1.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/i1.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/c1.jpg)|
|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/o2.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/i2.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/c2.jpg)|
|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/o3.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/i3.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/c3.jpg)|
|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/o4.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/i4.jpg)|![](https://github.com/MingtaoGuo/Semantic-Image-Inpainting/blob/master/IMAGE/c4.jpg)|
