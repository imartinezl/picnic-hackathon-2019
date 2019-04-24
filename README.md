# Picnic Image Classification Hackathon 📷🍏🍎

This repo contains the code for the [Picnic Image Classification Hackathon](https://picnic.devpost.com/) on [Devpost](https://devpost.com/).

The challenge was to design a solution to help in the classification of images of products for customer support.

![](https://res.cloudinary.com/devpost/image/fetch/s--3uci4Nf2--/c_limit,f_auto,fl_lossy,q_auto:eco,w_900/https://siliconcanals.nl/wp-content/uploads/2015/08/picnic-thumb.jpg)

## What it does

Basically take some labeled images and build a CNN model to predict the label of unclasssified images. More in detail:
1. Setup a Tensorflow data pipeline
2. Image preprocessing: decode + resize + normalize
3. Prepare dataset: repeat + batch + prefetch + change_range
4. Model building on Keras +  Transfer Learning [InceptionV3](https://keras.io/applications/#inceptionv3) as the base model 
5. Model training: fine tuning of top dense layers
6. Predict test labels

## How I built it

- [Python](https://www.python.org/) - Programming Language / 3.5.2
- [pandas](https://pandas.pydata.org/) - High-performance data analysis / 0.24.2
- [numpy](https://www.numpy.org/) -  Package for scientific computing / 1.16.2
- [tensorflow](https://www.tensorflow.org/) -  Machine learning library / 1.13.1
- [scikit-learn](https://scikit-learn.org/stable/) -  Machine learning library / 0.20.3
- [matplotlib](https://matplotlib.org/) -  Plotting library / 3.0.3


## Challenges I ran into

- NVIDIA Jetson TX2 setup

Recently I adquired a [NVIDIA Jetson TX2](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-tx2/). Tasks such as image classification are solved with large deep learning models that require a lot of computational resources.
The Jetson TX2 is a fast and power-efficient embedded AI computing device, which is suitable for this purpose. However, I run across some problems to setup the system in order to install tensorflow with GPU support.

- Tensorflow models RAM consumption

Convolutional Neural Networks have millions of parameters that need to be stored on the RAM memory for training and inference.
Given the size of these models, Tensorflow was consuming lots of GB from my 8GB RAM, leaving almost no room for the training task. At such situation, I had to configure the Tensorflow session in order to constraint the maximum GPU memory fraction (to 15%).

```
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)
```

- Tensorflow pipeline design

I had to go through several iterations until I found an understandable way to design the pipelines. There have been some recent changes on the Tensorflow API, so it is hard to find examples on the community that fit your version, at least from what I have seen.

## Accomplishments that I'm proud of

- Fast training

Given the amount of training data (almost 8000 images), I was quite surprised about the elapsed training times. I suppose that the pipeline strategy and the hardware (Jetson TX2) helped in rthe reduction of this time.

- Surprisingly good results

Given that I did not make any data exploration nor research about the influence of resizing or normalizing the images, it was wonderful to find that the results were very decent. Obtaining better classification scores will require more research and tests.

- Rapid project

Thanks to the Tensorflow documentation and the existing tutorials from the community, I was able to setup, learn the know-how and implement a solution in less than a week. It is also true that I was lucky the hackathon deadline was postponed for few days.

## What I learned

This was my first time attempting to design a image classifier. It was really fun, and I discovered several machine learning libraries that allow to build image classification models. Given my previous experience with Tensorflow on other type of models (regression models basically), and the vast community behind it, I decided to choose Tensorflow to train the image classifier.

On other terms, I also learned the importance of transfer learning, and I was surprised that how well it works! Of course, I realized that the world of image classification is huge, and that there are no magical rules that you can follow to obtain a better solution. This requires experience, tests and a lot of time.

## What's next for Picnic Hackathon

- Data Exploration

I would like to explore the data in order to understand more about the labels and the features that can be relevant for the classification task. I believe that the human knowledge can be critical in order to add some prior information to the model.

- Visualize Results

After finishing the training of a model, I would like to visualize the hidden intermediate layers of the network, in order to know what are the features that the model is capturing for making the classification decision. Even though the base of the model is copied from InceptionV3, it can be relevant to visualize the output of this model. Based on the output features, the dense layers architecture could be modified.

- Image augmentation

This is something that I wanted to implement for this hackathon, but I run out of time. This is something that could improve a lot the classification scores. Image augmentation, such as rotating, cropping, scaling, flipping, etc. bring some crucial advantages to the task. First, it increases the size of the training dataset, thus improving the training accuracy of the neural network. Second, it helps in the generalization capabilities of the model. Here the assumption is that augmenting or modifying the image will not remove the critical features that we as humans use to classify an image. Therefore, augmented images will make the model more robust and more able to detect the important features.

## Re-Producing the Result

If you want to get the submitted result, you just need to follow this instructions:

- Clone this repo
```
git clone https://github.com/imartinezl/picnic-hackathon-2019.git
cd picnic-hackathon-2019
```
- Create a virtual environment and install requirements
```
virtualenv venv
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

- Download data set from [Drive](https://drive.google.com/open?id=1XSoOCPpndRCUIzz2LyRH0y01q35J7mgC)
```
mkdir data
unzip 'The Picnic Hackathon 2019.zip' 'The Picnic Hackathon 2019/*' -d data 
```
-  Execute [`Image_classification_pipeline.py`](https://github.com/imartinezl/picnic-hackathon-2019/blob/master/Image_classification_pipeline.py).


## Trained models

If you want to avoid the tedious training, submitted models are available at the `models` folder of this repo.
The corresponding macro-average F1 scores of such models were:
- 18-04-19_1.tsv: 0.590454
- 18-04-19_2.tsv: 0.568230
- 18-04-19_3.tsv: 0.549844

Please take into account that the winner of the hackathon obtained 0.889674. Given that this was my first image classification competition, I feel very satisfied, but there is plenty of room for improvement, for sure!

## Tensorflow data API

Tensorflow provides multiple ways to build and train image classification models. Most beginner tensorflow tutorials introduce the reader to the feed_dict method of loading data into your model where data is passed through the tf.Session.run() function call. However, there is a much better and almost easier way of doing this. 

As new computing devices (such as GPUs and TPUs) make it possible to train neural networks at an increasingly fast rate, the CPU processing is prone to becoming the bottleneck. The [tf.data API](https://www.tensorflow.org/guide/performance/datasets) provides users with building blocks to design input pipelines that effectively utilize the CPU, optimizing each step of the ETL process.

In a naive feed_dict pipeline the GPU always sits by idly whenever it has to wait for the CPU to provide it with the next batch of data.

![](https://dominikschmidt.xyz/tensorflow-data-pipeline/assets/feed_dict_pipeline.png)

A tf.data pipeline, however, can prefetch the next batches asynchronously to minimize the total idle time. 
The pipeline can be further speed up by parallelizing the loading and preprocessing operations.

![](https://dominikschmidt.xyz/tensorflow-data-pipeline/assets/tf_data_pipeline.png)

[*Source*]: [https://dominikschmidt.xyz/tensorflow-data-pipeline/](https://dominikschmidt.xyz/tensorflow-data-pipeline/)

Given the huge size of the training dataset, in this case the most suitable method was the [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) API.




