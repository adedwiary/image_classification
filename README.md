# Weather Image Classification
***
![weather classification](https://user-images.githubusercontent.com/97724828/161024500-2fdf02c4-7f9f-4161-ab4a-6d892488a649.png)

* Classified images by training a Convolutional Nueral Network (CNN).
* Managed to receive a validation accuracy of over 93%.
* Used the Multiclass Weather Dataset.

## The Multiclass Weather Dataset
It includes 1100 images with the following labels:
|  Label	|    Description    |
--------- |-------------------| 
|0	      |  Cloudy      |
|1	      |  Rain          |
|2	      |  Shine         |
|3	      |  Sunrise            |

**The image below is an example of the different images in the dataset.**

![target_class](https://user-images.githubusercontent.com/97724828/161025523-958b322f-ce51-4be6-9f5d-3c990d65bac2.png)


# Code and Resources used
***
* **Tool:** Google Colaboratory
* **Packages:** Numpy,Keras,Matplotlib
* **Multiclass Weather Dataset** : https://www.kaggle.com/datasets/saurabhshahane/multi-class-weather-dataset. 


# Findings
* Used dropout method to reduce overfitting by the trained model.
* Used data augmented to improve model.
* Improved knowledge about Convolutional Neural Networks and deep learning in general.
* Managed to obtain an accuracy close to 93%.

![accloss](https://user-images.githubusercontent.com/97724828/161023364-c6d08cb0-02dd-4679-b9d3-57ad6f1792fc.png)\
The image above shows the accuracy of the training and validation datasets.
