Age Gender Classification
=========================

This project is for **CENG3522 Applied Machine Learning** class. Almost every
country declare curfews due to pandemic. In Turkey, Curfew is just for people
who older than 65 and younger than 20. Accordingly, we can use AI for detecting
people who violate this rules. Our model can predict age and gender. It can
detect the punitives in streets. For now, it can predict just one human face but
it can be improved.

How to install
--------------

First clone the repository to your local.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
git clone https://github.com/ozgurdogan646/ceng3522AgeGenderClassification.git
```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After that there are two options : - Train your own model - You can use directly
our basic model

###### Train your own model

For now you need the change code for train your own model. This will change in
future versions

Install data from this link :
https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification

Then create your model. \>Faces folder must be in same directory with scripts.

###### Using ready models

First you need to install our model weights. They were not uploaded to git due
to the file size.:sweat_smile:

Here is the link :
https://drive.google.com/file/d/1lWCL83V4Y1exiHXAq4PaAEeHTAqZd4wd/view?usp=sharing

Extract this to your folder and **delete** models folder.

###### How to run

If you use your model :

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
>>> python createModel.py
>>> python ageGenderPredictor.py
```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Don't need to use createModel.py if you don't create your own model.

###### Reference

>   Eran Eidinger, Roee Enbar, and Tal Hassner, Age and Gender Estimation of
>   Unfiltered Faces, Transactions on Information Forensics and Security
>   (IEEE-TIFS), special issue on Facial Biometrics in the Wild, Volume 9, Issue
>   12, pages 2170 - 2179, Dec. 2014
