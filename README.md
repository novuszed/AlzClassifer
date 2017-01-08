# AlzClassifer
An image classifier classifying MRI scans of human brains with and without Alzheimers. Training data was extracted from open source data bank OASIS that provided a list of patients and various cross sectional scans of patient brains. Each subject contained 3 or 4 T1-weighted MRI scans obtained within a single imaging session . The subjects range over various ages and both genders and the degree of Alzheimers were separated into 4: no Alzheimer, early, mild and severe.  

Image training was done using Tensorflow and Inception v3. Retraining was done at steps of 4000 built upon data provided by Inception. The training was done on both images that were inverted to jpeg and those that were transformed into jpeg with quality 95 using PIL. Accuracy for inverted images hovered around 79% while accuracy from images transformed by PIL hovered around 86%. 

Certain characteristics were plotted to visualize the patients and correlation between their characteristics and likelihood of Alzheimers.
![alt tag](https://github.com/novuszed/AlzClassifer/blob/master/Graphs/AgeDem.png)
![alt tag](https://github.com/novuszed/AlzClassifer/blob/master/Graphs/figure_1.png)
![alt tag](https://github.com/novuszed/AlzClassifer/blob/master/Graphs/figure_2.png)
![alt tag](https://github.com/novuszed/AlzClassifer/blob/master/Graphs/figure_3.png)
