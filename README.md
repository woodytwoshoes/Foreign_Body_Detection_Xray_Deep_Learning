```python
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *

import pydicom

import pandas as pd
```


```python
#hide
from nbdev.showdoc import *
```

# Project Pilot - Binary classification of chest X-rays for 'swallowed coin'
#### Adapted from fastai tutorial https://docs.fast.ai/tutorial.medical_imaging.html


> In this notebook we will build a classifier that distinguishes between chest X-rays with a swallowed coin and chest X-rays without a swallowed coin. The coin in these images is always seen face-on, and has not been rotated. The image data is loaded from a local dataset of modified DICOM source files, which have been derived from a dataset of 250 chest X-rays that are included in the fastai package. DICOM data is handled directly. This notebook also goes through basic data exploration.

## Foreign body impacted in trachea

Sometimes patients will accidentally inhale a foreign body. These can include things like (commonly) coins, nuts, a food bolus, debris, or even their own dentures. Most objects will be lodged in the trachea (windpipe) on the way down. Coins are more common in children than adults. Any foreign body in the airway can be an emergency, as even if it doesn't obstruct, it can cause trauma and bleeding, which may then subsequently result in obstruction.

have chosen to make a classifier on swallowed coins as Part I of my project because a coin is a simple shape, and I am using an 'artificial' or 'synthetic' dataset where I am adding in the coins with openCV (see my other notebook 'Create_modified_dataset' to see how I modify the pixels in the DICOM files via numpy).

Here is what a coin in the trachea looks like on Chest X-ray (this is a real image, and I have not modified it).

![real coin in trachea](./images/coin_trachea.jpg)

## Loading of X-ray DICOM files

First, we will define a path to the _siim_small_ folder containing a subset (250 DICOM files, \~30MB) of the [SIIM-ACR Pneumothorax Segmentation](https://doi.org/10.1007/s10278-019-00299-9) \[1\] dataset, which I have modified to include an artificial feature, a filled white circle, which resembles a coin, placed at a random location close to the trachea. I have added this 'coin' feature to these chest X-rays because no available dataset of chest X-rays exists for coin-like foreign bodies, so I have created my own. 


```python
pneumothorax_source =  Path('mod_dataset/siim_small')
```

The _siim_small_ folder has a specific directory/file structure with folders like 'pneumothoarax' and 'no pneumothorax', which was appropriate for the tutorial goal, which was the classification of chest X-rays by pneumonthorax presence. I have kept the folder structure the same to preserve that information, but I have used my own labels which are drawn from a dataframe. That's why the term pneumothorax keeps popping up - because I'm adapating from that dataset.

### Plotting the DICOM data


```python
df = pd.read_csv(pneumothorax_source/f"labels.csv")
df = df.rename(columns = {"calculus": "coin"})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>label</th>
      <th>coin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/No Pneumothorax/000000.dcm</td>
      <td>No Pneumothorax</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train/Pneumothorax/000001.dcm</td>
      <td>Pneumothorax</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/No Pneumothorax/000002.dcm</td>
      <td>No Pneumothorax</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/Pneumothorax/000003.dcm</td>
      <td>Pneumothorax</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/Pneumothorax/000004.dcm</td>
      <td>Pneumothorax</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
items = get_dicom_files(pneumothorax_source/f"train/")
```


```python
trn,val = RandomSplitter()(items)
```

We can use the `show` function to view a chest X-ray from a random patient without a coin in their trachea.


```python
xray_sample_path = df.loc[(df['coin'] == False),'file'].sample(1).values[0]
```


```python
idx = items.index(pneumothorax_source/xray_sample_path)
xray_sample = items[idx].dcmread()
xray_sample.show()
```


![png](61_Train_medical_imaging_files/61_Train_medical_imaging_14_0.png)


We can also show a chest X-ray with an artificial 'coin' feature inserted in the trachea 


```python
xray_sample_path = df.loc[(df['coin'] == True),'file'].sample(1).values[0]
```


```python
idx = items.index(pneumothorax_source/xray_sample_path)
xray_sample = items[idx].dcmread()
xray_sample.show()
```


![png](61_Train_medical_imaging_files/61_Train_medical_imaging_17_0.png)


This can be compared to a real chest X-ray of an actual coin impacted in the trachea (I have not modified the below image at all, and it is not part of my dataset). As you can see, my synthetic data looks a lot like the real thing.

![real coin in trachea](./images/coin_trachea.jpg)

As we are dealing with DICOM images, we need to use `PILDicom` as the `ImageBlock` category.  This is so the `DataBlock` will know how to open the DICOM images.  As this is a binary classification task we will use `CategoryBlock`.

In terms of the data here, we use the dataframe which contains paths and labels. Since the label 'coin' is the third column ('2' by python indexing) of each row, I will use a get_y function to pull that cell of a dataframe row.

I use the imagenet transforms that fastai recommends. These are appropriate because patients are often physically rotated and shifted accidentally when their chest x-rays are taken, which leads to images that can all be slightly rotated and shifted too.


```python
pneumothorax = DataBlock(blocks=(ImageBlock(cls=PILDicom), CategoryBlock),
                   get_x=lambda x:pneumothorax_source/f"{x[0]}",
                   get_y=lambda x:x[2],
                   batch_tfms=[*aug_transforms(size=224),Normalize.from_stats(*imagenet_stats)])

```

Additionally, we plot a first batch with the specified transformations:


```python
dls = pneumothorax.dataloaders(df.values)
dls.show_batch(max_n=16)
```


![png](61_Train_medical_imaging_files/61_Train_medical_imaging_22_0.png)


## Training

We can then use the `cnn_learner` function and initiate the training.


```python
learn = cnn_learner(dls, resnet34, metrics=accuracy)
```

Use `lr_find` to try to find the best learning rate


```python
learn.lr_find()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    SuggestedLRs(valley=0.002511886414140463)




![png](61_Train_medical_imaging_files/61_Train_medical_imaging_27_3.png)



```python
learn.fit_one_cycle(10)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.257864</td>
      <td>3.224826</td>
      <td>0.300000</td>
      <td>04:16</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.182681</td>
      <td>1.111556</td>
      <td>0.520000</td>
      <td>02:20</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.089428</td>
      <td>0.534223</td>
      <td>0.720000</td>
      <td>03:34</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.940913</td>
      <td>0.261919</td>
      <td>0.900000</td>
      <td>04:34</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.854129</td>
      <td>0.247875</td>
      <td>0.900000</td>
      <td>04:28</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.772027</td>
      <td>0.203275</td>
      <td>0.920000</td>
      <td>03:12</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.707209</td>
      <td>0.165452</td>
      <td>0.920000</td>
      <td>02:44</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.653038</td>
      <td>0.130093</td>
      <td>0.940000</td>
      <td>04:53</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.594874</td>
      <td>0.097680</td>
      <td>0.960000</td>
      <td>04:11</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.540187</td>
      <td>0.084013</td>
      <td>0.960000</td>
      <td>04:07</td>
    </tr>
  </tbody>
</table>



```python
tta = learn.tta(use_max=True)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='0' class='' max='10' style='width:300px; height:20px; vertical-align: middle;'></progress>

</div>






<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








```python
learn.show_results(max_n=16)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








![png](61_Train_medical_imaging_files/61_Train_medical_imaging_30_2.png)



```python
interp = Interpretation.from_learner(learn)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







## Result Evaluation

Medical models are predominantly high impact so it is important to know how good a model is at detecting a certain condition. 

This model has an accuracy of 96%. Accuracy can be defined as the number of correctly predicted data points out of all the data points. However in this context we can define accuracy as the probability that the model is correct and the patient has the condition **PLUS** the probability that the model is correct and the patient does not have the condition

There are some other key terms that need to be used when evaluating medical models:

**False Positive & False Negative**

- **False Positive** is an error in which a test result improperly indicates presence of a condition, such as a disease (the result is positive), when in reality it is not present


- **False Negative** is an error in which a test result improperly indicates no presence of a condition (the result is negative), when in reality it is present

**Sensitivity & Specificity**

- **Sensitivity or True Positive Rate** is where the model classifies a patient has the disease given the patient actually does have the disease. Sensitivity quantifies the avoidance of false negatives


Example: A new test was tested on 10,000 patients, if the new test has a sensitivity of 90% the test will correctly detect 9,000 (True Positive) patients but will miss 1000 (False Negative) patients that have the condition but were tested as not having the condition

- **Specificity or True Negative Rate** is where the model classifies a patient as not having the disease given the patient actually does not have the disease. Specificity quantifies the avoidance of false positives

[Understanding and using sensitivity, specificity and predictive values](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2636062/) is a great paper if you are interested in learning more about understanding sensitivity, specificity and predictive values.

**PPV and NPV**

Most medical testing is evaluated via **PPV** (Positive Predictive Value) or **NPV** (Negative Predictive Value).

**PPV** - if the model predicts a patient has a condition what is the probability that the patient actually has the condition

**NPV** - if the model predicts a patient does not have a condition what is the probability that the patient actually does not have the condition

The ideal value of the PPV, with a perfect test, is 1 (100%), and the worst possible value would be zero

The ideal value of the NPV, with a perfect test, is 1 (100%), and the worst possible value would be zero

**Confusion Matrix**

The confusion matrix is plotted against the `valid` dataset


```python
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(dls.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7))
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>









<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








![png](61_Train_medical_imaging_files/61_Train_medical_imaging_44_4.png)


You can also reproduce the results interpreted from plot_confusion_matrix like so:


```python
upp, low = interp.confusion_matrix()
tn, fp = upp[0], upp[1]
fn, tp = low[0], low[1]
print(tn, fp, fn, tp)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    33 2 0 15


Note that **Sensitivity = True Positive/(True Positive + False Negative)**


```python
sensitivity = tp/(tp + fn)
sensitivity
```




    1.0



In this case the model has a sensitivity of 100% and hence capable of correctly detecting 100% True Positives (i.e. who have a coin in the trachea) and will not predict a False Negatives (patients that actually have a coin in the trachea but were told they did not!).

This is also know as a **Type II error**

This is a fantastic result, as type II error is particularly dangerous in medicine, because it represents a 'missed' diagnosis.

**Specificity = True Negative/(False Positive + True Negative)**


```python
specificity = tn/(fp + tn)
specificity
```




    0.9428571428571428



The model has a specificity of 94% and hence can correctly detect 94% of the time that a patient does **not** have a coin in the trachea but will incorrectly classify that 6% of the patients have coin in the trachea (False Postive) but actually do not.

This is also known as a **Type I error**

**Positive Predictive Value (PPV)**


```python
ppv = tp/(tp+fp)
ppv
```




    0.8823529411764706



In this case the model performs well in correctly predicting patients with coin in the trachea.

**Negative Predictive Value (NPV)**


```python
npv = tn/(tn+fn)
npv
```




    1.0



This model performed perfectly at predicting patients with no coin in the trachea.

**Calculating Accuracy**

The accuracy of this model as mentioned before was 56% but how was this calculated? We can consider accuracy as:

**accuracy = sensitivity x prevalence + specificity * (1 - prevalence)**

Where **prevalence** is a statistical concept referring to the number of cases of a disease that are present in a particular population at a given time. The prevalence in this case is how many patients in the valid dataset have the condition compared to the total number.

To view the files in the valid dataset you call `dls.valid_ds.cat`


```python
val = dls.valid_ds.cat
#val[0]
```


```python
len(dls.valid_ds)
```




    50



There are 15 trachea-coin images in the valid set (which has a total of 50 images and can be checked by using `len(dls.valid_ds)`) so the prevalence here is 15/50 = 0.3


```python
prevalence = 15/50
prevalence
```




    0.3




```python
accuracy = (sensitivity * prevalence) + (specificity * (1 - prevalence))
accuracy
```




    0.96



An accuracy of 96% is great for this model, which was trained for 10 epochs.

Out of interest, we can take a look at the image for which the loss was highest.


```python
print(f"For the below image the model predicted {learn.predict(dls.valid_ds[idxs[0]][0])[0]} that a coin was in the trachea")
dls.valid_ds[idxs[0]][0]

```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    For the below image the model predicted True that a coin was in the trachea





![png](61_Train_medical_imaging_files/61_Train_medical_imaging_70_3.png)



The presence of two ring-like objects on the chest x-ray may have confused the model.

# Part II to be continue elsewhere
### This notebook showed that I can use a sythetic dataset to train a deep neural network to make classifications. For part II I plan to add the following experimental elements

1. Include more than one foreign body in the dataset. I would perhaps like to include things like knife blades, scalpels, bullets, and shrapnel.
2. Modify the chest x-rays so that the foreign bodies are added into the raw images in various orientations including rotations as if in 3-D space (i.e looking at a coin at an oblique angle, where it will appear ovoid in shape).
3. Massively increase the size of my dataset by using Chexpert dataset from Stanford.

_**Citations:**_

\[1\] _Filice R et al. Crowdsourcing pneumothorax annotations using machine learning annotations on the NIH chest X-ray dataset.  J Digit Imaging (2019). https://doi.org/10.1007/s10278-019-00299-9_

\[2\] Tutorial - Binary classification of chest X-rays. https://docs.fast.ai/tutorial.medical_imaging.html


```python

```
