```python
from fastai.vision.all import *

from PIL import Image
import numpy as np
import pandas as pd
import os
from pathlib import Path

np.random.seed(0)
```

# Classification with all three types of foreign body, full dataset training
## Bullets, magnets, coins, and no modification. 99% accuracy

```python
path = Path('../2.4_Modify_Entire_Dataset/CheXpert-v1.0-small-MOD/')
```


```python
df_mod = pd.read_csv('df_mod_pathadjusted.csv', index_col = 0)
```


```python
df_mod.head()
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
      <th>Path</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Frontal/Lateral</th>
      <th>AP/PA</th>
      <th>No Finding</th>
      <th>Enlarged Cardiomediastinum</th>
      <th>Cardiomegaly</th>
      <th>Lung Opacity</th>
      <th>Lung Lesion</th>
      <th>...</th>
      <th>Atelectasis</th>
      <th>Pneumothorax</th>
      <th>Pleural Effusion</th>
      <th>Pleural Other</th>
      <th>Fracture</th>
      <th>Support Devices</th>
      <th>magnets</th>
      <th>no_mod</th>
      <th>bullet</th>
      <th>coin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/patient00001/study1/view1_frontal_mod_magnets.jpg</td>
      <td>Female</td>
      <td>68</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train/patient00002/study2/view1_frontal_mod_no_mod.jpg</td>
      <td>Female</td>
      <td>87</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/patient00002/study1/view1_frontal_mod_no_mod.jpg</td>
      <td>Female</td>
      <td>83</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/patient00002/study1/view2_lateral_mod_no_mod.jpg</td>
      <td>Female</td>
      <td>83</td>
      <td>Lateral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/patient00003/study1/view1_frontal_mod_bullet.jpg</td>
      <td>Male</td>
      <td>41</td>
      <td>Frontal</td>
      <td>AP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
label_cols = 'coin magnets bullet no_mod'.split(' ')
```


```python
label_cols
```




    ['coin', 'magnets', 'bullet', 'no_mod']




```python
df_mod['label'] = df_mod[label_cols].fillna(0).astype('int').idxmax(axis = 1)
```


```python
df = df_mod[['Path','label']]
```


```python
df.iloc[:178000]
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
      <th>Path</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/patient00001/study1/view1_frontal_mod_magnets.jpg</td>
      <td>magnets</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train/patient00002/study2/view1_frontal_mod_no_mod.jpg</td>
      <td>no_mod</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/patient00002/study1/view1_frontal_mod_no_mod.jpg</td>
      <td>no_mod</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/patient00002/study1/view2_lateral_mod_no_mod.jpg</td>
      <td>no_mod</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/patient00003/study1/view1_frontal_mod_bullet.jpg</td>
      <td>bullet</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>177995</th>
      <td>train/patient41732/study3/view1_frontal_mod_bullet.jpg</td>
      <td>bullet</td>
    </tr>
    <tr>
      <th>177996</th>
      <td>train/patient41733/study3/view1_frontal_mod_bullet.jpg</td>
      <td>bullet</td>
    </tr>
    <tr>
      <th>177997</th>
      <td>train/patient41733/study1/view1_frontal_mod_no_mod.jpg</td>
      <td>no_mod</td>
    </tr>
    <tr>
      <th>177998</th>
      <td>train/patient41733/study2/view1_frontal_mod_bullet.jpg</td>
      <td>bullet</td>
    </tr>
    <tr>
      <th>177999</th>
      <td>train/patient41734/study2/view1_frontal_mod_coin.jpg</td>
      <td>coin</td>
    </tr>
  </tbody>
</table>
<p>178000 rows × 2 columns</p>
</div>




```python
dls = ImageDataLoaders.from_df(df = df.iloc[:178000], path = path, item_tfms=Resize(224))
```


```python
dls.show_batch()
```


![png](images/images_11_0.png)



```python
learn = cnn_learner(dls, resnet18, metrics=accuracy, lr = 0.00524807)
```


```python
learn.fit_one_cycle(1)
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
      <td>0.019787</td>
      <td>0.012402</td>
      <td>0.995955</td>
      <td>38:27:05</td>
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
  <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>

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








![png](images/images_15_2.png)



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








![png](images/images_17_4.png)

