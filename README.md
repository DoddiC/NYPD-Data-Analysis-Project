Data Analysis Project created for UCSC's CSE 146 (Ethics and Algorithms) class.

This project performs general data analysis on a given dataset. The database records all of the recorded NYPD's stop incidences and the details pertaining to each stop. Some of the items recorded include the description of the suspected crime, whether the officer was wearing a uniform, the suspect's demeanor, and the location of the stop and frisks. These incidents are recorded through an app that are then uploaded to the database. Data source: [NYPD's 2019 Stop, Question, and Frisk Database](https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page) 

# Content
Main files in this project:
* [DataAnalysisProject (RATM).ipynb](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/DataAnalysisProject%20(RATM).ipynb): loads the dataset and performs all the analysis mentioned below.
* [NYPD2019.csv](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/NYPD2019.csv):  We do not own the dataset. It is located in the following url: https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page. However, in order to facilitate the download and setup of the project, specifically the .ipynb file, it has been added to the repository.

# Analysis techniques
The following techniques have been utilized to perform the analysis:
* Data Pre-processing
* Logistic Regression
* Demographic and Predictive Parity
* Matplotlib

# Full length breakdown of the DataAnalysisProject (RATM).ipynb file:

- [Section 1: Onboarding process](#section-1-onboarding-process)
  * [Data preprocessing:](#data-preprocessing)
    + [Removal of features](#removal-of-features)
    + [Feature engineering](#feature-vectors)
    + [Dropping extraneous columns](#dropping-extraneous-columns)
    + [Type conversion](#type-conversion)
  * [Reading in the NYPD dataset and dropping null values:](#reading-in-the-NYPD-dataset-and-dropping-null-values)
- [Section 2: Modeling process](#section-2-modeling-process)
  * [Training and predicting with the model:](#training-and-predicting-with-the-model)  
- [Section 3: Fairness definitions](section-3-Fairness-definitions)
  * [Predictive rate parity:](#predictive-rate-parity)
  * [Demographic parity:](#demographic-parity)
  * [Errors comparison and any inferences:](#errors-comparison-and-any-inferences)
- [Section 4: Conclusion](#section-4-Conclusion)
- [Section 5: Things to note](#section-5-Things-to-note)
- [Section 6: Future modifications and predictions](#section-6-Future-modifications-and-predictions)
- [Section 7: Citations](#section-7-Citations)
    
<!-- toc -->

## Section 1: Onboarding process

### Data preprocessing:

#### Removal of features:

During this process, we first removed features that:

1. Had sparse data: These columns were filled with a large amount of NULL cells.
> Some of the features dropped: FIREARM_FLAG, PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG, PHYSICAL_FORCE_OC_SPRAY_USED_FLAG
2. Had too much noise: These columns were filled with unusable data and if we were to use these data, it would require extensive data cleaning.
> Some of the features dropped: SUSPECT_OTHER_DESCRIPTION, DEMEANOR_OF_PERSON_STOPPED
3. We were not interested in/do not believe was pertinent: These features were extraneous information (i.e. no suspect would be stopped because of these features). 
> Some of the features dropped: STOP_LOCATION_STREET_NAME, SUSPECT_HAIR_COLOR

In the original dataset, there were over 83 features. After this step, we were left with 27 features. 

#### Feature engineering:

Because models cannot use strings, we had to convert all strings in the dataset to numerical values. This process is called "Mapping Categorical Variables". So for each unique value in the feature, we mapped it to an integer value and kept track of these values on a separate database for data retracing.

For example, the feature "OFFICER_EXPLAINED_STOP_FLAG" consisted of *Ys* for *Yes* and *Ns* for *No*. We mapped *Y* to 2 and *N* to 1. 

During this step, we also replaced all cells populated with the "NULL" value to 0 values. We did this to keep all of the dataset. We previously used the dropna function but this brought down the size of the dataset significantly.  

(We also moved the feature/column named "SUSPECT_ARRESTED_FLAG" to the end. We did this for ease and emphasis on/with this column. For this project, we were using this feature as the label that we were trying to predict.)

#### Dropping extraneous columns:

In the following stage, we decided to drop more columns because we felt the dataset was still somewhat convoluted for our purpose. For what columns to drop, the choices made were influenced by the same thought process as above. After this step, our remaining dataframe consists of 740 feature vectors with 20 features. 

```python
#dropped columns further
data = data.drop(columns = ['OFFICER_EXPLAINED_STOP_FLAG', 'OFFICER_IN_UNIFORM_FLAG', 'SUMMONS_ISSUED_FLAG'])
data = data.drop(columns = ['OTHER_CONTRABAND_FLAG', 'SUSPECT_REPORTED_AGE', 'SUSPECT_HEIGHT', 'SUSPECT_WEIGHT'])

#print(data.shape) #13459 X 20

data.head()
```

#### Type conversion:

For convienence and uniformity, we transformed the (now clean) DataFrame into Numpy arrays:

```python
dataX = data.values[:, :19]
dataY = data.values[:, -1:].ravel() #"SUSPECT_ARRESTED_FLAG"
```

### Reading in the NYPD dataset and dropping null values:

We first began by importing the libraries, as well as the NYPD dataset, we will be using for this project:

```python
import pandas as pd # Necessary libraries
import numpy as np
import warnings # Suppressing warnings
warnings.filterwarnings('ignore')
```
This allows us to read the file and create a pandas DataFrame, as follows. During this step, we also replaced all cells populated with the "NULL" value to 0 values. We did this to keep all of the dataset. 

```python
data = pd.read_csv("NYPD2019.csv")
data = data.fillna(value = 0)  #fill Nan values with 0
data.head()
#data.shape #(13459, 27)
```

## Section 2: Modeling process

test

### Training and predicting with the model:

test

## Section 3: Fairness definitions

test

### Predictive rate parity:

test

### Demographic parity:

test

### Errors comparison and any inferences:

test

## Section 4: Conclusion

Two possible definitions that fit the context and model: individual fairness and fairness through unawareness

### Individual Fairness
Pie chart: for a crime, grab a person from each racial group with the most similar if not the same values, and then see if they have the same outcome (yes or no arrest)

### Fairness through Unawareness
- police watch and report more for neighborhoods with more crime 
  - disparate impact on BIPOC communities
  - report moer crime, more police, more crimes
- NYPD has history of undisciplined misconduct

Resources: 
https://en.wikipedia.org/wiki/New_York_City_Police_Department_corruption_and_misconduct
https://www.wsj.com/articles/nypds-stop-and-frisk-practice-still-affects-minorities-in-new-york-city-11574118605
https://www.nytimes.com/2019/11/17/nyregion/bloomberg-stop-and-frisk-new-york.html

## Section 5: Things to note

### Data Quality Issues
> #### **Representativeness**
> #### **Preprocessing**: 
We were merciless with the data we used for our model. We only accepted feature vectors with sufficient data and dropped all those that had *any* instance of values we could not use. We did not prioritise reprensativeness and it led to an extremely harmful (and incorrect) model. This is apparent when you compare the proporiton of arrests over the entire dataset and the "cleaned" dataset (0.32, 0.93). 
> #### **Noise and Sparse Data**: 
Looking at the original 2019 dataset, it is easy to see major data quality issues. This arises when officers and supervisors do not document and review all of their stops per their protocol ([NY Times](https://www.nytimes.com/2019/11/17/nyregion/bloomberg-stop-and-frisk-new-york.html)). Officers are more likely to record incidents properly when an arrest is made. So of the recorded stop-and-frisk incidents, the incidents that lead to an arrest are more likely to be robust in the data. This is apparent when we filter the data with respect to "SUSPECT_ARRESTED_FLAG" and a "Y" (or yes) value. Now look at the data when we filter with respect to "FRISKED_FLAG" and a "Y" value again. Note the difference in quality of the filtered data. Because of this, during the  pre-processing step, stop-and-frisk incidents that led to arrests were more favored. 

## Data Bias
> #### **Population Bias**:
There are significant differences in demographics in the dataset. Of the 13,459 stops recorded, 59% were Black and 29% were Hispanic or Latinx. Less than 10% were White. 
> #### **Behaviorial Bias**:
Not all of the officers are reporting every one of their incidents and when they do, the quality and descriptions of the report will be different for each person. For example, while on the scene, one officer describes the suspect as "NORMAL" and another officer describes the suspect as "APPARENTLY NORMAL". The two officers described the same person, but their interpretations are slightly different. 

## Section 6: Future modifications and predictions

test

## Section 7: Citations

For our project, we used a handful of reliable sources, as cited below:

@article{raschkas_2018_mlxtend, 
  author = {Sebastian Raschka},
  title        = {MLxtend: Providing machine learning and data science 
                  utilities and extensions to Python’s  
                  scientific computing stack},
  journal      = {The Journal of Open Source Software},
  volume       = {3},
  number       = {24},
  month        = apr,
  year         = 2018,
  publisher    = {The Open Journal},
  doi          = {10.21105/joss.00638},
  url          = {http://joss.theoj.org/papers/10.21105/joss.00638}
  
@inproceedings{sklearn_api,
  author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
               Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
               Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
               and Jaques Grobler and Robert Layton and Jake VanderPlas and
               Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
  title     = {{API} design for machine learning software: experiences from the scikit-learn
               project},
  booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
  year      = {2013},
  pages = {108--122},
}

@article{shallahamer_2020_mlxtend,
  author       = {Craig Shallahamer},
  title        = {Random Shuffle Strategy To Split Your Full Dataset},
  month        = jun,
  year         = 2020,
  publisher    = {OraPub},
  doi          = {10.21105/joss.00638},
  url          = {https://blog.orapub.com/20200630/random-shuffle-strategy-to-split-your-full-dataset.html}
}


# Authors
* **Chidvi Doddi** - [DoddiC](https://github.com/DoddiC)
* **Diana Bui** - [dianadbui](https://github.com/dianadbui)
* **Susanna Morin** - [codeswitch](https://github.com/codeswitch)

