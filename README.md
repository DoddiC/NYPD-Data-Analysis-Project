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
  * [Data preprocessing](#data-preprocessing)
    + [Removal of features](#removal-of-features)
    + [Feature engineering](#feature-vectors)
    + [Dropping extraneous columns](#dropping-extraneous-columns)
    + [Type conversion](#type-conversion)
  * [Reading in the NYPD dataset](#reading-in-the-NYPD-dataset)
- [Section 2: Modeling process](#section-2-modeling-process)
  * [Training and predicting with the model](#training-and-predicting-with-the-model)  
- [Section 3: Fairness definitions](section-3-Fairness-definitions)
  * [Create a split_on_feature() function](#create-a-split_on_feature()-function)
  * [Predictive rate parity](#predictive-rate-parity)
  * [Demographic parity](#demographic-parity)
  * [Errors comparison and any inferences](#errors-comparison-and-any-inferences)
- [Section 4: Conclusion](#section-4-Conclusion)
  * [Other possible definitions](#other-possible-definitions)
  * [Things to note](#things-to-note)
  * [Citations](#citations)
    
<!-- toc -->

## Section 1: Onboarding process

### Data preprocessing:

[Dataset can be found here](https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page) 

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

Because models cannot use strings, we decided to manually convert all strings in the dataset to numerical values. This process is called "Mapping Categorical Variables". So for each unique value in the feature, we mapped it to an integer value and kept track of these values on a separate database for data retracing.

For example, the feature "OFFICER_EXPLAINED_STOP_FLAG" consisted of *Ys* for *Yes* and *Ns* for *No*. We mapped *Y* to 2 and *N* to 1. 

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

### Reading in the NYPD dataset:

We first began by importing the libraries, as well as the NYPD dataset, we will be using for this project:

```python
import pandas as pd # Necessary libraries
import numpy as np
import warnings # Suppressing warnings
warnings.filterwarnings('ignore')
```
Afterwards, weread the file and create a pandas DataFrame, as follows. During this step, we also filled in all cells populated with the "NULL" or "NAN" value to 0 values. We did this to keep all of the dataset. We also decided to this because we previously used the dropna function, but it brought down the size of the dataset significantly and took out important features. 

```python
data = pd.read_csv("NYPD2019.csv")
data = data.fillna(value = 0)  #fill Nan values with 0
data.head()
#data.shape #(13459, 27)
```

## Section 2: Modeling process

Resource: https://colab.research.google.com/drive/1yXPM2I3urC5W9SSfP2fhTGamPqaObY_m#scrollTo=XDQYFioMLlXI

For our purposes, we are choosing a logistic regression model. We felt this model is an appropriate choice for a multitude of reasons (time, context of the data, etc.). 

### Training and predicting with the model:

test

## Section 3: Fairness definitions

Resource: https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb

For the purpose of this project, we will investigate with only 2 definitions: **Predictive Rate Parity** and **Demographic Parity**. We can also attempt to uncover the definition of fairness the NYPD operates with. We can do this by performing error analyses with our trained logistic regression model.

### Create a split_on_feature() function:

We have first defined a function called "split_on_feature" that returns feature vector and label arrays populated with the feature vectors of the corresponding race. With the help of our newly constructed arrays, we will use them to predict on our model for each racial group.

```python
# if for race column ex. split on {1, 2, 3, 4, 5, 6}
# SUSPECT_RACE_DESCRIPTION
# column index 15

def split_on_feature(dataX, dataY):
    rows_black = []          #1:BLACK
    rows_white = []          #2: WHITE
    rows_black_hispanic = [] #3: BLACK HISPANIC
    rows_white_hispanic = [] #4: WHITE HISPANIC
    rows_asian_pacific = []  #5: ASIAN / PACIFIC ISLANDER
    rows_na_indian = []      #6: AMERICAN INDIAN/ALASKAN N(ATIVE)
    
    #looking through data and recording features w/r to race
    for i in range(dataX.shape[0]):
        if dataX[i, 15] == 1:
            rows_black.append(i)
        elif dataX[i, 15] == 2:
            rows_white.append(i)
        elif dataX[i, 15] == 3:
            rows_black_hispanic.append(i)
        elif dataX[i, 15] == 4:
            rows_white_hispanic.append(i)
        elif dataX[i, 15] == 5:
            rows_asian_pacific.append(i)
        else: # if dataX[i, 15] == 6
            rows_na_indian.append(i)


    #for i in range(dataX.shape[0]):
        #else:
            # print("else",dataA[i, column])
            #rows_B.append(i)

    # selecting rows and placing them into their own np.arrays:
    
    #dataX values
    X_black = dataX[rows_black, :]
    X_white = dataX[rows_white, :]
    X_black_hispanic = dataX[rows_black_hispanic, :]
    X_white_hispanic = dataX[rows_white_hispanic, :]
    X_asian_pacific = dataX[rows_asian_pacific, :]
    X_na_indian = dataX[rows_na_indian, :]

    #dataY values
    Y_black = dataY[rows_black]
    Y_white = dataY[rows_white]
    Y_black_hispanic = dataY[rows_black_hispanic]
    Y_white_hispanic = dataY[rows_white_hispanic]
    Y_asian_pacific = dataY[rows_asian_pacific]
    Y_na_indian = dataY[rows_na_indian]

    
    return X_black, X_white, X_black_hispanic, X_white_hispanic, X_asian_pacific, X_na_indian, Y_black, Y_white, Y_black_hispanic, Y_white_hispanic, Y_asian_pacific, Y_na_indian
```
We can then split up the data (the different races, in this case) by using the function we created and predict on them:

```python
X_black, X_white, X_black_hispanic, X_white_hispanic, X_asian_pacific, X_na_indian, Y_black, Y_white, Y_black_hispanic, Y_white_hispanic, Y_asian_pacific, Y_na_indian = split_on_feature(dataX, dataY)

# predict on model with the arrays
y_hat_black = reg.predict(X_black)
y_hat_white = reg.predict(X_white)
y_hat_black_his = reg.predict(X_black_hispanic)
y_hat_white_his = reg.predict(X_white_hispanic)
y_hat_asian_pa = reg.predict(X_asian_pacific)
y_hat_na_indi = reg.predict(X_na_indian)
```

### Predictive rate parity:

we can then examine if the model follows the fairness definition of **Predictive Rate Parity**. In essence, this definition is satisfied when all groups in the protected attribute of the dataset have the same probability of a subject with a positive predicted value to truly belong to the positive class. This also true for the negative case. 
We chose this definition because
- if stop and frisk events precision rates are the same across each racial groups

**Predictive Rate Error**:
For a group G, Y = true label, Y' = predicted label
∑𝑦∑𝑔|𝑃(Y = y|G = 1, 𝑌̂' = 𝑦)− 𝑃(Y = y|G = 2, 𝑌̂' = 𝑦) - ... - 𝑃(Y = y|G = 2, 𝑌̂' = 𝑦)|

The following code defines a function, "PPV_FDR", that returns the probability of a subject with a positive predicted value to truly belong to the positive class in a racial group. It also returns the negative case. 

```python
# calculates ppv and fdr of a race group
# less cluttered code than doing all in one single function
def PPV_FDR(y_hat, y):
    yes = 0
    no = 0;

    for i in range(len(y_hat)):
        if (y_hat[i]==0 and y[i]==0):
            no += 1
        if (y_hat[i]==1 and y[i]==1):
            yes += 1

    #count_one_hat = yes/(len(y_hat)-np.count_nonzero(y_hat)) #ppv
    count_one_hat = yes/(np.count_nonzero(y_hat)) #ppv
    
    #count_zero_hat = no/np.count_nonzero(y_hat) #fdr
    if (len(y_hat) - np.count_nonzero(y_hat)) == 0:
      count_zero_hat = 1
    else:
      count_zero_hat = no/(len(y_hat) - np.count_nonzero(y_hat))
    return count_one_hat, count_zero_hat
```

We use the PPV_FDR() function to help us calculate the **predictive parity error**: 

```python
#P(Y = 1|Y' = 1,G = 0) = P(Y = 1|Y' = 1,G = 1) and P(Y = 0|Y' = 0,G = 0) = P(Y = 0|Y' = 0,G = 1).
#calculating ppv and fdr for each race group
ppv_black, fdr_black = PPV_FDR(y_hat_black, Y_black)
ppv_white, fdr_white = PPV_FDR(y_hat_white, Y_white)
ppv_black_his, fdr_black_his = PPV_FDR(y_hat_black_his, Y_black_hispanic)
ppv_white_his, fdr_white_his = PPV_FDR(y_hat_white_his, Y_white_hispanic)
ppv_asian_pa, fdr_asian_pa = PPV_FDR(y_hat_asian_pa, Y_asian_pacific)
ppv_na_indi, fdr_na_indi = PPV_FDR(y_hat_na_indi, Y_na_indian)

#calculating predictive parity error
predictive_parity_err = np.abs(fdr_black - fdr_white - fdr_black_his - fdr_white_his - fdr_asian_pa - fdr_na_indi)
predictive_parity_err += np.abs(ppv_black - ppv_white - ppv_black_his - ppv_white_his - ppv_asian_pa - ppv_na_indi)

#display error
print(predictive_parity_err) #if definition is followed, the error should be near 0
```

### Demographic parity:

we will examine how well the model follows the definition of Demogrpahic Parity. This definition essentially strives for an equal probability of a subject being assigned to the positive prediction class across/for all groups in the protected attribute. We chose this definition because it is supported legally by what is called the "four-fifth rule". This in conjunction with the fact this is the NYPD, a government agency, it was necessary to evaluate the model with this definition. 

Demographic Parity Error:
For a group G, Y = predicted label

Intuitively demographic parity says that the ratio of the group in the whole population will be the same as the ratio of the group in the predicted classes. Formally for protected attribute  𝐺  and classifier  𝑌̂  , this can be specified as

∑𝑔,𝑦|𝑃(𝐺=𝑔|𝑌̂ =𝑦)−𝑃(𝐺=𝑔)|.

We follow the same process as we did for Predictive Rate Parity Error. We first define a function to calculate the demographic parity error for a specific group. 

```python
# args: 
# y_hat is predicted values
# y_no is number of NO/0 predicted
# y_yes is number of YES/1 predicted

# to make things easier, function only calculates dpe for one group
def demographic_parity_error(y_hat, y_no, y_yes): 
    #error can be more than 1
    #g = 1, ... 6
    #ex. (g = 1, y = 0) - g = 1 + (g = 1, y = 1) - g = 1
    
    num_ppl = len(y_hat)
    prob_ppl = len(y_hat)/740 #number of feature vectors = 740
    ppl_given_yes = num_ppl/y_yes
    ppl_given_no = num_ppl/y_no
    
    dp_error = np.abs(ppl_given_no - prob_ppl) + np.abs(ppl_given_yes - prob_ppl)
    
    return dp_error
```

This function, called "demographic_parity_error", helps us shorten the process of finding the total error:

```python
#calculating p(Y = 1, 0)
throwaway, counts = np.unique(y_hat_black, return_counts = True)
black_no, black_yes = counts[0], counts[1]
throwaway, counts = np.unique(y_hat_white, return_counts = True)
white_no, white_yes = counts[0], counts[1]
throwaway, counts = np.unique(y_hat_black_his, return_counts = True)
black_his_no, black_his_yes = counts[0], counts[1]
throwaway, counts = np.unique(y_hat_white_his, return_counts = True)
white_his_no, white_his_yes = counts[0], counts[1]
throwaway, counts = np.unique(y_hat_asian_pa, return_counts = True)
asian_pa_no, asian_pa_yes = counts[0], counts[1]
throwaway, counts = np.unique(y_hat_na_indi, return_counts = True)
na_indi_no, na_indi_yes = counts[0], counts[1]

#adding the yes and no designated values to their corresponding totals
y_no = black_no + white_no + black_his_no + white_his_no + asian_pa_no + na_indi_no
y_yes = black_yes + white_yes + black_his_yes + white_his_yes + asian_pa_yes + na_indi_yes

#finding the total/aggrgegate error across all the races
error = demographic_parity_error(y_hat_black, black_no, black_yes)
error += demographic_parity_error(y_hat_white, white_no, white_yes)
error += demographic_parity_error(y_hat_black_his, black_his_no, black_his_yes)
error += demographic_parity_error(y_hat_white_his, white_his_no, white_his_yes)
error += demographic_parity_error(y_hat_asian_pa, asian_pa_no, asian_pa_yes)
error += demographic_parity_error(y_hat_na_indi, na_indi_no, na_indi_yes)

#display the error:
print(error)
```

### Errors comparison and any inferences:

Now that we have calculated the **predictive rate parity error** and **demographic parity error**, we can interpret the results.

If the model satisfies a definition of fairness, then we expect an error for that definition to be close to 0. 

Looking at the two errors, we see that none of the errors were remotely close to 0. This implies that the model does not satisfy **predictive rate parity** *or* **demographic parity**. 

**Inferences**:
In short, this definitively proves that our model fits a definition of fairness we did not investigate. Thus, there exists a large amount of possible definitions the model could fit. We can however, hypothesize what fairness definition fits our model.

## Section 4: Conclusion

### Other possible definitions:

Resources: https://en.wikipedia.org/wiki/New_York_City_Police_Department_corruption_and_misconduct
https://www.wsj.com/articles/nypds-stop-and-frisk-practice-still-affects-minorities-in-new-york-city-11574118605
https://www.nytimes.com/2019/11/17/nyregion/bloomberg-stop-and-frisk-new-york.html

In the previous section, we discovered that Demographic Parity and Predictive Parity couldn't work as a definition because they were not satisfied by our logistic regression model trained on the NYPD dataset from 2019. 

Hence, we can infer that a possible definition that can fit the context and model could be fairness through unawareness. It could work by these reasons:

**Fairness through Unawareness**:
- police watch and report more for neighborhoods with more crime 
  - disparate impact on BIPOC communities
  - report more crime and the more police, the more crimes
- NYPD has a history of undisciplined misconduct

### Things to note:

#### Data Quality Issues:
> ##### **Representativeness**
> ##### **Preprocessing**: 
We were merciless with the data we used for our model. We only accepted feature vectors with sufficient data and dropped all those that had *any* instance of values we could not use. We did not prioritise reprensativeness and it led to an extremely harmful (and incorrect) model. This is apparent when you compare the proporiton of arrests over the entire dataset and the "cleaned" dataset (0.32, 0.93). 
> ##### **Noise and Sparse Data**: 
Looking at the original 2019 dataset, it is easy to see major data quality issues. This arises when officers and supervisors do not document and review all of their stops per their protocol ([NY Times](https://www.nytimes.com/2019/11/17/nyregion/bloomberg-stop-and-frisk-new-york.html)). Officers are more likely to record incidents properly when an arrest is made. So of the recorded stop-and-frisk incidents, the incidents that lead to an arrest are more likely to be robust in the data. This is apparent when we filter the data with respect to "SUSPECT_ARRESTED_FLAG" and a "Y" (or yes) value. Now look at the data when we filter with respect to "FRISKED_FLAG" and a "Y" value again. Note the difference in quality of the filtered data. Because of this, during the  pre-processing step, stop-and-frisk incidents that led to arrests were more favored. 

#### Data Bias:
> ##### **Population Bias**:
There are significant differences in demographics in the dataset. Of the 13,459 stops recorded, 59% were Black and 29% were Hispanic or Latinx. Less than 10% were White. 
> ##### **Behaviorial Bias**:
Not all of the officers are reporting every one of their incidents and when they do, the quality and descriptions of the report will be different for each person. For example, while on the scene, one officer describes the suspect as "NORMAL" and another officer describes the suspect as "APPARENTLY NORMAL". The two officers described the same person, but their interpretations are slightly different. 

### Citations

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

