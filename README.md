# DataAnalysisProject-RATM
Data Analysis Project created for UCSC's CSE 146 (Ethics and Algorithms) class.

This project performs general data analysis on a given dataset. The database records all of the recorded NYPD's stop incidences and the details pertaining to each stop. Some of the items recorded include the description of the suspected crime, whether the officer was wearing a uniform, the suspect's demeanor, and the location of the stop and frisks. These incidents are recorded through an app that are then uploaded to the database. Data source: [NYPD's 2019 Stop, Question, and Frisk Database](https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page) 

## Content
Main files in this project:
* [DataAnalysisProject (RATM).ipynb](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/DataAnalysisProject%20(RATM).ipynb): loads the dataset and performs all the analysis mentioned below.
* [NYPD2019.csv](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/NYPD2019.csv):  We do not own the dataset. It is located in the following url: https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page. However, in order to facilitate the download and setup of the project, specifically the .ipynb file, it has been added to the repository.

## Analysis techniques
The following techniques have been utilized to perform the analysis:
* Data Pre-processing
* Logistic Regression
* Demographic and Predictive Parity
* Matplotlib

## Full length breakdown of the DataAnalysisProject (RATM).ipynb file:
Section 1: Onboarding process

A. Reading in the NYPD dataset and dropping null values

B. Data preprocessing

Section 2: Modeling process

A. Training and predicting with the model

Section 3: Fairness definitions

A. Predictive rate parity

B. Demographic parity

C. Errors comparison and any inferences

Section 4: Conclusion

Section 5: Things to note

Section 6: Future modifications and predictions

Section 7: Citations


## Authors
* **Chidvi Doddi** - [DoddiC](https://github.com/DoddiC)
* **Diana Bui** - [dianadbui](https://github.com/dianadbui)
* **Susanna Morin** - [codeswitch](https://github.com/codeswitch)

