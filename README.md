# **README File for the Data Analysis Project Repository created for UCSC's CSE 146 (Ethics and Algorithms) class.**

This project performs general data analysis on a given dataset. The database records all of the recorded NYPD's stop incidences and the details pertaining to each stop. Some of the items recorded include the description of the suspected crime, whether the officer was wearing a uniform, the suspect's demeanor, and the location of the stop and frisks. These incidents are recorded through an app that are then uploaded to the database. Data source: [NYPD's 2019 Stop, Question, and Frisk Database](https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page) 

# Content
Main files in this project:
* [DataAnalysisProject (RATM).ipynb](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/DataAnalysisProject%20(RATM).ipynb): loads the dataset and performs all the analysis mentioned below.
* [NYPD2019.csv](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/NYPD2019.csv):  We do not own the dataset. The full .xls file is located at this url: https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page. However, in order to assist setup of the .ipynb project, a preprocessed version has been added to the repository. 
* [README.md](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/README.md):  This is a brief overview of the project repository. So references to files, techniques, concepts, and sources.

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
  * [Errors comparison and inferences](#errors-comparison-and-inferences)
- [Section 4: Conclusion](#section-4-Conclusion)
  * [Other possible definitions](#other-possible-definitions)
  * [Things to note](#things-to-note)
  * [Citations](#citations)
    
<!-- toc --> 

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

