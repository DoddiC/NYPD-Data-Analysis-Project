# **README for the Data Analysis Project Repository created for UCSC's CSE 146 (Ethics and Algorithms) class.**

This project performs general data analysis on a given dataset. The database records all of the recorded NYPD's stop incidences and the details pertaining to each stop. Some of the items recorded include the description of the suspected crime, whether the officer was wearing a uniform, the suspect's demeanor, and the location of the stop and frisks. These incidents are recorded through an app that are then uploaded to the database. Data source: [NYPD's 2019 Stop, Question, and Frisk Database](https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page) 

# Content
Main files in this project:
* [DataAnalysisProject(RATM).ipynb](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/DataAnalysisProject(RATM).ipynb): This file loads the dataset and performs all the analysis mentioned below.
* [NYPD2019.csv](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/NYPD2019.csv):  We do not own the dataset. The full .xls file is located at this url: https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page. However, in order to assist setup of the .ipynb project, a preprocessed version has been added to the repository. 
* [README.md](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/README.md):  This is a brief overview of the project repository. So references to files, techniques, concepts, and sources.

# Analysis techniques
The following techniques have been utilized to perform the analysis:
* Data Pre-processing
* Logistic Regression
* Demographic and Predictive Parity
* Matplotlib

# Outline of the [DataAnalysisProject(RATM).ipynb](https://github.com/DoddiC/DataAnalysisProject-RATM/blob/master/DataAnalysisProject(RATM).ipynb) file:

- Section 1: Onboarding process
  * Data preprocessing
    + Removal of features
    + Feature engineering
    + Dropping extraneous columns
    + Type conversion
  * Reading in the NYPD dataset
- Section 2: Modeling process
  * Training and predicting with the model
- Section 3: Fairness definitions
  * Create a split_on_feature() function
  * Predictive rate parity
  * Demographic parity
  * Errors comparison and inferences
- Section 4: Conclusion
  * Other possible definitions
  * Things to note
  * [Citations](#citations)
    
<!-- toc --> 

### Citations

For our project, we used a handful of reliable sources, as cited below:

# Citations

**MLXtend**:

Sebastian Raschka, MLxtend: Providing machine learning and data science utilities and extensions to Python’s scientific computing stack, The Journal of Open Source Software, 3, 24, apr, 2018,The Open Journal, doi:10.21105/joss.00638, https://joss.theoj.org/papers/by/Sebastian%20Andersen, Accessed 5 Dec 2020

**SKlearn**:

scikit-learn, Scikit-learn: Machine Learning in Python, Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E., Journal of Machine Learning Research, 12, 2825--2830, 2011, https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html, Accessed 4 Dec 2020


**A Strategy to Split the dataset**:

Craig Shallahamer, Random Shuffle Strategy To Split Your Full Dataset, jun, 2020, OraPub} doi:10.21105/joss.00638}, https://blog.orapub.com/20200630/random-shuffle-strategy-to-split-your-full-dataset.html, Accessed 4 Dec 2020

**NYPD dataset**:

NYPD, “Stop, Question and Frisk Data.” Publications, Reports - NYPD, www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page, Accessed 2 Dec 2020

**TLDR dataset**: 

NYPD, “Stop-and-Frisk Data.” New York Civil Liberties Union, 11 Mar. 2020, www.nyclu.org/en/stop-and-frisk-data, Accessed 4 Dec 2020

**How to handle missing values**: 

Kumar, Satyam. “7 Ways to Handle Missing Values in Machine Learning.” Medium, Towards Data Science, 2 Aug. 2020, towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e, Accessed 7 Dec 2020

**Feature Engineering**: 

“Representation: Feature Engineering &nbsp;|&nbsp; Machine Learning Crash Course.” Google, Google, developers.google.com/machine-learning/crash-course/representation/feature-engineering, Accessed 7 Dec 2020

**Create Fit Model**: 

Craig Shallahamer, Random Shuffle Strategy To Split Your Full Dataset, jun, 2020, OraPub} doi:10.21105/joss.00638}, https://blog.orapub.com/20200630/random-shuffle-strategy-to-split-your-full-dataset.html, Accessed 4 Dec 2020

**Fairness Definition**: 

Zhong, Ziyuan. “A Tutorial on Fairness in Machine Learning.” Medium, Towards Data Science, 19 June 2020, towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb, Accessed 8 Dec 2020

**NYPD Corruption and Misconduct data**: 

“New York City Police Department Corruption and Misconduct.” Wikipedia, Wikimedia Foundation, 1 Dec. 2020, en.wikipedia.org/wiki/New_York_City_Police_Department_corruption_and_misconduct, Accessed 8 Dec 2020

**Affects on minorities**:

Chapman, Ben, and Katie Honan. “NYPD's Stop-and-Frisk Practice Still Affects Minorities in New York City.” The Wall Street Journal, Dow Jones &amp; Company, 18 Nov. 2019, www.wsj.com/articles/nypds-stop-and-frisk-practice-still-affects-minorities-in-new-york-city-11574118605, Accessed 8 Dec 2020

**Police self prophecy**: 

“From the President: Predictive Policing: The Modernization of Historical Human Injustice.” NACDL, www.nacdl.org/Article/September-October2017-FromthePresidentPredictivePo, Accessed 9 Dec 2020

**NY census**:

“U.S. Census Bureau QuickFacts: New York.” Census Bureau QuickFacts, www.census.gov/quickfacts/NY, Accessed 9 Dec 2020


# Authors
* **Chidvi Doddi** - [DoddiC](https://github.com/DoddiC)
* **Diana Bui** - [dianadbui](https://github.com/dianadbui)
* **Susanna Morin** - [codeswitch](https://github.com/codeswitch)

