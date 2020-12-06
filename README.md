# DSND-Capstone-TMDB5000
This project was completed as part of project requirements for the Udacity [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) and submitted as the Capstone project.

## Required libraries
- nltk 
- numpy
- pandas
- seaborn
- json
- scikit-learn 
- plotly
- collections
- matplotlib
- warnings
- requests
- bs4
- re
- urllib
- tqdm
- lightgbm
- skopt
- pickle 


## Motivation
Since the stakes in the film business are often very high, studio bosses, movie producers, and many others have been looking for a surefire way to turn their investment into big box office returns since the earliest days of the industry.
A movie's box office and critical success, like that of any creative endeavor, is a combination of a large number of tangible and intangible elements. The onscreen chemistry of the main characters, the director's vision and execution, production values / special effects, and the movie's release timing are just a few of the components that can turn low- or no-budget indy flicks into huge blockbusters, or doom an expensively produced studio vehicle or even a 'surefire' sequel to crater at the box office.

Can past movie release data provide us with a  'formula' to increase our chance of commercial success? Are there 'winning combinations' of actors, topics, directors, and genres? This project explores a database of 4,800 movies to first create a visual and quantitative intuition for potential patterns present in movie data and then attempts to create a machine-learning model to predict a movie's revenue.
This project uses a data set generously provided by Kaggle and TMDB [here](https://medium.com/r/?url=https%3A%2F%2Fwww.kaggle.com%2Ftmdb%2Ftmdb-movie-metadata)

## Implementation
### Summary 
A LGBMRegressor was trained using a supervised learning pipeline containing normalization steps (one-hot encoding, TFIDFVectorizer, log1p scaling) and a train/fit step. The regressor was then used to predict movie revenue for the holdout test set to evaluate performance and tuned using a Bayesian tuner. 

### Detail
Initially, an ETL pipeline combines data from two distinct files, transforms JSON dictionaries into iterable lists of custom class objects, and creates several polynomial features. 
This is followed by extensive exploratory data analysis and visualization of both categorical and numeric data. 
After initial feature engineering with a custom one-hot encoder, the data is then processed with a machine learning pipeline consisting of a column transformer that contains additional one-hot, TFIDF and log1p preprocessors as well as the LGBMRegressor that is configured with custom hyperparameters. 
Based on the initial output, a Bayesian tuner is configured with a hyperparameter search space and the resulting optimized parameters are used to retrain / refit the LBGMRegressor. The model can then be used to score additional movies or movie ideas to predict revenue. 



## Files
- data - folder containing source data
  - tmdb_5000_movies.csv - Movie metadata
  - tmdb_5000_credits.csv - Crew/cast metadata
- DSND-Capstone-StudioBossInABox.ipynb - Jupyter notebook containing end-to-end code including exploratory data anlysis, visualizations, model pipeline and results.
- df_expanded_movies_plus_IMDB.p - A pickle file containing a dataframe already augmented with IMDB data. Load this dataframe by setting the load_from_file variable to True in the import section of the notebook. This will prevent the (time-consuming) BeautifulSoup script from running again. Set the flag to False if you'd like to run the BeautifulSoup script and create an updated version of this pickle file. 

## Project Screenshots
Exploratory data analysis - Popularity vs. revenue
!['Popularity vs. Revenue'](/readme_imgs/pop_revenue.png)

Exploratory data analysis - Movie production by country over time
!['Movies by country / year'](readme_imgs/movies_by_year.PNG)

Exploratory data analysis - Feature correlation
!['Feature correlation'](/readme_imgs/corr.png)

Exploratory data analysis - Genre-specific keyword WordCloud
!['Western keywords'](/readme_imgs/western.png)

Exploratory data analysis - Genre-specific keyword WordCloud
!['Western keywords'](/readme_imgs/western.png)

Model output - R2 scatterplot
!['Example Output'](/readme_imgs/initial_scatter.png)



## Acknowledgements
- Kaggle and TMDB for providing the labelled training data set [here](https://medium.com/r/?url=https%3A%2F%2Fwww.kaggle.com%2Ftmdb%2Ftmdb-movie-metadata)