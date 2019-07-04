Sentiment Analysis on Movie Reviews
===================================

Practice Competition on Kaggle  
[Competition Website](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)  
Individual Practice Work

Task
----

Classify the sentiment of sentences from the Rotten Tomatoes dataset

Source
------

### Bash
* [get-glove.sh](get-glove.sh) downloads glove vectors from stanford.edu
* [requirements.sh](requirements.sh) fulfils package requirements

### Python
* [glove.py](glove.py) loads glove vectors (found it on standford.edu and adapted it for my needs)
* [models.py](model.py) contains model definitions
* [visualize.py](visualize.py) plots visualizations
* [utils.py](utils.py) stores utility functions
* [sentiment.py](sentiment.py) is the main program

Requirements
------------

- nltk
- numpy
- matplotlib
- keras
- scikit-learn
- pandas
- graphviz
- pydot

If [*conda*](https://www.anaconda.com/) is installed, run *requirements.sh* script and supply python3 environment name.
```
bash requirements.sh
```

Instructions
------------

1. Clone the repository `git clone https://github.com/pawanmsr/sentiment-analysis-on-movie-reviews.git`
2. Install all the necessary packages mentioned in **Requirements**
3. Download the kaggle dataset from 
[Competition Website](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) and extract it inside the cloned repository
4. Run [sentiment.py](sentiment.py)
```python
python sentiment.py ARGUMENTS
```
Enter `python sentiment.py -h` for help

Methods
-------

*TODO: describe various models used*