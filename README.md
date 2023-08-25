# math_wikipedia_views
## Motivation
Math's being my first area of study, I'd like to know how people are using Wikipedia and how it is structured.

## Description
It is first performed an analysis on the given data. Namely a clustering based on the hyperlinks and the correlation of time series, and also of the stationarity and overall values of the views. After that it is chosen a model (GRU and LSTM) to perform training and prediction over the Mathematics webpage views.

## Build Status
Completed. New ideas to try in the future.

## Files
*DATA*
- wikivital_mathematics.json: Dictionary with views for each page, and weighted graph hyperlink connections

*CODE*
- _GRU.py_: implementation of a GRU in pytorch
- _LSTM.py_: implementation of a LSTM U in pytorch

*PARAMETERS*
- params: this folder contains pre trained parameters 

- _main.ipynb_: Main notebook where the tests are performed.

## Packages
- torch
- time
- pandas
- matplotlib
- pickle
- json
- networkx
- scipy
- community
- statsmodels
- sklearn
- numpy
- random
