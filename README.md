# Flextensions: Analysis Across Course and Assignment Types (Data Science Honors Thesis)

This repository includes most of what is needed to replicate the exploration. The original data includes personally identifiable data which cannot be shared nor uploaded to this repository. Instead, the `data` folder includes text files describing the schema of each dataset that was used.

## code

This folder contains the code used to conduct the exploration. Code outputs are not included in order to maintain privacy of the data. The code folder includes two files:

* `combine_clean.py` includes the code for combining the multiple CSV files containing raw logs, extracting the relevant question-answer pairs, mapping responses to course names and assignment deadlines, and preparing the data for analysis.
* `analysis.py` includes the code for conducting the hypothesis tests and creating visualizations. The data comes from reading in `cleaned_data.csv`.

## data

Instead of the datasets used (which cannot be shared), this folder includes one text file for each dataset that was used in the exploration, describing the schema. This folder consists of three files and one folder:

* `cleaned_data.txt` which describes the schema of `cleaned_data.csv`, the dataset including the extension requests collected
* `classes.txt` which describes `classes.csv`, used to map form responses to concrete class names
* `deadlines.txt` which describes `deadlines.csv`, used to map form responses to the appropriate assignment and deadlines

* `raw_data` is a folder that contains the CSV files of the raw Google Cloud project logs. `example_log.txt` is a placeholder for the 14 CSV files that were in the original `raw_data` folder.

## figures

This folder includes all the figures created from running the code in `analysis.py`. 
