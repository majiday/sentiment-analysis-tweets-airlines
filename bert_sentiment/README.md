# Sentiment Analysis of Tweets

## Project Overview
This project implements a sentiment analysis tool to classify tweets into three categories: positive, negative, and neutral. The application uses a BERT-based model from the Hugging Face `transformers` library to analyze the sentiment of sampled tweets from an airline dataset. The goal is to assess the performance of the model by comparing its predictions against human-labeled sentiments.

## Application and Goal
The primary application of this project is to demonstrate the use of natural language processing (NLP) techniques in real-world data, specifically in understanding public sentiment from social media. The goal is to accurately map the sentiments of tweets to help in sentiment tracking over time, which can be valuable for businesses and researchers.

## Project Structure
- **`main.py`**: The entry point of the application that orchestrates the loading of data, sentiment prediction, and accuracy calculation.
- **`data_processing.py`**: Contains functions related to data loading and sampling.
- **`sentiment_analysis.py`**: Includes functions to handle the initialization of the sentiment analysis model, prediction of sentiments, and accuracy evaluation.
- **`requirements.txt`**: Lists all necessary Python packages required to run the project.

## Requirements
To run this project, you need Python 3.x and the following packages:
- pandas
- transformers
- torch

You can install all required packages using the command:
```bash
pip install -r requirements.txt
```

## How to Work with This Code
1. **Clone the Repository:**
   Clone this repository to your local machine using `git clone`, or download the ZIP file and extract it.

2. **Install Dependencies:**
   Navigate to the project directory in your terminal and run:

```bash
pip install -r requirements.txt
```
