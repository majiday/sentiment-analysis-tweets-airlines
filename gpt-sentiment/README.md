# Tweet Sentiment Analysis

This project analyzes the sentiments of tweets using OpenAI's language model. The analysis involves loading a dataset of tweets, predicting the sentiment using an AI model, and comparing these predictions to the original sentiments. The results are then reported in terms of precision, recall, and overall match rate.

## Project Structure

- `data_processing.py`: Handles the loading and sampling of tweet data.
- `sentiment_analysis.py`: Contains functions for initializing the OpenAI client and performing sentiment analysis.
- `results_reporting.py`: Manages saving the analysis results to a CSV file and generating a classification report.
- `main.py`: The main script that orchestrates the flow of the program.

## Setup

### Requirements

- Python 3.x
- pandas
- openai
- scikit-learn


### Configuration

- Place your OpenAI API key in a file named `api_key.txt` in the project root directory.
- Ensure your dataset (`Tweets.csv`) is in the project root directory or update the path in `data_processing.py` accordingly.

## Results
### performance og gpt-3.5 compared to human lables
![accuracy of model](results/entire_data.jpg)

## Usage

Run the main script to perform sentiment analysis.
