# Twitter Topic Analysis

## Overview
This project aims to analyze tweets to identify prevailing topics discussed in relation to different airlines. Using natural language processing (NLP) techniques, the code preprocesses the data, applies topic modeling using Latent Dirichlet Allocation (LDA), and visualizes the results as word clouds. This helps in understanding customer sentiments and prevalent themes across various airline services.

## Application
The application is structured into several modules:
- **Text Processing**: Cleans and preprocesses the tweet data for analysis.
- **Topic Modeling**: Applies LDA to extract significant topics from the cleaned data.
- **Visualization**: Generates and saves word clouds for each identified topic to visualize the frequency of term occurrence.
- **Main Execution**: Orchestrates the workflow from data loading, processing, modeling, to visualization.

## Goal
The goal of this project is to provide a clear understanding of customer opinions and concerns regarding different airlines, enabling businesses to tailor their strategies and improve customer satisfaction.

## Functions
### 1. `preprocess_dataset(text_data)`
- Cleans the tweet text by removing mentions, converting to lowercase, removing stopwords and punctuation, and applying lemmatization.

### 2. `lda_for_entire_dataset(clean_text)`
- Performs LDA on the entire dataset to determine common topics.

### 3. `lda_for_airline(tweets_df, airline)`
- Filters the dataset for tweets related to a specific airline and performs LDA to find relevant topics.

### 4. `create_word_clouds(entity, topics)`
- Generates word clouds for the topics extracted from the LDA analysis.

## Results
Results include visual word clouds that represent the most frequent and significant terms associated with topics in the tweets. These can be used to quickly gauge public opinion and highlight areas of concern or praise. *(Results to be uploaded later.)*

Below are visual word clouds representing the most frequent and significant terms associated with topics in the tweets. These visuals help to quickly gauge public opinion and highlight areas of concern or praise.

### Entire Dataset
![Word Cloud - Entire Dataset](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABiIAAAFJCAYAAAAWtjWsAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOzdd5xcV3338c8t03dm+2q1q1Xv3ZblXrExNtimxNQYTAvOE4MhgeSBhMQE8gBJSICENCCxTSghbhhcMbbl3pssWdKqt9X2On3m3vP8MauVVrur5l1Lsr7vFzLSzL13ztydPXPv+Z3z+1nGGIOIiIiIiIiIiIiIiMgEsI91A0RERERERERERERE5K1LgQgREREREREREREREZkwCkSIiIiIiIiIiIiIiMiEUSBCREREREREREREREQmjAIRIiIiIiIiIiIiIiIyYRSIEBERERERERERERGRCaNAhIiIiIiIiIiIiIiITBgFIkREREREREREREREZMIoECEiIiIiIiIiIiIiIhNGgQgREREROa5Mnz4dy7IO+efmm29+U9qz9/WORzfffPOI8xIMBqmpqWHhwoV85CMf4Yc//CH9/f3HuqkiIiIiInISc491A0RERERERnPOOecwe/bsMZ8/2HOH68ILL+TRRx/lkUce4cILL3zDxztWYrEYV199NQC+79PX18eWLVv45S9/yS9+8Qv+5E/+hG9+85t87nOfG7egysc//nFuueUWbrrpJj7+8Y+PyzHfTHvPgzHmGLdEREREROStT4EIERERETkuffrTnz4uBrjXrVt3rJtwSDU1NaOuENmzZw9/93d/x/e//30+//nPs2vXLv7u7/7uzW+giIiIiIic1JSaSURERETkIObPn8/8+fOPdTOOyuTJk/nud7/LD37wAwD+/u//nscff/wYt0pERERERE42CkSIiIiIyAlv/zoOt99+O+eeey6JRIJYLMY555zDvffeO2z7VatWYVkWjz76KAAXXXTRmPUnxqoRsbeWxbZt27jrrrt429veRlVVFZZlsWrVqqHtenp6uPHGG1m+fDnxeJxoNMqSJUv4m7/5G9Lp9PifjFH80R/9EStXrgQYsSKiUCjw05/+lN///d9n/vz5JBIJIpEI8+bN44YbbqClpWXY9tu2bcOyLG655RYAPvGJTww7d1/72teGtn3uuef4sz/7M04//XTq6+sJBoNMmjSJK6+8kt/97ndjtvfWW2/lkksuobq6mkAgQHV1NQsXLuQP/uAPWL169aj73HbbbVx22WXU1tYSDAZpbGzkmmuu4fXXXx+23de+9rVhP88Da2xs27btkOdTRERERESOjFIziYiIiMhbxo033sg3vvENzj77bN75zneyfv16nnrqKa644gpuv/123vve9wJQX1/Ptddey/33309bWxvveMc7qK+vHzrOkdSf+Id/+Ad+8IMfcNppp3HZZZfR0tKC4zgAvP7661x22WXs3LmTyZMnc+655xIIBHjuuef4y7/8S26//XZWrVpFeXn5+J6IUVxzzTU8//zzrFq1imKxiOuWbgXa2tr46Ec/Snl5OQsWLGDp0qWkUileeeUV/vmf/5n/+Z//4amnnho6J2VlZVx77bU88cQTbN68eUQtj+XLlw/9/c///M955JFHWLRoEStWrCAWi7F582buvvtu7r77br73ve/x+c9/flg7v/71r3PjjTfiui5nn302jY2N9PX1sWPHDv7zP/+TRYsWsXTp0qHti8Uiv//7v8///u//EgqFWLFiBY2NjTQ3N/Ozn/2MO+64gzvuuIPLLrtsqH3XXnvtUCDl2muvHfb6ZWVl43fSRURERESkxIiIiIiIHEemTZtmAHPTTTcd9j6AAUxFRYV55plnhj134403GsDMnTt3xH4XXHCBAcwjjzxyyGOP1U7Hccxdd9014vl0Om1mzZplAPPVr37V5HK5oedSqZT58Ic/bADziU984rDf54FuuukmA5hp06Ydctsnnnhi6L1s2rRp6PH+/n5z1113DWufMcbk83nzla98xQDmne9854jjXXvttYf8Od17772mpaVlxONPPfWUSSQSJhAImF27dg09ns1mTSQSMWVlZWb9+vUj9tu2bZtZt27dsMf+/M//3ADmjDPOMFu2bBn23K233mocxzGVlZWmp6dn2HNj/VxFRERERGT8KTWTiIiIiByXDkz5c+Cf3t7eEft8/etf54wzzhj22Fe+8hXKy8tpbm5m586d497Oa6+9lquuumrE47fccgubN2/miiuu4Bvf+AbBYHDouWg0yg9/+EPq6ur47//+b3p6esa9XQeqqakZ+ntXV9fQ3+PxOFddddWw9gEEAgG++c1v0tDQwP3338/AwMARv+bll1/O5MmTRzx+1llncf3111MoFLjrrruGHu/v7yeTyTBz5kzmzZs3Yr9p06YNq9fR3d3Nd7/7XcLhMLfffjszZswYtv3VV1/NddddR09PDz/96U+PuP0iIiIiIjI+lJpJRERERI5LB6b8OdCBA+cAV1555YjHQqEQM2fO5OWXX2b37t00NTWNazuvvvrqUR+/5557APjgBz846vNlZWWcdtpp3HvvvTz//PNceuml49quA/m+P/T30WpevPrqqzz00ENs3bqVVCo1tH2xWMT3fTZt2sQpp5xyxK/b1dXFPffcw5o1a+jp6aFQKACwceNGADZs2DC0bW1tLdOnT2f16tV88Ytf5FOf+hQLFy4c89iPPPIImUyGiy++mMbGxlG3ufDCC/nXf/1XnnrqKT772c8ecftFREREROSNUyBCRERERI5Ln/70p/n4xz9+RPtMnTp11McTiQQA2Wz2jTZrhOnTp4/6+JYtWwD46Ec/ykc/+tGDHqOjo2O8mzVCZ2fn0N+rqqqG/p5KpfjoRz/KnXfeedD9+/v7j/g1f/SjH/HHf/zHpFKpwz7uT37yE66++mr+8R//kX/8x3+kqqqKM844g7e//e189KMfHbayY+85fuihh0YNruzvzTjHIiIiIiIyOgUiREREROQtw7bf/MyjkUhk1Mf3rii47LLLmDRp0kGPMW3atHFv14FeeukloJSKaf/gyVe+8hXuvPNO5s+fz7e//W1WrlxJTU3N0IqTs88+m6effhpjzBG93osvvsh1112H4zj87d/+LVdeeSVTp04lGo1iWRY//OEPue6660Yc97zzzmPbtm3cc889PProozz11FM88MAD3Hfffdx4443ceeedXHzxxcC+czx79mzOOeecg7Zn/5ROIiIiIiLy5lIgQkRERERkAjQ1NbF+/Xo+9alPjZm+6c30s5/9DIC3ve1tOI4z9Pj//u//AvDLX/6SpUuXjthvbwqlI3XrrbdijOFzn/scf/Znf3ZEx41EIlx99dVD562jo4OvfvWr/PCHP+STn/wk27dvBxhKszVv3jxuvvnmo2qniIiIiIhMPBWrFhEREZGT1t5Z/8VicdyPffnllwP7BvqPpX/913/l+eefBxgRFOju7gZGX5XxwAMPDEvptL9DnbuDHTebzXL77bcfZutLtSP+7u/+DoAdO3YMFfe++OKLCQaDrFq1ivb29sM+HpSKcR+s/SIiIiIiMn4UiBARERGRk9aUKVMAWLt27bgf+zOf+QzTpk3j1ltv5f/+3//LwMDAiG1aW1v50Y9+NO6vvf/x/+RP/mSoSPNXvvIVzj777GHbLFiwAIB//ud/Hvb4hg0b+MM//MMxj32oc7f3uLfccsuw957NZvmjP/ojtm7dOmKf7du38+Mf/3jUehS/+c1vAKisrByq+TFp0iQ+97nPkUqluPLKK3nttddG7JfL5fj1r3/N+vXrj6j9IiIiIiIyfixzpMleRUREREQm0PTp09m+fTvnnHMOs2fPHnO7Sy+9lI985CMAQ4WKx7q0vfDCC3n00Ud55JFHuPDCC4cev+eee7jiiisIBoNceuml1NXVYVkWn/zkJ4cG7Mc69t52bt26dcyC1WvXruWKK65g27ZtVFRUsHTpUqZMmUI6naa5uZl169ZRV1dHa2vrYZ2bA91888184hOfIBaLDaUx8n2fgY)

## Requirements
This project requires Python 3.x and the following packages:
- pandas
- nltk
- scikit-learn
- matplotlib
- wordcloud

You can install these packages using pip:
```bash
pip install pandas nltk scikit-learn matplotlib wordcloud
