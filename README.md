# Quantitative Analysis of News Data for Oil Price Movement Prediction

This repository contains the full end-to-end pipeline for a quantitative research project aimed at identifying and evaluating a predictive signal ("alpha") for WTI Crude Oil price movements based on unstructured financial news data.

In this project I handled everything from sourcing the data to running a realistic walk-forward backtest. My main focus was on exploring the gap between a signal that's statistically significant and one that's actually practical for real-world trading.

-----

## Project Narrative & Key Findings

The core hypothesis was that a predictive edge for short-term oil price drift could be extracted from financial news. The project systematically followed the quantitative research arc:

  * **Signal Discovery**: A weak but statistically significant signal was successfully identified using a combination of NLP embeddings and market-context features.
  * **Robustness Testing**: Walk-forward validation confirmed that this predictive edge was not a statistical fluke but persisted across multiple market regimes over a two-year period.
  * **Viability Analysis**: A rigorous, event-driven backtest incorporating transaction costs ultimately revealed that the signal, while real, was **not strong enough to be profitable as a standalone strategy**.

The most important finding was from the feature importance analysis, which showed that **market context (volatility, time of day) was a more dominant predictor than the news content itself.** This suggests the primary value of the NLP signal is as a diversifying input into a larger, multi-factor quantitative model.

-----

## Project Structure

The pipeline is organized into a series of modular Python scripts, each responsible for a specific stage of the research process.

### **`01_Data_sourcing.py`**

  * Fetches historical hourly price data for WTI Crude Oil (CL=F) from **`yfinance`**.
  * Deploys a custom web scraper using **`requests`** and **`BeautifulSoup`** to build a proprietary dataset of news headlines and precise timestamps from the specialist source **oilprice.com**.

### **`02_Feature_extraction.py`**

  * Loads the raw price and news data.
  * **Label Generation**: Creates the target variable by labeling each news event based on the future price movement over a defined look-forward period and threshold (e.g., did the price move more than 0.7% in the next 4 hours?). This uses an objective, non-subjective labeling methodology.
  * **NLP Feature Extraction**: Utilizes a pre-trained **`ProsusAI/finbert`** model to convert unstructured headlines into 768-dimensional semantic embeddings.
  * **Contextual Feature Engineering**: Creates additional features based on market context, including 24-hour rolling volatility and the hour of the day.
  * Saves the final, feature-rich DataFrame to **`OIL_processed_features.pkl`** for efficient access by downstream scripts.

### **`03_Model_training.py`** (Used for initial exploration)

  * Loads the processed features.
  * Trains a **`LightGBM`** classifier, using **`class_weight='balanced'`** to handle the severe class imbalance inherent in financial data.
  * Performs initial model evaluation on a simple train/test split and generates a feature importance plot.

    <img width="654" height="503" alt="Feature_importance" src="https://github.com/user-attachments/assets/35ea1654-d496-432a-b96e-112d2a4bb8da" />
    <img width="366" height="282" alt="CM" src="https://github.com/user-attachments/assets/bd8c0bcf-b13c-431b-af45-239b8bede549" />



### **`04_walk_forward_validation.py`**

  * Implements a robust walk-forward validation to assess model performance over time.
  * Uses **`sklearn.model_selection.TimeSeriesSplit`** to ensure the model is always trained on past data to predict future events.
  * Aggregates performance metrics (Precision, Recall, F1-Score) across all folds to provide a reliable estimate of the model's predictive power.

### **`05_backtest_with_walk_forward.py`** (The capstone script)

  * Implements the industry-standard validation technique for time-series strategies.
  * Uses **`sklearn.model_selection.TimeSeriesSplit`** to create multiple training/testing folds, ensuring the model is always trained on past data to predict the future.
  * For each fold:
      * Trains a new LightGBM model.
      * Generates predictions on the test set.
      * Runs a **vectorized backtest** on the predictions, correctly calculating returns and applying transaction costs only when a trade occurs to avoid lookahead bias.
  * Aggregates the P\&L from all folds into a single, continuous equity curve.
  * Calculates and reports the final, robust **Annualized Sharpe Ratio** and total cumulative return.

-----

## How to Run the Pipeline

### 1\. Setup

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### 2\. Data Sourcing and Feature Engineering

This step scrapes the data, processes it, and saves the final features.

```bash
python Data_sourcing.py
python Feature_Extraction.py
```

### 3\. Model Training and Evaluation

To run the final, most rigorous analysis that produces the walk-forward P\&L and Sharpe Ratio:

```bash
python backtest_with_walk_forward.py
```

-----

## Key Technologies & Libraries

  * **Data Science & Modeling**: `pandas`, `numpy`, `scikit-learn`, `lightgbm`
  * **NLP**: `transformers` (Hugging Face) for FinBERT
  * **Web Scraping**: `requests`, `beautifulsoup4`
  * **Data Visualization**: `matplotlib`
  * **Utilities**: `tqdm`, `yfinance`
