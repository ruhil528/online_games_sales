# online_games_sales
 User and expert reviews, genres, platforms (e.g. Xbox or PlayStation), and historical data on game sales are available from open sources.

# Online Games Sales Data Analysis

## Overview

This project is a Streamlit web application that provides an Exploratory Data Analysis (EDA) of an online video games sales dataset. The dataset contains information about video game sales across various regions, including North America, Europe, and Japan. It also includes data on genres, platforms, and ESRB ratings. The main goal of this project is to explore trends and insights that contribute to the success of video games in different markets.

## Features

The Streamlit app includes the following features:

Data Overview: Display of the data sample and statistical summary to understand the structure of the dataset.

Sales by Platform: Bar charts depicting total sales by platform in North America, Europe, and Japan.

Market Share by Platform in Japan: A pie chart illustrating the sales share of different platforms in the Japanese market.

Top Genres by Sales: Bar charts displaying the top five genres by sales for North America, Europe, and Japan.

Effect of ESRB Ratings on Sales: Analysis of sales distribution by ESRB ratings in each region using bar charts.

## How to Run the App

To run the Streamlit app locally, follow these steps:

Clone the repository and navigate to the project directory:

git clone <repository-url>
cd online_games_sales

Create a virtual environment and activate it:

conda create -n games_sales python=3.9
conda activate games_sales

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

## Requirements

Python 3.9

Streamlit

Pandas

NumPy

Plotly

Matplotlib

Seaborn

All required libraries are listed in the requirements.txt file.

## Dataset

The dataset used in this project is named games.csv. It contains information on various video games, including sales figures in different regions, genre, platform, year of release, and ESRB rating.

Folder Structure

app.py: Main script to run the Streamlit application.

games.csv: Dataset used for the analysis.

requirements.txt: List of dependencies required to run the app.

## Insights

This analysis helps us identify:

The platforms with the highest sales in different regions.

The genres that are most popular in each region.

The influence of ESRB ratings on sales in various markets.

These insights are helpful for game developers, marketers, and industry analysts who want to understand the factors behind successful games.

## License

This project is licensed under the MIT License.

## Author

Developed by Ruhil Dongol.
