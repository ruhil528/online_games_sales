#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 5</b>
# 
# 
# Thank you very much for such a good job! I've left a couple of new comments with digit 5. However, there are no issues that need to be fixed, so I can accept the project. Congratulations and good luck! 😊 
#     
#  
# </div>

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 4</b>
# 
# 
# You almost finished it, well done! My new comments have digit 4. Would you take a look?
#  
# </div>

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 3</b>
# 
# 
# Thank you for updating the project, it looks great! However, there are still several issues that need your attention. For instance, we should update the relevant time interval a little bit. I tried to leave detailed comments with digit 3, please take a look :) I've also left there some recommendations for improving the project.
#  
# </div>

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 2</b>
# 
# 
# Thank you for the updates! I've left a few comments titled as **Reviewer's comment 2**. Would you take a look?
#  
# </div>

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment</b>
#     
# Hi, Ruhil! I am a reviewer on this project.
# 
# Before we start, I want to pay your attention to the color marking:
#     
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# Great solutions and ideas that can and should be used in the future are in green comments.   
# </div>    
#     
#     
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
# 
# Yellow color indicates what should be optimized. This is not necessary, but it will be great if you make changes to this project.
# </div>      
#     
#     
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
# 
# Issues that need to be corrected to get right results are indicated in red comments. Note that the project cannot be accepted until these issues are resolved.
# </div>    
# 
# <hr>
#     
# **Please, use some color other than those listed to highlight answers to my comments.**
# I would also ask you **not to change, move or delete my comments** so that it would be easier for me to navigate during the next review.
#     
# In addition, my comments are defined as headings. 
# They can mess up the content; however, they are convenient, since you can immediately go to them. I will remove the headings from my comments in the next review. 
#    
#     
#     
# <hr>
#     
# <font color='dodgerblue'>**A few words about the project:**</font> you did a great job here, thank you so much for submitting the project! The project looks very good, but it is not finished. Would you try to complete the tasks? Please don't forget to write intermediate conclusions. 
#     
#     
# I will wait for the project for a second review :)
#     
#     
# 
# <hr>
#     
# Please feel free to schedule a 1:1 with our tutors or TAs, join daily coworking sessions, or ask questions in the sprint channels on Discord if you need assistance. 
# 
# </div>

# # Exploratory Data Analysis
# - Online video games sales data divided by regions, NA (North America), EU (Europe), and JP (Japan).
# - Data contains expert reviews, genres, platforms, and historical data on game sales
# - Data is from 2016. plan to campaingn for 2017. Forecasting 2017 based on 2016 data
# 

# ## Task
# - Identify patterns to determine whether a game succeds or not
# - Spot potential bid winners and plan advertising campaigns

# ### Rating glossary
# ESRB (Entertainment Software Rating Board) evaluates a game's content an assigns an age rating. E.g. T (Teen) or M (Mature).
# 
# - E: Everyone
# - T: Teen
# - M: Mature 17+
# - E10+: Everyone 10+
# - EC: Early Childhood (Retired ratings)
# - K-A: ???
# - RP: Rating Pending
# - AO: Adults only 18+

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# There's an introduction, which is good. It is important to write an introductory part, because it gives an idea about the content of the project.
#     
# </div>

# In[71]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[72]:


# Load data
data = pd.read_csv('/datasets/games.csv')


# In[73]:


data.shape


# In[74]:


# Replace column name to lowercase
data.columns = data.columns.str.lower()
data.columns


# #### Why were the datatypes changed?
# - The column of 'year_of_release' need to be changed from float64 to int32. This will save memory space and operational efficiency.
# - The 'user_score' needs to be changed from object to float64 to perform arithmetic operations.

# In[75]:


# Convert datatypes
# The 'year_of_release' should be in int as it does not require float, which will take more memory
data['year_of_release'] = data['year_of_release'].astype('Int32')

# The 'user_score' needs to be in float as it would require some statistical analysis (e.g. taking average of scores, etc. )
# The 'tbd' data can be converted to np.nan to allow for float data processing
data['user_score'] = data['user_score'].replace('tbd', np.nan)
data['user_score'] = data['user_score'].astype('float64')  # This can later be filled with mean/median per genre


# #### Check for duplicates
# 
# - Two rows are deleted and the values are assigned to the other semi-duplicated values. 
# 

# In[76]:


# There are no duplicates that matches the entire row data.
# However, more specific duplicate search needs to be performed to clean the data.
data.duplicated().sum()


# In[77]:


# There are two duplicate for the columns specified below.
data[data.duplicated(['name', 'year_of_release', 'platform'])]


# <span style="color: blue;">
# There are two rows with "Madden NFL 13" game in the 'PS3' platform that was released in year 2012.
# 
#     However, the row are not completely duplicated, the sales information are different. 
#     I am going to add the EU sales in the two rows. 
# </span>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Makes sense. 
#     
# </div>

# In[78]:


data.query('name == "Madden NFL 13" and platform == "PS3" and year_of_release == 2012')


# In[79]:


# Add the eu_sales for the two 'Madden NFL 13' game
# Delete one of the duplicates.
data.loc[604, 'eu_sales'] = data.loc[604, 'eu_sales'] + data.loc[16230, 'eu_sales']
data = data.drop(index=16230)
data.loc[604, 'eu_sales']


# <span style="color: blue;">
#     Similarly issue with the duplicated row for GEN platform with game released in 1993. The jp_sales are different. We are goind to add it to make it a single data row.
#     </span>
#     

# In[80]:


data.query('platform == "GEN" and year_of_release == 1993')


# In[81]:


# Combine two rows to eliminate duplicates
# Add jp_sales from index=14244 t data index=659
data.loc[659, 'jp_sales'] = data.loc[659, 'jp_sales'] + data.loc[14244, 'jp_sales']
data = data.drop(index=14244)
data.loc[659, 'jp_sales']


# 
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment </h2>
#     
# 
# Yes, it's very important to check for the duplicates. 
# 
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# Try to check for the name-year-platform duplicates as well. 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Great!     
# </div>

# <span style="color:red;">
#       The duplicates for the name-year-platform has been addressed. Found two duplicates. Assigned value to more populated duplicated row.
# </span>

# #### Missing values
# 
# - The 'name' and 'genre' columns have 2 data with missing values. Critic score, User score, and rating are also missing for these 2 data points. It is reasonable to exclude these 2 data points as important information are missing.
# 
# - 269 games are missing year_of_release. Some games have year information. The year information can be extracted and imputed into the year_of_release column. year_of_release information was imputed in this way for 17 data points.

# In[82]:


print(f'The number of not reported values in ESBR ratings: {data["rating"].isna().sum()}')


# In[83]:


# The 'name' and 'genre' columns has same 2 NaN data
# Both associated to the platform GEN. Additionally, for these two data, genre, critic_score, user_score, and rating are also NaN
# It is better to eliminate these two columns
data[data['name'].isna()]


# In[84]:


# Remove the any rows with NaN value for name
data = data[~data['name'].isna()]
data.shape


# In[85]:


# How many NaN values are there for 'year_of_release' columns?
# preprocessing number of nan values = 269
data[data['year_of_release'].isna()]


# In[86]:


data[data['year_of_release'].isna()][data[data['year_of_release'].isna()]['name'].str.contains(r'\b\d{4}\b')]


# In[87]:


data[data['year_of_release'].isna()]


# <span style="color: blue;">
# Extract year information from 'name' column and add to the missing 'year_of_release' information    
# </span>

# In[88]:


# String in game name to check for assigning 'year_of_release' that is 
# one year earlier than the year in the game name
game_string = ['nfl', 'fifa', 'wwe', 'nascar', 'soccer', 'football', 'nba', 'baseball',]

for idx, row in data[data['year_of_release'].isna()][data[data['year_of_release'].isna()]['name'].str.contains(r'\b\d{4}\b')].iterrows():
    # Extract four-digit years as a Series
    extracted_year = re.search(r'(\b\d{4}\b)', data.loc[idx, 'name'])[0]
    
    # Find common items between the game_string list and game name string
    common_items = set(row['name'].lower().split(' ')).intersection(game_string)
    if common_items:
        data.loc[idx, 'year_of_release'] = int(extracted_year) - 1
    else:
        data.loc[idx, 'year_of_release'] = int(extracted_year)
    


# In[89]:


data[data['year_of_release'].isna()]


# In[90]:


# There are no values for 'platform' columns
data['platform'].isna().sum()


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment </h2>
#     
# 
# Great idea! However, you should be careful with it. For example, FIFA 15 should be released in 2014. It is called 15 because the season ends in 2015.
# </div>

# <span style="color: green;">
#     The above concern in assigning the year from text is addressed. 
#     </span>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Looks great!     
# </div>

# In[91]:


# Drop nan in 'year_of_release'
data = data.dropna(subset=['year_of_release'])


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# - Make sure each chart in the project has a title.  
# 
# 
# 
# - Would you add a conclusion?</div>
# 
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment </b>
#     
# 
# - Axes labels such as `Count` or `Frequency` may seem unclear to a reader. 
#     
# 
# 
# -  Labels should not include the underscore character.</div>

# #### Calculate the total sales (the sum of sales in all regions) for each game and put it in separate column

# In[92]:


# Total_sales is the sum of sales in all regions
# data['total_sales'] = data['na_sales'] + data['eu_sales'] + data['jp_sales'] + data['other_sales']
data['total_sales'] = data[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)
data[['name', 'na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales']]


# #### Intermediary conclusion
# 
# - Initially, there were 11 columns, 'total_sales' was added to the data, which is the sum of sales in all the regions. The data is described below:
#     - name: name of the game (string, e.g., Wii Sports);
#     - platform: platform the game was released (string, e.g., Wii);
#     - year_of_release: year the game was release (string, e.g. 2009);
#     - genre: genre of the game (string, e.g., Sports);
#     - na_sales: game's sales in North America (float);
#     - eu_sales: game's sales in eErope (float);
#     - jp_sales: games' sales in Japan (float);
#     - other_sales: game's sales in other regions (float);
#     - critic_score: critic's ratings for the game out of 100 (float);
#     - user_score: user's ratings for the game out of 10 (float); and,
#     - rating: ESBR ratings (string).
#     
# 
# - Except for two values, no other values were duplicated. One was deleted as majority for information was missing and the other was considered split data, thus, the information of the duplicated rows were merged appropriately. 
# 
# - Data types were converted into appropriate datatypes. 
# 
# - The missing values for year_of_release of 17 games were extracted from game's name (e.g. Madden NFL 2004) and added to the year_of_release column with appropriate assumptions. 
# 
# - The critic's rating, user rating, and ESBR ratings have significant number of missing values. Possible reasons could be:
#     - niche or small-scales games may not attract a large audience or attention from critics;
#     - platform specific, for example, games release in smaller platform may not be widely reviewed;
#     - games may only be released in beta stages;
#     - games may have be of poor quality or incomplete;
#     - games may have limited release; and
#     - big platforms may not have user review features, e.g. Nintendo eShop does not offer user reviews.
# 
# 
# **Hypothesis:**: The information about the game platform, genre, year of release, critic ratings, user ratings, and ESBR ratings are good indicators whree a game is going to be popular, successful, and profitable. 
#     

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# You can also use **sum** with **axis=1** argument:
# </div>
# 
# 
# ```python
# 
# 
# df['total_sales'] = df[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)
# ```
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#  
# - In your opinion, how these missing values could occur? From the task: `Why do you think the values are missing? Give possible reasons.`
#     
#     
# - Please, add an intermediate conclusion about this introductory part. What have been done, what hypotheses about the data we have and what we are going to do next. 
# 
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
# The comment above is still relevant.     
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
#     
# Excellent! 
#     
# </div>

# ## Analysis

# ### How many games were released in different years? Is the data in every period significant?
# - The bar plot shows the number of games released in differnt years.
# - More than 200 games have release every year since 1995.

# In[93]:


#data['year_of_release'] = pd.to_datetime(data['year_of_release'], format='%Y').dt.year


# In[94]:


# Set plot style 
sns.set(style="whitegrid")

# Plot the data
data['year_of_release'].value_counts().sort_index().plot(kind='bar', color='skyblue', 
                                                         edgecolor='black', figsize=(10, 6))

plt.xlabel("Game Release Year", fontsize=14, labelpad=10)
plt.ylabel('Number of Games', fontsize=14, labelpad=10)
plt.title("Number of Games Released by Year", fontsize=16, pad=15)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, )#ha='right')
plt.show()


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Good!     
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
# Bar chart would be a better choice, since it may help us correctly identify the relevant time interval.    
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
#     
# Very good! Now we can easily see the years on the X axis.     
# </div>

# ### Which platform has the greatest total sales?
#  - PS2 has the greatest total sales over all regions.
#  - PS2, X360, PS3, Wii, and DS are platforms with more that 600 Million sales are selected 
#  - 

# In[95]:


# Which platform has the greatest total sales?
# fig, ax = plt.subplots(figsize)
ax = data.groupby('platform')['total_sales'].sum().sort_values(ascending=False).plot(kind='bar', figsize=(10,6))
# ax.axis('off')
# plt.grid(axis='y') 
plt.xticks(rotation=45, )#ha='right')
plt.xlabel('Platform')
plt.ylabel('Sales (Mil)')
plt.title('Total Sales by Platform')
plt.show()


# The total sales by platform shows that the platform with the most global sales are  These all have more that 600 Million in total sales. 
# - more than 600 Million sales: PS2, X360, PS2, Wii, DS, and PS.
# - between 600 - 200 million: PS4, FBA, PSP, 3DS, PC GB, XB, NES, N64, and SNES
# - 10-200 million: GC, XOne, 2600, WiiU, PSV, SAT, GEN, DC
# - Less than 10 million: SCD, NG, WS, TG16, 3DO, GG, PCFX

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
# What can be inferred from this chart?     
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
#     
# Good.     
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Correct.     
# </div>

# #### Find platforms with the greatest total sales and build a distribution based data for each year

# In[96]:


# Distribution of PS2 games released on differnt years.

# Select platforms with at least 600 Mil sales from previous bar plot
selected_platforms = ['PS2', 'X360', 'PS3', 'Wii', 'DS', 'PS']
fig = px.histogram(data.query('platform == @selected_platforms'), x='year_of_release', y='total_sales', 
                   color='platform',barmode='overlay',
                   title="Total Sales of Games Released in Years by Platforms with more than 600 Million Sales"
                  )

fig.update_layout(xaxis_title='Year', yaxis_title='Total Sales (Mil)')

fig.show()


# The platforms with greatest total sales are PS2, X360, PS3, Wii, DS, and PS. The DS is interesting as DS was released back in 1985 and revived in 2004 and did extremely well.

# #### Show total sales by platforms 
# - Categorize the data based-on the most recent games released before 2000, between 2000-2010, and after 2010.
# - This will help create a clear visualization of the total sales distribution for differnt platforms that are no longer operational. 

# In[97]:


# The latest year games were released per platform
platform_lt_2000 = [] # platforms with latest games released before 2000
platform_lt_2010 = [] # platforms with latest games released before 2010
platform_mt_2010 = [] # platforms with latest games released after 2010
for idx, val in data.groupby('platform')['year_of_release'].max().items():
    if val < 2000:
        platform_lt_2000.append(idx)
    elif val < 2010:
        platform_lt_2010.append(idx)
    else:
        platform_mt_2010.append(idx)


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
#     
# Keep it simple:
# 
# </div>

# In[98]:


# Reviewer's code 3
list(data.groupby('platform')['year_of_release'].max().to_frame().query('2000 <= year_of_release <2010').index)


# In[99]:


fig = px.histogram(data.query('platform == @platform_lt_2000'), x='year_of_release', y='total_sales', color='platform', 
                   barmode='overlay',
                   labels={'year_of_release': 'Year',
                        },
                   title='Platform with the most recent game released before 2000'
                  )

fig.update_yaxes(title_text='Sales (Mil)')
fig.show()


# The chart shows sales for platforms with the most recent games released before 2000. These platforms are no longer operational. 
#     
# - These platforms include NES, SNES, 2600, GEN, SAT, SCD, NG, TG16, 3DO, GG, and PCFX.
# - 2600 is the first platform back in the 80's.
# - NES and SNES are the popular platforms from early 80's to mid 90's.
# 

# In[100]:


fig = px.histogram(data.query('platform == @platform_lt_2010'), x='year_of_release', y='total_sales', color='platform', 
                   barmode='overlay',
                   labels={'year_of_release': 'Year',
                        },
                   title='Platform with latest games released before 2010'
                  )

fig.update_yaxes(title_text='Sales (Mil)')
fig.show()


# The chart shows sales for platforms with the most recent game released before 2010. These platforms are not operational.
# - The most popular platform is play station (PS). 
# - There are alot of nintendo platforms, e.g., GB, GBA, GC, and N64
# - The WS platform seems to have negligible sales.

# In[101]:


fig = px.histogram(data.query('platform == @platform_mt_2010'), x='year_of_release', y='total_sales', color='platform', 
                   barmode='overlay',
                   labels={'year_of_release': 'Year',
                        },
                   title='Platform with the most recent games released after 2010'
                  )

fig.update_yaxes(title_text='Sales (Mil)')
fig.show()


# These platform are the most recent ones, with operation windown between 2000 - 2015, except for PC and DS platforms. 
# - The PC platform is considerably differnt from other gaming platforms as PC, in general, have usage more than gaming. Therefore, the longevity of PC for gaming has spanned since the first years of gaming in 1985 to 2016. 
# - DS is also the interest platform, as it was one of the first platfoms created back in 1985 and revived in 2004 that made significant traction and became the top 5 platform with nearly 800 Mil sales.
# - In recent years, the main platform are playstation, Xbox, nitento variant. These companies/platforms dominate the gaming markets. 
# 
# 

# In[102]:


fig = px.histogram(data, x='year_of_release', y='total_sales', color='platform', barmode='overlay',
                  labels={'year_of_release': 'Year',
                        },
                   title='Total Sales for all Platform'
                  )

fig.update_yaxes(title_text='Sales (Mil)')
fig.show()


# The chart puts sales information of all the platforms in one chart. 

# #### How long does it take for new platforms to appera and old ones to fade?
# - average lifetime to platform is 7.62 years.
# - New platforms appear every 1.06 years.

# In[103]:


# Determine average lifetime of a platfrom
platform_lifetime = data.groupby('platform')['year_of_release'].agg(['min', 'max']).sort_values('max')
platform_lifetime['lifetime'] = platform_lifetime['max'] - platform_lifetime['min']
print(f"The average lifetime of platform is {platform_lifetime['lifetime'].mean()}")


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 </h2>
#     
# 
# Correct.
# </div>

# In[104]:


# Determine average years for new platform to appear
platform_lifetime = platform_lifetime.sort_values('min')
platform_lifetime['years_diff'] = platform_lifetime['min'].diff()
platform_lifetime['years_diff'] = platform_lifetime['years_diff'].fillna(0).astype(int)
print(f'The average lifetime of platform is {platform_lifetime["years_diff"].mean()}')


# #### What period should you take the data for?
# - The data period I have selected from 2013 to 2016. 
# - Data after 2013 are more reliable to work with. 

# In[105]:


# Filter data by year_of_release >= 2013
# Market stabilized after 2013
data = data[data['year_of_release']>=2013]
data.shape


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# - Please add a conclusion to address the question from the task. 
# 
# 
# 
# - Please don't forget about titles here and further. </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
# Please make sure each cell works fine.     
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 3</b>
#     
# Is it a good idea to consider more than 20 years? Market is dynamic, everything is changing. Some popular genre may not remain popular 10 years later, not to mention 20 years. And vice versa. During this time, several platform generations will change :) The industry is evolving, the games are getting better, the graphics is getting better and the users are getting more demanding. In such tasks, we need fresh information, fresh estimation. If we forecast for 2017, then who cares what happened 10-20 years ago?  Potentially profitable platforms can be easily selected with charts and pivot tables, but user portraits may change. If you look at the sales bar chart (in the very beginning of this part), you will see in which year the market stabilized after huge volume sales. We can also take the half of a lifetime period.
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 4</h2>
#     
# 
# If we look at the very first sales chart, we will see in which year the market stabilized after huge volume sales. Let's adjust this choice and take years from 2012 or 2013. Years before 2012 (here you include 2011) have huge sales volume, which can influence our distributions. Moreover, new platforms appeared in 2012-2013, which means that these huge sales were generated by older platforms that cannot be relevant for so many years.     
# 
# </div>

# #### Which platforms are leading in sales? Which ones are growing and shrinking? Select several potentially profitable platforms.
# - We are taking data from 2013
# - platforms leading in sales in 2016 are PS4, Xbox One, 3DS, PSV, and PC. 
# - PS3, and X360 platforms are in decline.
# - Others still have reasonable sales, especially PS4 and Xbox One.
# - Potentially profitable platforms are PS4, XOne, 3DS.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 3</b>
#     
# > Potentially profitable platforms are PS, PS2, PS3, Wii, X360, PS4, and DS.
#     
#     
# Are you sure? PS2, for instance, does not even have sales in 2016. After you modify the time interval, please display the chart that will show how sales change over time on each platform in the dataframe with the relevant time interval. 
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 4</b>
#     
# > Potentially profitable platforms are PS, PS2, PS3, Wii, X360, PS4, and DS.
#     
#     
# This is not true. In order to choose promising platforms, please take all platforms within relevant time interval and look at how their sales change over time. Try to display a pivot table or a simple line chart. 
# 
# </div>

# In[106]:


data.groupby('platform')['total_sales'].sum().plot(kind='bar')#, figsize=(12,6), marker='o')
plt.xticks(rotation=45)
plt.xlabel('Platforms')
plt.ylabel('Sales (Mil)')
plt.title('Platform Sales after 2013')
plt.show()


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 5</h2>
#     
#     
# > Potentially profitable platforms are PS4, XOne, 3DS.
#     
#     
# Correct! </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 5</h2>
#     
#     
# However, in the cell above, you took platforms that have most total sales in some year. What if there's a brand new platform that did not have enough time to gain high sales volume? 
#     
#     
# In order to choose promising platforms, we should take all platforms within relevant time interval and look at how their sales change over time within relevant interval. Take a look: 
# 
# </div>

# In[107]:


# Reviewer's code 5

data.pivot_table(index='year_of_release', columns='platform', values='total_sales', aggfunc='sum')


# - PS4, PS3, XOne, X360, and 3DS are the platforms with the most sales from 2013.

# In[108]:


fig = px.histogram(data, x='year_of_release', y='total_sales', color='platform', barmode='overlay',
                   labels={'year_of_release':'Year', 'total_sales':'Sales (Mil)'},
                   title='Games Sales (Mil) by Platform',
                   nbins=4
                  )
fig.update_layout(yaxis_title='Sales (Mil)')
fig.show()


# The chart shows total sales for platforms with games released after 1994.
# - PS4, X360, xOne, , PS3, Wii, XOne, and DS are the platforms with most sales.
# - There are alot of platforms that were created and are now non operational between 1994 and 2005. These platfrom had short lifetime.
# - The major companies dominating the market are Sony with playstation, Xbox, and nintendo. 

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
# If we add a chart, we also should add a conclusion.    
# </div>

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
# - Would you finish this part? 
#     
#  
# - What is the approximate lifespan?    
#     
#     
# - Don't forget to choose the time interval for further analysis.
# 
# 
# 
# - After you do that, please don't forget to identify the most promising platforms. </div>

# #### Box plot
# - global sales of all games broken down by platform
# - Are the differences in sales significant?
# - What about the average sales on various platforms? 

# In[109]:


fig = px.box(data, x='platform', y='total_sales', hover_data=['name', 'platform', 'year_of_release', 'total_sales',], color='platform',
             title="Box Plot of Global Sales by Platform",
             
            )
fig.update_layout(xaxis_title='Platform', yaxis_title='Global Sales (Mil)')
fig.update_yaxes(range=[-0.5, 2], ) # Using ylim for better visualiztion
fig.show() 


# Global sales distribution by platfroms after 2013 are positively skewed.
# - The outliers are games with higher global sales. 
# - Grand theft auto V release on 2013 has the most sales of 20.5 Mil.
# - All interquantile range is below 1 Mil for all platforms.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 3</b>
#     
# 
# Please don't forget to use the correct time interval here and further. 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 4</h2>
#     
# Good.
#     
# 
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Very good.
# 
# </div>
# 
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
#  
# 
# - It is difficult to compare the boxes. Please, try to use the **ylim**, it will scale the graph. 
#     
#     
# - There will be some outliers there, don't drop them. What do you think outliers can tell us? Just write any suggestions you have.
# 
# 
# - Please add a conclusion. 
# 
# 
# - Here and further, don't forget to use the relevant time interval. </div>

# ##### Perform ANOVA analysis test for all platform distribution using scipy.stats.f_oneway test.
# 

# In[110]:


data['platform'].value_counts().plot(kind='bar')

plt.xticks(rotation=45)
plt.xlabel('Platform')
plt.ylabel('Number of Games')
plt.title('Popular Platform after 2013')
plt.show()


# In[111]:


from scipy import stats
# To understand whether the difference in the global sales across various platforms are significant or not. 
# I performed the one-way ANOVA analysis test, which checks whether the means of several groups are equal
# Assumption normally distributed data, equal variance, independent samples.

# Group sales by platform
platform_groups = [group['total_sales'].dropna() for name, group in data.groupby('platform')]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(*platform_groups)

# Print results
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Check significance
if p_value < 0.05:
    print("There is a significant difference in sales across gaming platforms.")
else:
    print("There is no significant difference in sales across gaming platforms.")


# In[112]:


fig, ax = plt.subplots(figsize=(14,7))
ax.scatter(x=data.groupby('platform')['total_sales'].mean().sort_values(ascending=False).index, 
            y=data.groupby('platform')['total_sales'].mean().sort_values(ascending=False).values,
          )
plt.xticks(rotation=45)
plt.xlabel('Platform')
plt.ylabel('Mean of Global Sales')
plt.title('Average Global Sales (Mil) for the Platforms')
plt.show()


# <span style="color: red;">
#     Conclusions on Global sales variations:
#     
#     - The ANOVA test for global sales indicate that there is a significant difference in sales across gaming platforms. 
#     - The mean global sales for PS4, X360, and XOne are higher compared to other platforms.
# </span>

# #### How user and professional reviews affect sales for one popular platfrom. 
# 
# Build scatter plot and calculate the correlation between review and sales. Draw conclusions
# - The scatter plot of Critics and User score vs Global sales shows that there is an lack of strong correlations, which is further verified by low Pearson correlation values.
# - Spearman and Kendall correlation, suitable for ordinal data, are also shown below. Both evaluation methods show low correlations. 
# - High volume global sales PS2 games have user score > 8 and critic score > 8.5.

# In[113]:


def plot_scatter_and_correlate(platform):
    filtered_data = data.query('platform == @platform')
    corr_critic = filtered_data['critic_score'].corr(filtered_data['total_sales'])
    corr_user = filtered_data['user_score'].corr(filtered_data['total_sales'])

    sns.scatterplot(x=filtered_data['critic_score']/10, y=filtered_data['total_sales'], 
                    label=f'Critic Score (Pearson Correlation: {corr_critic:.2f})')
    sns.scatterplot(x=filtered_data['user_score'], y=filtered_data['total_sales'], 
                    label=f'User Score (Pearson Correlation: {corr_user:.2f})')
    plt.xlabel('Score')
    plt.ylabel('Global Scales (Mil)')
    plt.title(f'Global Sales vs Scores for {platform}')
    plt.show()

    # More correlation comparison
    # Pearson correlation assumes normality, while Spearman and Kendall correlations works for data distribution without normal distirbution assumption.
    corrs = ['Pearson', 'Spearman', 'Kendall']
    for corr in corrs:
        corr_critic = filtered_data['critic_score'].corr(filtered_data['total_sales'], method=corr.lower())
        corr_user = filtered_data['user_score'].corr(filtered_data['total_sales'], method=corr.lower())

        print(f"{corr} correlation | Critics score: {corr_critic:.2f}, User score: {corr_user:.2f}")


# In[114]:


# Plot scatter plot for PS2 and calculate Pearson, Spearman, and Kendall correlation
plot_scatter_and_correlate("PS4")


# ##### Conclusion of PS4 platform: 
# - All three correlation metrics show low-correlation between the critics and users scores to the global sales. 
# - Generally, low critics and user scores have low global sales. 
# - However, high critics and scores do not necessarily assure high-global sales.
# - Only few games with relatively higher critics and user scores have high global sales.
# - Critics score is positively correlated to global sales, while user score is not a good indicator of a game's profitability.

# In[115]:


# Repeat the analysis for other popular platform
# Let's compare correlation coefficients for PS4, X360, Wii, PS3, DS, and PS
selected_platforms = ['X360', 'PS3', ]#'Wii', 'DS', 'PS']
for platform in selected_platforms:
    plot_scatter_and_correlate(platform)


# ### Analysis of correlation of Critics and User scores to global sales for:
# **Xbox 360:**
# - All correlation metrics for X360 are low, except for Spearman correlation for critics score, which is reasonably high, 0.63. 
# - Games with higher global sales tend to have higher critics scores. However, some of these games have low-user scores. 
# 
# **PS3:**
# - Similar trends are observed for PS3 as X360.
# - Spearman correlation for critics score is relatively high, 0.59.
# - Games with higher global sales tend to have higher critics scores. However, some of these games have low-user scores. 
# 
# **Overall correlation conclusion**
# - The PS4, X360, and PS3 platforms show generally low Pearson, Spearman, and Kendall correlations. 
# - The critics score is more correlated to the global sales compared to user score. Therefore, critics score can be used to modell 2017 sales than user score.
# - In general, games with low user and critics score tend to have low-global sales.
# - However, not all high-score games have high-global sales, but high-global sales tend to have high scores. There are only few (5 games) that do not adhere to this conclusion, especially for X360 and PS3 platforms.

# 
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
#  
# Please don't forget to use the correct time interval.
# 
# 
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 3</b>
#     
# 
# The comment is relevant.
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 4</h2>
#     
# Well done!     
# 
# </div>

# ##### The data is filtered between 2013-2016.

# 
# 
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
#  
# 
# Please make sure each cell works fine before you send a project for review.
# 
# 
#     
# According to the task, we have to choose 1 platform, analyze how sales depend on critics' and users' scores, display scatter plots and calculate the correlation coefficients. Then we have to choose  2 or more other platforms, repeat the correlation analysis for them and compare the results. Would you add it?  
# 
# 
# The wording in this task is ambiguous, but since we are comparing platforms, it is reasonable to take scores for all games on the platform within a chosen period. In other words, you do not need to check each game in the dataset. 
# 
# All you need here is to take 2 or more other platforms and repeat the analysis. Then compare the results. </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 </h2>
#     
# 
# Good. 
# </div>

# #### General distribution of games by genre
# 
# What can you say about the most profitable genres?
# - The most profitable genres are action, sports, and shooter games. These tend to be active and dynamics games. I believe these have multiplayer games, which makes then attractive to large user base.
# 
# 
# 
# What can you generalize about genres with high and low sales?
# - There is a different of a factor for high- and low-sales genre, i.e, action and strategy, respectively. 
# - Apart from action, sports and shooter have global sales in billion. These games tend to be more dynamic, active, and most likely multiplayer games. 
# - In contrast, the three low-sales genres are puzzle, adventure, and strategy, in decreasing order, tend to be much slower paced, single player, methodological, and more celebral. Although, there may be multiplayer games, especially, in adventure games, a distinct nature of story arc elements. 

# In[116]:


data.groupby('genre')['total_sales'].agg(['sum', 'mean', 'var', 'std']).sort_values('sum', ascending=False)


# In[117]:


fig = px.histogram(data, x='year_of_release', y='total_sales', color='genre', barmode='overlay',
             #marginal='box', 
            )
fig.show()


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 </h2>
#     
# 
# Interesting! 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 </h2>
#     
# 
# However, it may be hard to analyze this chart. 
# </div>

# In[118]:


fig, ax = plt.subplots(figsize=(14,7))
ax.scatter(x=data.groupby('genre')['total_sales'].mean().sort_values(ascending=False).index, 
            y=data.groupby('genre')['total_sales'].mean().sort_values(ascending=False).values,
          )
plt.xticks(rotation=45)
plt.xlabel('Game Genre')
plt.ylabel('Average Global Sales (Mil)')
plt.title('Average Global Sales (Mil) by Game Genre')
plt.show()


# ##### Chart Conclusion:
# - The average global sales show shooter genre has the highest average sales. 
# - The role-playing, racing, sports, action, fighting, and misc. games have global sales between 0.4-07 Mil sales.
# - Simulation, Puzzle, strategy, and adventure games have average global sales < 0.4 Mil. 

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
#     
# Try to display how many sales on average (or you can use median) each genre has. Don't forget about conclusions. 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
#     
# 
# The result may still change after you alter the time interval. </div>

# ## Create a user profile for each region (NA, EU, JP)

# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# 
# The distributions may change after we use the correct time interval, so don't forget to update the conclusions :) 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
#     
# 
# The comment above is still relevant :) </div>

# Yes, the numbers are updated.

# In[119]:


fig = px.histogram(data, x='genre',
            labels={'genre':'Genre'},
                   title='Games Sold by Genre'
                  )
# Update y-axis label using update_yaxes
fig.update_yaxes(title_text='Games Sold')
fig.update_xaxes(tickangle=45) 
fig.show()


# The chart shows number of games by genre.
# - Action and Sports are most popular genres with more than 2000 games 
# - Platform and shooter genre have higher average sales although they have relatively low number of games, especially for Platform platform. 

# ##### Top five platforms for each region

# In[120]:


print('Top five platforms in North America')
print(data.groupby('platform')['na_sales'].sum().sort_values(ascending=False)[:5]/data['na_sales'].sum()*100)
print("*******************************")

print('Top five platforms in Europe')
print(data.groupby('platform')['eu_sales'].sum().sort_values(ascending=False)[:5]/data['eu_sales'].sum()*100)
print("*******************************")

print('Top five platforms in North America')
print(data.groupby('platform')['jp_sales'].sum().sort_values(ascending=False)[:5]/data['jp_sales'].sum()*100)
print("*******************************")


# ##### Top five platforms for each region

# In[121]:


def plot_market_shares(market='na_sales', category='platform'):
    labels = {'na_sales':'North American', 'eu_sales':'European', 'jp_sales':'Japanese', 
              'total_sales':'Global',
              'platform':'Platforms', 'genre':'Genres'
             }
    fig = px.pie(data, values=market, names=category,
                 title=f"{labels[market]} Market Sales Share (%) by {labels[category]}",
                )
    fig.update_traces(textinfo='label+percent', textposition='inside')
    fig.update_layout(showlegend=False, width=800, height=600)
    fig.show()


# In[122]:


plot_market_shares('na_sales', 'platform')
plot_market_shares('eu_sales', 'platform')
plot_market_shares('jp_sales', 'platform')


# ### Conclusions on market shares by platforms in different regions
# - All three - NA, EU, and JP - markets are very different.
# 
# 
# - X360, PS3, PS4, XOne, and 3DS are the most popular platforms in NA, while EU markets are dominated by PS3, PS4, X360, 3DS, and PC. 
# 
# 
# - Meanwhile, Japanese market is dominated by handheld platforms like 3DS, PS3, PSP, PSV, and PS4. Xbox has has 0.36% of Japanese market.  
# 
# 
# - The maximum market share by a particular platform is 41% - 3DS in JP)
# 
# 
# - The console platforms are competitive in NA and EU. 
# 

# ### What is the top five genres for each region (NA, EU, JP)

# In[123]:


print('Top five genres in North America')
print(data.groupby('genre')['na_sales'].sum().sort_values(ascending=False)[:5]/data['na_sales'].sum()*100)
print("*******************************")

print('Top five genres in Europe')
print(data.groupby('genre')['eu_sales'].sum().sort_values(ascending=False)[:5]/data['na_sales'].sum()*100)
print("*******************************")

print('Top five genres in Japan')
print(data.groupby('genre')['jp_sales'].sum().sort_values(ascending=False)[:5]/data['na_sales'].sum()*100)
print("*******************************")


# In[124]:


plot_market_shares('na_sales', 'genre')
plot_market_shares('eu_sales', 'genre')
plot_market_shares('jp_sales', 'genre')


# <span style="color: red;">
#     Conclusion on Genre Market Share
#     
#     - The Japanese gaming market is significantly different from the american and european markets, with role-playing genres capturing approximately a third of the market. 
#     
#     - Meanwhile, the North American and European markets are similar in terms of top four most popular genres, i.e., Action, Shotter, Sports, and Role-Playing genres. Additionally, their market share are almost identical. 
#     
#     - Action genre does capture a significant amount Japanese market as well. 
#     
#     
# </span>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 5 </h2>
# 
#     
# There are some similiarities between Europe the Northern America, while  people in Japan prefer portable platforms and japanese market. We definitely should not recommend them XBox :)  In addition, Japanese also don't like shooters as much as in NA and EU people do.  
#  
# </div>   

# ### Do ESBR ratings affect sales in individual regions?

# In[125]:


data['rating'].value_counts(dropna=False).plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('ESBR Ratings')
plt.ylabel('Number of Games')
plt.show()


# - More than 1600 games are not ESBR rated.
# - M, E, and T have approximately 600 games.
# - Approx. 450 games are E10+ rated.
# 

# In[126]:


data.groupby('rating')[['na_sales', 'eu_sales', 'jp_sales']].sum()


# In[127]:


data.groupby('rating')[['na_sales', 'eu_sales', 'jp_sales']].sum().plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('ESBR Ratings')
plt.ylabel('Sales (Mil)')
plt.show()


# - NA and EU sales are significantly higher than JP sales.
# - For NA, the sales by ESBR ratings in decreasing orders are: M, E, E10+, and T.
# - For EU, the sales by ESBR ratings in decreasing orders are: M, E, T, and E10+.
# - For JP, the sales by ESBR ratings in decreasing orders are: E, T, M, and E10+.

# In[128]:


fig = px.box(data.dropna(), x='rating', y='na_sales', hover_data=['name', 'total_sales',], color='rating',
             labels={'E':"Everyone"},
             title='North America Sales by ESBR Ratings'
            )
fig.update_layout(xaxis_title='ESBR Ratings', yaxis_title='Sales (Mil)', 
                  yaxis=dict(range=[-1, 2]))
fig.show()


# - NA sales are more for M and E rated games.
# - GTA V is the highest grossing games

# In[129]:


fig = px.box(data.dropna(), x='rating', y='eu_sales', hover_data=['name', 'total_sales',], color='rating',
             labels={'E':"Everyone"},
             title='European Sales by ESBR Ratings'
            )
fig.update_layout(xaxis_title='ESBR Ratings', yaxis_title='Sales (Mil)', 
                  yaxis=dict(range=[-1, 2]))
fig.show()


# - EU sales are more for M and E rated games.
# - GTA V is the highest grossing games

# In[130]:


fig = px.box(data.dropna(), x='rating', y='jp_sales', hover_data=['name', 'total_sales',], color='rating',
             labels={'E':"Everyone"},
             title='Japanese Sales by ESBR Ratings'
            )
fig.update_layout(xaxis_title='ESBR Ratings', yaxis_title='Sales (Mil)', 
                  yaxis=dict(range=[-0.2, 0.5]))
fig.show()


# - Sales in JP market is significantly lower compared to EU and NA.
# - Animal Cross - new Leaf is the highest grossing games that is rated E.
# 

# In[131]:


# Desriptive statistics
df_rating = data.groupby('rating')[['eu_sales', 'na_sales', 'jp_sales']].agg(['count', 'sum', 'mean', 'median'])
df_rating


# In[132]:


def compare_esbr_sales_by_region(val='mean'):
    ratings = df_rating.loc[['E', 'E10+', 'M', 'T']].index
    eu_sales_means = df_rating.loc[['E', 'E10+', 'M', 'T'],'eu_sales'][val]
    na_sales_means = df_rating.loc[['E', 'E10+', 'M', 'T'],'na_sales'][val]
    jp_sales_means = df_rating.loc[['E', 'E10+', 'M', 'T'],'jp_sales'][val]

    # Set the figure size
    fig, ax = plt.subplots(figsize=(14, 7))

    # Define bar width and positions
    bar_width = 0.25
    indices = np.arange(len(ratings))  # Index for each platform

    # Plot bars for each region side by side
    ax.bar(indices, eu_sales_means, width=bar_width, color='k', label=f'EU Sales ({val})')
    ax.bar(indices + bar_width, na_sales_means, width=bar_width, color='b', label=f'NA Sales ({val})')
    ax.bar(indices + 2 * bar_width, jp_sales_means, width=bar_width, color='r', label=f'JP Sales ({val})')

    # Set labels and title
    ax.set_xlabel('ESBR Ratings')
    ax.set_ylabel('Average Sales (Mil)')
    ax.set_title('Average Sales by ESBR Ratings and Region')

    # Set the x-ticks to be in the middle of the bars
    ax.set_xticks(indices + bar_width)
    ax.set_xticklabels(ratings)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


# In[133]:


compare_esbr_sales_by_region(val='mean')


# ##### The average regional sales by ESBR rating 
# - Mature ratings have higher average sales for EU and NA
# - Average sales for EU and NA for E, E10+, M and T are significantly higher than for JP
# - NA and EU sales trends by ratings closely resemble each other with NA sales higher than EU sales for E, E10+, M, and T rated games. 

# In[134]:


ratings1 = ['E', 'E10+', 'M', 'T']
fig = px.histogram(data.query('rating == @ratings1'), x='na_sales', color='rating', barmode='overlay',
                   labels={''},
                   marginal='box', 
            )
fig.update_layout(xaxis_title='ESBR Ratings', yaxis_title='Sales (Mil)', 
                  xaxis=dict(range=[0, 5]))
fig.show()


# In[135]:


# Desriptive statistics
df_rating = data.groupby('rating')[['eu_sales', 'na_sales', 'jp_sales']].agg(['count', 'sum', 'mean', 'median'])
df_rating


# ##### Conclusion: Effects of ESBR ratings on Regional Sales
# 
# - There are 617 games with E rating, 456 with E10+ rating, 623 with M rating, and 616 with T rating. The rest of the rating include only small number of games. Therefore, not included in this conclusion.
# 
# - NA and EY sales trends for different ratings, but with higher mean sales. This is also apparent in the box plot analysis. 
# 
# - Again, the JP sales for differnt ratings trends and distribution are different compared to NA and EU sales. 
# 
# - From earlier analysis of market share, it is shown that the JP market are differnt than of NA and EU. THe ESBR ratings for differnt genre and platform can significantly affect the regional sales.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 5 </h2>
# 
#     
# All distributions are correct now. 
#     
# </div>   

# ## Hypothesis testing

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2 </b>
#     
#     
# Don't forget to use the correct time interval here as well. </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 3 </b>
#     
#     
# The comment is still relevant :) 
# 
# 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 4</h2>
#     
# The results may still change.
# </div>

# In[136]:


from scipy import stats


# In[137]:


data['platform'].unique()


# ### Hypothesis testing for Xbox One and PC platforms

# In[138]:


fig = px.histogram(data.query('platform == "XOne" or platform == "PC"'), x='user_score', color='platform', 
                   marginal='box', barmode='overlay',
                   title='Distribution of User Score for PC and Xbox One platform (Vertical line indicates mean)',
                   labels={'user_score':'User Score'}
            )

fig.add_vline(x=data.query('platform == "PC"')['user_score'].mean(), line_dash='dash', line_color='blue', )
              #annotation_text="PC Mean", annotation_position="top left", )
fig.add_vline(x=data.query('platform == "XOne"')['user_score'].mean(), line_dash='dash', line_color='red', )
              #annotation_text="Xbox One Mean", annotation_position="top left")

fig.update_layout(yaxis_title='Number of Users',)
fig.show()

print(f"Mean of Xbox One platform user score {data.groupby('platform')['user_score'].mean()['XOne']:.2f}")
print(f"Mean of PC platform user score {data.groupby('platform')['user_score'].mean()['PC']:.2f}")


# In[139]:


# Perform two_sample t-tests
xone_ratings = data.query('platform == "XOne"')['user_score'].dropna()
pc_ratings = data.query('platform == "PC"')['user_score'].dropna()

# t-test
t_stat, p_value = stats.ttest_ind(xone_ratings, pc_ratings)

print(f"T-statistics: {t_stat}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The average user ratings for Xbox One and PC are equal.")
else:
    print("Failed to reject the null hypothesis, i.e., the average user ratings of Xbox One and PC are equal.")


# **Statistical test conclusion**
# 
# - Null hypothesis: The mean of the user score distributions of Xbox One and PC gaming platforms are equal.
# 
# H0: Mean(Xbox One) = Mean(PC)
# 
# 
# - Alternative hypothesis: The mean of the user score distribution of Xbox One and PC gaming platforms are not equal. 
# 
# H1: Mean(Xbox One) != Mean(PC)
# 
# 
# We failed to reject the null hypothesis as the p-value is larger than the threshold alpha value, 0.66. Therefore, we conclude that there is no significant difference in the average user ratings for Xbox One and PC users.
# 
# 
# - The alpha value is set to 0.05, a widely accepted value, so that there is a 5% chance of incorrectly rejecting the null hypothesis. Choosing 0.05 value brings a reasonable balance between the Type I (false positive) and Type II (false negative) errors. Reducing the alpha value to lower value, e.g. 0.01, will reduce the chance of Type I error. Alpha value of 0.01 are used in studies where type I errors are costly, e.g. medical studies. For this study, we could have use alpha = 0.10 as a exploratory purposes.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 </h2>
# 
#     
# Good. The null hypothesis always includes the equal sign. This is because the test does not understand exactly how we set the problem: we can say, for instance, let's make sure that they are not equal. Or that they are equal. And regardless of the purpose of our study, we always put equality at null hypothesis. Then the test result is interpreted correctly.
# 
# </div>   
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
#     
# 
# The fact that we reject the null hypothesis does not entail that the 2nd hypothesis is true. We just reject it at a particular significance level. I would change the wording in the conclusion. 
#     
# </div>

# ### Hypothesis testing for Xbox One and PC platforms

# In[140]:


fig = px.histogram(data.query('genre == "Action" or genre == "Sports"'), x='user_score', color='genre', 
                   marginal='box', barmode='overlay',
                   title='Distribution of User Score for Action and Sports Genre (Vertical line indicates mean)',
                   labels={'user_score':'User Score'}
            )

fig.add_vline(x=data.query('genre == "Action"')['user_score'].mean(), line_dash='dash', line_color='red', )
              #annotation_text="PC Mean", annotation_position="top left", )
fig.add_vline(x=data.query('genre == "Sports"')['user_score'].mean(), line_dash='dash', line_color='blue', )
              #annotation_text="Xbox One Mean", annotation_position="top left")

fig.update_layout(yaxis_title='Number of Users',)
fig.show()

print(f"Mean of Action genre user score {data.groupby('genre')['user_score'].mean()['Action']:.2f}")
print(f"Mean of Sports genre user score {data.groupby('genre')['user_score'].mean()['Sports']:.2f}")


# In[141]:


# Perform two_sample t-tests
xone_ratings = data.query('genre == "Action"')['user_score'].dropna()
pc_ratings = data.query('genre == "Sports"')['user_score'].dropna()

# t-test
t_stat, p_value = stats.ttest_ind(xone_ratings, pc_ratings)

print(f"T-statistics: {t_stat}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The average user ratings for Action and Sports genre are significantly different.")
else:
    print("Failed to reject the null hypothesis: There is no significant different in average user ratings between Action and Sports genre.")


# **Statistical test conclusion**
# 
# - Null hypothesis: The mean of the user score distributions of Action and Sports genres are equal.
# 
# H0: Mean(Action) = Mean(Sports)
# 
# 
# - Alternative hypothesis: The mean of the user score distribution of Action and Sports genres are not equal. 
# 
# H1: Mean(Action) != Mean(Sports)
# 
# - Since, p-value < alpha, we reject the hypothesis that the average of the user score distribution for Action and Sports genres are equal. 
# 
# - The alpha value is set to 0.05, a widely accepted value, so that there is a 5% chance of incorrectly rejecting the null hypothesis. Choosing 0.05 value brings a reasonable balance between the Type I (false positive) and Type II (false negative) errors. Reducing the alpha value to lower value, e.g. 0.01, will reduce the chance of Type I error. Alpha value of 0.01 are used in studies where type I errors are costly, e.g. medical studies. For this study, we could have use alpha = 0.10 as a exploratory purposes.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2 </b>
#     
#     
# There should be two tests here. Please refer to the task. 
# </div>

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# Please finish the project :)</div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 </h2>
# 
#     
# Good!     
# </div>   

# # Overall Conclusion
# 
# - **Regional Conclusion**
#     - The JP market is considerably different than NA and EU markets.
#     - NA and EU gamers prefer X360, PS2, PS3, and Wii platfroms, but JP gamers prefer handheld Nintendo platforms
#     - JP games overwhelmingly prefer role-playing genre, while NA and EU games prefer action, sports, and shooting games.
# 
# 
# - **Platform Conclusion**
#     - PS2 has the greatest total sales over all regions.
#     - PS2, X360, PS3, Wii, and DS are platforms with more that 600 Million sales are selected 
#     - The PC platform is considerably differnt from other gaming platforms as PC, in general, have usage more than gaming. Therefore, the longevity of PC for gaming has spanned since the first years of gaming in 1985 to 2016. 
#     - DS is also the interest platform, as it was one of the first platfoms created back in 1985 and revived in 2004 that made significant traction and became the top 5 platform with nearly 800 Mil sales.
#     - In recent years, the main platform are playstation, Xbox, nitento variant. These companies/platforms dominate the gaming markets. 
#     - Most promising platforms are PS4, X
# 
#   
# - **Genre Conclusion**
#     - The Japanese gaming market is significantly different from the american and european markets, with role-playing genres capturing approximately a third of the market. 
#     
#     - Meanwhile, the North American and European markets are similar in terms of top three most popular genres, i.e., Action, Sports, Shooting, and Racing. Additionally, their market share are almost identical. 
#     
#     - Action and sports genres do capture a significant amount Japanese market as well. But the shooter genre captures least amount of Japanese market.
#     - Action and Sports are most popular genres with more than 2000 games 
#     - Platform and shooter genre have higher average sales although they have relatively low number of games, especially for Platform platform. 
#     
#     
# - **Ratings Conclusion**
#     - Mature ratings have higher average sales for EU and NA
#     - Average sales for EU and NA for E, E10+, M and T are significantly higher than for JP
#     - NA and EU sales trends by ratings closely resemble each other with NA sales higher than EU sales for all ratings. 
#     
#     
#     
# - **Statistical Testing**
#     - Statistical testing of PC and Xbox One user ratings indicate that we failed to reject the null hypothesis, i.e., the user score distribution for Xbox One and PC games are equal within the statistical significance.  
#     - The statistical testing of Action and Sports genre user scores indicate that we can reject the null hypothesis, i.e., the average of the user score disctribution for Action and Sports genres are equal. 

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3</h2>
# 
#     
# Excellent!     
# </div>   
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 3</b>
# 
#     
# Please don't forget to update it because the results will change.    
# </div>   
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 4</b>
#     
# The results may still change.
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
# 
#     
# The conclusion is written well.    
# </div>    
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2</b>
#     
# It will be great if you add a list of promising platforms. Don't forget to update the final conclusion if needed.
# </div>

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 5 </h2>
# 
#     
# Great job, thank you so much! 
#     
# </div>   

# In[ ]:




