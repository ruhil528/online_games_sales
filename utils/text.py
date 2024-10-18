"""
Text file with description and summary.
"""

##################
app_description = """This project explores sales data of online video games across various regions, including North America, Europe, and Japan. The data contains information on sales, platform, genres, ESRB ratings, critics and user scores providing insights into what factors contribute to a game’s success, including statistical hypothesis tests and examples of predictive models."""


##################
data_description ='''The dataset for this project contains information on video game sales across different platforms, genres, and regions, along with associated critical and user ratings. The data spans multiple years, covering sales in key regions including North America, Europe, Japan, and others. Each row in the dataset represents a unique video game, with the following key features:

- Name: The title of the video game.
- Platform: The gaming platform for which the game was released (e.g., Wii, NES, PlayStation).
- Year_of_Release: The year the game was released to the market.
- Genre: The category or type of game (e.g., Sports, Racing, Role-Playing).
- NA_sales, EU_sales, JP_sales, Other_sales: Regional sales data in millions of units for North America, Europe, Japan, and other regions.
- Critic_Score: The average score given by critics, typically on a scale of 0-100.
- User_Score: The score provided by users, typically on a scale of 0-10.
- Rating: The game's content rating (e.g., E for Everyone, T for Teen).

This dataset provides a comprehensive overview of the video game market, allowing for detailed analysis of trends in sales by region, platform performance, game genres, and the impact of critic and user ratings on sales performance.

The goal of this analysis is to explore the relationships between these variables and predict global sales based on available features such as platform, genre, and scores. This project will also aim to uncover key insights into what factors contribute to a game's commercial success.
'''

##################
game_by_year_summary = '''This bar chart shows the number of video games released per year from 1980 to 2016. 

The following observations can be made:

- Gradual Increase: There is a slow and steady increase in game releases from 1980 up until the mid-1990s.
- Significant Growth: Starting around 2000, there is a sharp rise in the number of games released each year, peaking around 2008 to 2010.
- Decline After Peak: After the peak in 2010, the number of games released each year gradually decreases.
- Early Period: Between 1980 and 1995, the number of games released is relatively low, suggesting that the gaming industry was still in its early stages or growing more slowly compared to the 2000s.
- Recent Decline: After 2010, the decline in game releases might suggest market saturation, industry shifts, or changes in development practices.

Overall, the plot indicates a major boom in game development in the early 2000s, followed by a subsequent decline post-2010.'''


##################
sales_by_platform = '''
    This bar chart shows the global sales distribution across various gaming platforms. 
    Key observations are:
    - Top Platforms: PS2 leads with the highest global sales, surpassing 1200 million units.
    The X360, PS3, Wii, DS, and PS follow with notable sales, all exceeding 600 million units.
    - Mid-Tier Platforms: Platforms like PSP, PS4, XOne, 3DS, and PC have significant global sales, though not as high as the top tier, ranging between 100-400 million units.  
    - Lower-Tier Platforms: Platforms such as 2600, WiiU, PSV, and others show progressively lower sales, generally ranging between 10-100 million units.
    - Least Sales: The platforms with the lowest sales in the dataset are TG16, GG, and PCFX, each barely showing up on the chart, indicating minimal sales compared to the leading platforms.
    
    Overall Insight:
    The plot indicates that home consoles, particularly from Sony and Microsoft (PS2, X360, PS3), have dominated the gaming market in terms of global sales, while handheld consoles and older generation platforms show significantly lower figures.
    
    NOTE: Newer platforms, for example, PS4, XOne, and WiiU, have active sales not accounted in the data set.'''


##################
sales_distribution = '''This plot, titled "Sales Distribution for Top Platforms," displays the sales distribution over the years for platforms with more than 600 million sales. Here are some key observations:
    
- Platform Sales Peaks:
    - PS2 (green) has the highest sales peak, with significant sales concentrated around the early 2000s, peaking between 2001 and 2005.
    - PS3 (light red) and X360 (red) show later peaks, around 2010, which indicates that these platforms became popular after the PS2's dominance declined.
    - Wii (blue) shows a more compressed peak, with sales concentrated between 2006 and 2010, reflecting its shorter but impactful market presence.
    
- Earlier Platforms:
    - PS1 (light green) shows significant sales in the late 1990s, peaking around 1998-2000, and it gradually declines as the PS2 gains popularity.
    
- Overlapping Timeframes:
    - Several platforms overlap in their peak sales periods. For example, X360 and PS3 have a similar sales distribution period, with both peaking around the same time (2010), indicating fierce competition between them.
    - Wii overlaps with X360 and PS3 but starts to decline slightly earlier, around 2011, whereas the other two decline later.
    
- Decline After Peaks:
    - The distribution shows that most platforms experience a steep decline in sales after their peak periods, which likely coincides with the release of newer platforms or technology shifts.
    
- Sales Duration:
    - PS2 has the longest sales period, reflecting its long-lasting success, whereas platforms like Wii have shorter yet sharp periods of dominance.
    
Conclusion:
    The sales distibution for platforms illustrates the lifecycle of different gaming platforms, highlighting the dominance of the PS2 in the early 2000s, followed by competitive sales from X360 and PS3 in the late 2000s and early 2010s. It also shows how the Wii had a shorter but intense market impact compared to others.'''

##################
def platform_lifetime_summary(avg_lifetime, avg_new_platform_time):
    return  f'''The Platform Lifetime displays the lifespan (in years) of various gaming platforms from their release to the point when they became obsolete or no longer actively supported. The average lifetime of platform is {avg_lifetime:.2f} years and the average time for a new platform to appear is {avg_new_platform_time:.2f} years.
            Key observations include:

Longest Lifetimes:
- PC has the longest platform lifetime, spanning multiple decades, from the early 1980s to beyond 2015. This indicates its continued relevance and adaptability over time.
- NES, DS, GB (Game Boy), PS, PS2, and XB also exhibit long lifetimes, each lasting for over and about 10 years, reflecting their sustained popularity and significant market impact.

Home Consoles:
- PS2, X360, and PS3 had notably long lifetimes, ranging from around 10 to 15 years, indicating their prolonged dominance in the gaming market.
- PS (PlayStation) also had a long life, from around 1995 to the mid-2000s, showcasing its success before the introduction of the PS2.

Short Lifespans:
- Platforms like 3DO, TG16, and PCFX had significantly shorter lifetimes, typically only a few years, suggesting they were either commercial failures or were quickly replaced by newer technology.
- Handheld systems like the WS (WonderSwan) and GG (Game Gear) also had relatively short lifetimes, possibly due to stiff competition from other portable consoles like the Game Boy.

Modern Platforms:
- More recent platforms like the PS4, XOne (Xbox One), WiiU, and 3DS show lifetimes beginning after 2010, with no definitive end yet, indicating they are still actively supported at the time of the plot.

Trend Over Time:
- Earlier platforms (e.g., 2600, NES, SNES) saw shorter lifetimes, typically between 5-15 years, compared to more modern consoles, which often last longer due to extended support and backward compatibility.

Overall Insight:
The chart shows a clear trend of certain platforms, particularly PC and major console brands like PlayStation, having long lifetimes due to continued support and popularity. In contrast, some platforms with shorter lifetimes struggled to compete or were quickly superseded by new technology. The increasing platform longevity over time reflects the evolution of gaming hardware, software updates, and extended product support in the gaming industry.
            '''


##################
market_share_platform = '''Conclusion on Platform Market Share

- All three - NA, EU, and JP - markets are very different.

- X360, PS3, PS4, XOne, and 3DS are the most popular platforms in NA, while EU markets are dominated by PS3, PS4, X360, 3DS, and PC.

- Meanwhile, Japanese market is dominated by handheld platforms like 3DS, PS3, PSP, PSV, and PS4. Xbox has has 0.36% of Japanese market.

- The maximum market share by a particular platform is 48.2% - 3DS in JP)

- The console platforms are competitive in NA and EU.'''

market_share_genre = '''Conclusion on Genre Market Share
- The Japanese gaming market is significantly different from the american and european markets, with role-playing genres capturing approximately a third of the market. 

- Meanwhile, the North American and European markets are similar in terms of top four most popular genres, i.e., Action, Shotter, Sports, and Role-Playing genres. Additionally, their market share are almost identical. 

- Action genre does capture a significant amount Japanese market as well.'''


##################
sales_analysis_by_platform = '''This box plot shows the distribution of global sales (in millions) for various gaming platforms from 2013 to 2016. Key observations include:

Higher Sales Variability:

- PS4 (red), X360 (lightblue), XOne (green), and Wii (orange) show the largest variability in sales, as indicated by their wider interquartile ranges (IQR) and taller whiskers, meaning that individual game sales for these platforms vary significantly. Both also have several outliers with extremely high sales.

Platforms with Consistently Lower Sales:
- Platforms like PSV, PSP, and PC exhibit much lower overall sales with smaller box plots, indicating less variability and generally lower global sales compared to other platforms.

Median Sales:
- PS3, X360, and XOne show similar median sales, around the 0.5 million mark, as indicated by the horizontal line inside their boxes.
- Wii has the highest median sales, close to 1 million, reflecting its strong market presence during this time period.

Outliers:
- Almost all platforms have outliers, indicating some games that sold exceptionally well. Platforms like PS4, X360, Wii, and PS3 have multiple outliers above the 1 million sales mark, suggesting blockbuster titles that drove sales.

Sales Concentration:
- Platforms such as 3DS and WiiU show tighter distributions, indicating that most games had similar sales performance, with fewer extreme outliers.

Overall Insight:
The plot shows that while certain platforms like PS4 and Wii had high variability and outliers with very high sales, others like PC, PSV, and DS had more consistent but lower sales. The box plot effectively highlights the diversity in game sales performance across platforms during the 2013-2016 period, with some platforms significantly outperforming others.'''


##################
sales_analysis_by_genre = '''This box plot shows the global sales distribution for different video game genres from 2013 to 2016. Key observations include:

Highest Sales Variability:
- Sports, Shooter, and Platform genres have the largest variability in sales, as indicated by their wide interquartile ranges (IQR) and tall whiskers. This suggests that games in these genres have widely varying sales figures, with both low-selling and high-selling games.
- Both genres also have several outliers, with some games exceeding 1.5 million in global sales.

Genres with Lower Sales:
- Strategy, Puzzle, and Simulation genres exhibit lower overall sales, as seen by their shorter box plots and lower medians. These genres tend to have fewer high-selling games compared to others.
- The Puzzle genre, in particular, shows very little variability and lower sales overall, indicating it might be a niche genre with fewer major hits.

Median Sales:
- Most genres, such as Shooter, Role-Playing, and Fighting, have median sales below 0.5 million units, reflecting more consistent but moderate sales across these categories.
- Sports stands out with a relatively higher median sales figure compared to other genres.

Outliers:
- Several genres, including Action, Shooter, and Miscellaneous, have many outliers, showing that while the majority of games fall within a certain sales range, a few blockbuster titles significantly outperform the rest.

Overall Insight:
The plot highlights that Sports and Shooter genres tend to have higher variability and potential for high sales, while genres like Strategy, Puzzle, and Simulation have consistently lower sales. This could indicate that certain genres appeal more to broader audiences and are more likely to produce hit games compared to others that cater to more niche markets.'''


##################
esbr_ratings = '''### Rating glossary
ESRB (Entertainment Software Rating Board) evaluates a game's content an assigns an age rating. E.g. T (Teen) or M (Mature).

- E: Everyone
- T: Teen
- M: Mature 17+
- E10+: Everyone 10+
- EC: Early Childhood (Retired ratings)
- K-A: ???
- RP: Rating Pending
- AO: Adults only 18+'''


##################
esbr_conclusion = '''Conclusion: Effects of ESBR ratings on Regional Sales
There are 617 games with E rating, 456 with E10+ rating, 623 with M rating, and 616 with T rating. The rest of the rating include only small number of games. Therefore, not included in this conclusion.

NA and EY sales trends for different ratings, but with higher mean sales. This is also apparent in the box plot analysis.

Again, the JP sales for differnt ratings trends and distribution are different compared to NA and EU sales.

From earlier analysis of market share, it is shown that the JP market are differnt than of NA and EU. THe ESBR ratings for differnt genre and platform can significantly affect the regional sales.'''


##################
review_analysis = '''Analysis of correlation of Critics and User scores to global sales for:
Xbox 360:

All correlation metrics for X360 are low, except for Spearman correlation for critics score, which is reasonably high, 0.63.
Games with higher global sales tend to have higher critics scores. However, some of these games have low-user scores.
PS3:

Similar trends are observed for PS3 as X360.
Spearman correlation for critics score is relatively high, 0.59.
Games with higher global sales tend to have higher critics scores. However, some of these games have low-user scores.
Overall correlation conclusion

The PS4, X360, and PS3 platforms show generally low Pearson, Spearman, and Kendall correlations.
The critics score is more correlated to the global sales compared to user score. Therefore, critics score can be used to modell 2017 sales than user score.
In general, games with low user and critics score tend to have low-global sales.
However, not all high-score games have high-global sales, but high-global sales tend to have high scores. There are only few (5 games) that do not adhere to this conclusion, especially for X360 and PS3 platforms.'''