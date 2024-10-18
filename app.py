"""
Steamlit app for Online Games Sales 
- EDA
- Visualizaiton
- Hypothesis Testing
- Predictive Models

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels

from scipy import stats

import streamlit as st
# from streamlit_extras.widgets import no_default_selectbox

from utils.text import app_description, data_description, game_by_year_summary, sales_by_platform, sales_distribution, platform_lifetime_summary, market_share_platform, market_share_genre, sales_analysis_by_platform, sales_analysis_by_genre, esbr_ratings, esbr_conclusion, review_analysis

from utils.prediction import preprocess_data, split_data, standardize_data, train_rf_model, train_xgb_model, evaluate_model, feature_importance, predict_sales

sns.set_style('whitegrid')


# Load and preprocess data once
@st.cache_data
def load_data():
    data = pd.read_csv('./data/games.csv')
    data.columns = data.columns.str.lower()
    # Convert 'tbd' to NaN and then to float
    data['year_of_release'] = data['year_of_release'].astype('Int32')
    # This can later be filled with mean/median per genre
    data['user_score'] = data['user_score'].replace('tbd', np.nan).astype('float64')

    # Add total sales column
    data['total_sales'] = data[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)
    
    return data


def plot_market_shares(data, market='na_sales', category='platform'):
    labels = {'na_sales':'North American', 'eu_sales':'European', 'jp_sales':'Japanese', 
              'total_sales':'Global',
              'platform':'Platforms', 'genre':'Genres'
             }
    fig = px.pie(
        data, values=market, names=category,
        # title=f"{labels[market]} Market Sales Share (%) by {labels[category]}",
        color=category,
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4  # Create a donut chart for more visual appeal
    )
    fig.update_traces(
        textinfo='label+percent', textposition='inside',
        marker=dict(line=dict(color='white', width=2)),
        pull=[0.1 if value == data[market].max() else 0 for value in data[market]]  # Emphasize the highest value
    )
    fig.update_layout(
        showlegend=True, legend_title_text=labels[category],
        annotations=[dict(
            text=f"{labels[market]} Market", x=0.5, y=1, font_size=20, showarrow=False
        )],
        width=800, height=600,
        plot_bgcolor='black'
    )
    st.plotly_chart(fig)

def plot_scatter_and_correlate(data, platform):
    f_data = data.query('platform == @platform')
    corr_critic = f_data['critic_score'].corr(f_data['total_sales'])
    corr_user = f_data['user_score'].corr(f_data['total_sales'])

    fig = px.scatter(
        f_data, x='user_score', y='total_sales',
        title=f'Global Sales vs Scores for {platform}',
        labels={'critic_score': 'Critic Score', 'total_sales': 'Global Sales (Mil)'},
        color_discrete_sequence=['lightblue'],
        # trendline='ols'
    )
    fig.add_scatter(
        x=f_data['critic_score']/10, y=f_data['total_sales'],
        mode='markers',
        name=f'User Score (Pearson Correlation: {corr_user:.2f})',
        marker=dict(color='orange')
    )
    fig.update_layout(
        legend_title_text='Score Types',
        width=800, height=600, paper_bgcolor='lightgrey',
        xaxis_title='Score', yaxis_title='Global Sales (Mil)'
    )
        # platform_data, x='user_score', y='critic_score',
        # title=f'Correlation between User Score and Critic Score for {platform}',
        # labels={'user_score': 'User Score', 'critic_score': 'Critic Score'},
        # trendline='ols'
    # )
    fig.update_layout(width=800, height=600, )#paper_bgcolor='lightgrey')
    st.plotly_chart(fig)

    # More correlation comparison
    # Pearson correlation assumes normality, while Spearman and Kendall correlations works for data distribution without normal distirbution assumption.
    corrs = ['Pearson', 'Spearman', 'Kendall']
    for corr in corrs:
        corr_critic = f_data['critic_score'].corr(f_data['total_sales'], method=corr.lower())
        corr_user = f_data['user_score'].corr(f_data['total_sales'], method=corr.lower())

        st.write(f"{corr} correlation | Critics score: {corr_critic:.2f}, User score: {corr_user:.2f}")
        

def main():

    # Load data
    data = load_data()
    filtered_data = data[data['year_of_release']>=2013]

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Games Overview", "Games by Year",  "Platform Analysis",  "Sales Analysis", 
                                          "Market Share by Region", "Review and Rating Analysis", "Hypothesis Test",
                                          "Models"], index=None) # "Genre Analysis", "Year-over-Year Growth",

     # Customize the sidebar
    markdown = """
    Web App URL: <>
    GitHub Repository: <https://github.com/ruhil528/online_games_sales>
    """
    st.sidebar.title("About")
    st.sidebar.info(markdown)

    # Streamlit app
    st.title('Online Games Sales Analysis')
    st.write(app_description)
    st.markdown("""---""")
    

    ############################################################################
    if page == "Games Overview":
        st.header("Games Data Overview")
        st.write(data_description)
        # Show raw data if user requests
        data_preview = st.checkbox('Show Raw Sample Data')
        if data_preview:
            st.write('Five Sample data')
            st.write(data.sample(5))

        # Data description
        st.subheader('Data Description')
        st.write(data.describe())


    ############################################################################
    elif page == "Games by Year":
        st.header("Games Released by Year")
        # Plot the data
        fig, ax = plt.subplots()
        fig = px.bar(data['year_of_release'].value_counts().reset_index(),
                     x='year_of_release', y='count',
                     color_discrete_sequence=['skyblue'],
                     labels={'year_of_release': 'Year', 'count': 'Number of Games'})

        fig.update_traces(marker=dict(line=dict(color='black', width=1)))
        st.plotly_chart(fig)

        st.write(game_by_year_summary)

    
    ############################################################################
    elif page == "Platform Analysis":
        st.header("Platform Analysis")

        tab1, tab2, tab3 = st.tabs(["Sales by Platform", "Sales Distribution", "Platform Lifetime"])

        with tab1:
            st.subheader("Sales by Platform")
            fig = px.bar(
                data.groupby('platform')['total_sales'].sum().sort_values(ascending=False).reset_index(),
                x='platform', y='total_sales',
                color_discrete_sequence=['skyblue'],
                labels={'total_sales': 'Global Sales (Millions)', 'platform': 'Platform'})
            fig.update_traces(marker=dict(line=dict(color='black', width=1)))
            st.plotly_chart(fig)
    
            st.write(sales_by_platform)

        
        with tab2:
            st.subheader('Sales Distribution for Platforms')
            num_platforms = st.selectbox('Select Platforms with', ['Over 600 Million Sales', 'Over 100 Million Sales', 'All Sales'])

            if num_platforms == 'Over 600 Million Sales':
                st.caption('Platforms with more than 600 Million Sales')
                selected_platforms = ['PS2', 'X360', 'PS3', 'Wii', 'DS', 'PS']

            elif num_platforms == 'Over 100 Million Sales':
                st.caption('Platforms with more than 100 Million Sales')
                filtered_platforms = data.groupby('platform')['total_sales'].sum()
                selected_platforms = filtered_platforms[filtered_platforms > 100].index.tolist()

            elif num_platforms == 'All Sales':
                st.caption('Platforms with Any Amount of Sales')
                selected_platforms = data['platform'].unique().tolist()
                
    
            # Distribution of PS2 games released on differnt years.
            # Select platforms with at least 600 Mil sales from previous bar plot
            fig = px.histogram(data.query('platform == @selected_platforms'), x='year_of_release', y='total_sales', 
                               color='platform',barmode='overlay',
                               title="Sales Distribution of Platforms"
                              )
            
            fig.update_layout(xaxis_title='Year', yaxis_title='Total Sales (Millions)')
            st.plotly_chart(fig)
    
            st.write(sales_distribution)

        with tab3:
            st.subheader("Platform Lifetime")

             # Plot the platform lifetime as a horizontal bar graph
            platform_lifetime = data.groupby('platform')['year_of_release'].agg(['min', 'max']).sort_values('min').reset_index()
            platform_lifetime['lifetime'] = platform_lifetime['max'] - platform_lifetime['min'] # Life time of a platform
            platform_lifetime['years_diff'] = platform_lifetime['min'].diff()
            platform_lifetime['years_diff'] = platform_lifetime['years_diff'].fillna(0).astype(int)
            avg_lifetime = platform_lifetime['lifetime'].mean()
            avg_new_platform_time = platform_lifetime["years_diff"].mean()
            
            # Convert min and max years to datetime objects
            platform_lifetime['min'] = pd.to_datetime(platform_lifetime['min'], format='%Y')
            platform_lifetime['max'] = pd.to_datetime(platform_lifetime['max'], format='%Y')
            fig = px.timeline(platform_lifetime, x_start="min", x_end="max", y="platform",
                              title="Platform Lifetime (Years)",
                              labels={"platform": "Platform", "year_of_release": 'Year'},
                              color_discrete_sequence=['skyblue'],
            )
            fig.update_traces(marker=dict(line=dict(color='black', width=1)))
            fig.update_layout(height=800)
            fig.update_yaxes(autorange="reversed", showgrid=True, tickmode='linear') # otherwise tasks are listed from the bottom up
            fig.update_xaxes(showgrid=True)
            st.plotly_chart(fig)

            st.write(platform_lifetime_summary(avg_lifetime, avg_new_platform_time))

    
    ############################################################################
    # elif page == "Genre Analysis":
    #     st.header("Genre Analysis")
    #     top_na_genres = data.groupby('genre')['na_sales'].sum().sort_values(ascending=False)[:5]
    #     st.bar_chart(top_na_genres)

    
    ############################################################################
    elif page == "Sales Analysis":
        st.header("Sales Analysis")
        st.caption("Sales data for 2013-2016")

        tab1, tab2 = st.tabs(['By Platform', 'By Genre'])
        with tab1:
            fig = px.box(filtered_data, x='platform', y='total_sales', hover_data=['name', 'platform', 'year_of_release', 'total_sales',], 
                         color='platform',
                         title="Box Plot of Global Sales by Platform",
                        )
            fig.update_layout(xaxis_title='Platform', yaxis_title='Global Sales (Mil)')
            fig.update_yaxes(range=[-0.1, 2], ) # Using ylim for better visualiztion
            fig.update_xaxes(showgrid=True)
            st.plotly_chart(fig)
    
            st.write(sales_analysis_by_platform)
            
        with tab2:
            fig = px.box(filtered_data, x='genre', y='total_sales', hover_data=['name', 'platform', 'genre', 'year_of_release', 'total_sales',], 
                         color='genre',
                         title="Box Plot of Global Sales by Genre",
                        )
            fig.update_layout(xaxis_title='Genre', yaxis_title='Global Sales (Mil)')
            fig.update_yaxes(range=[-0.1, 2], ) # Using ylim for better visualiztion
            fig.update_xaxes(showgrid=True)
            st.plotly_chart(fig)
    
            st.write(sales_analysis_by_genre)

    
    ############################################################################
    elif page  == "Market Share by Region":
        st.header("Market Share by Region")
        st.caption("Sales data for 2013-2016")
        filtered_data = data[data['year_of_release']>=2013]

        tab1, tab2 = st.tabs(["Platform", "Genre"])

        with tab1: 
            st.subheader("Market Share of Platform by Region")
            col1, col2 = st.columns(2)
            with col1:
                plot_market_shares(filtered_data, 'total_sales', 'platform')
            with col2:
                plot_market_shares(filtered_data, 'na_sales', 'platform')
    
            col1, col2 = st.columns(2)
            with col1:
                plot_market_shares(filtered_data, 'eu_sales', 'platform')
            with col2:
                plot_market_shares(filtered_data, 'jp_sales', 'platform')

            st.write(market_share_platform)

        with tab2:
            st.subheader("Market Share of Genre by Region")
            col1, col2 = st.columns(2)
            with col1:
                plot_market_shares(filtered_data, 'total_sales', 'genre')
            with col2:
                plot_market_shares(filtered_data, 'na_sales', 'genre')
    
            col1, col2 = st.columns(2)
            with col1:
                plot_market_shares(filtered_data, 'eu_sales', 'genre')
            with col2:
                plot_market_shares(filtered_data, 'jp_sales', 'genre')
            
            st.write(market_share_genre)

    
    ############################################################################
    # elif page == "Year-over-Year Growth":
    #     st.header("Year-over-Year Growth for Platforms")
    #     platform_sales = data.groupby(['year_of_release', 'platform'])['na_sales'].sum().reset_index()
    #     platform_sales = platform_sales.pivot(index='year_of_release', columns='platform', values='na_sales').fillna(0)
    #     yoy_growth = platform_sales.pct_change().fillna(0) * 100
    #     st.write("Year-over-Year Growth (%) for Platforms")
    #     st.dataframe(yoy_growth)

    #     # Line plot for Year-over-Year Growth
    #     yoy_growth_long = yoy_growth.reset_index().melt(id_vars='year_of_release', var_name='Platform', value_name='Growth (%)')
    #     fig = px.line(
    #         yoy_growth_long, x='year_of_release', y='Growth (%)', color='Platform',
    #         title='Year-over-Year Growth for Platforms',
    #         labels={'year_of_release': 'Year of Release', 'Growth (%)': 'Growth (%)'},
    #         markers=True
    #     )
    #     fig.update_layout(width=800, height=600)
    #     st.plotly_chart(fig)

        
        
    ############################################################################
    elif page == "Review and Rating Analysis":
        st.header("Review and Rating Analysis")
        st.caption("Sales data for 2013-2016")
        filtered_data = data[data['year_of_release']>=2013]

        tab1, tab2 = st.tabs(['Critics and User Review Correlation', 'ESBR Rating Effect',])
        with tab1:
            platform = st.selectbox('Select Platform', ['PS4', 'X360', 'PS3'])
            plot_scatter_and_correlate(filtered_data, platform)

            st.write(review_analysis)
            
        with tab2:
            st.write(esbr_ratings)
            # Bar chart
            fig = px.bar(
                filtered_data.groupby('rating')[['na_sales', 'eu_sales', 'jp_sales']].sum().reset_index(),
                x='rating', y=['na_sales', 'eu_sales', 'jp_sales'],
                title='Sales by ESBR Ratings',
                labels={'rating': 'ESBR Ratings', 'value': 'Sales (Mil)'},
                barmode='group'
            )
            fig.update_layout(xaxis_title='ESBR Ratings', yaxis_title='Sales (Mil)', width=800, height=600)
            st.plotly_chart(fig)

            # Box plot
            region = st.selectbox('Select Region', ['total_sales', 'na_sales', 'eu_sales', 'jp_sales'])
            fig = px.box(data.dropna(), x='rating', y=region, hover_data=['name', region,], color='rating',
             labels={'E':"Everyone"},
             title='North America Sales by ESBR Ratings'
            )
            fig.update_layout(xaxis_title='ESBR Ratings', yaxis_title='Sales (Mil)', 
                              yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig)

            # Distribution
            ratings1 = ['E', 'E10+', 'M', 'T']
            fig = px.histogram(filtered_data.query('rating == @ratings1'), x='na_sales', color='rating', barmode='overlay',
                               labels={''},
                               marginal='box', 
                        )
            fig.update_layout(xaxis_title='ESBR Ratings', yaxis_title='Sales (Mil)', 
                              xaxis=dict(range=[0, 5]))
            st.plotly_chart(fig)

            st.write(esbr_conclusion)


    ############################################################################
    elif page == "Hypothesis Test":
        st.header("Statistical Hypothesis Test")
        
        tab1, tab2 = st.tabs(['Test 1', 'Test 2'])
        with tab1:
            st.write("""Null hypothesis: The mean of the user score distributions of Xbox One and PC gaming platforms are equal.
H0: Mean(Xbox One) = Mean(PC)

Alternative hypothesis: The mean of the user score distribution of Xbox One and PC gaming platforms are not equal.
H1: Mean(Xbox One) != Mean(PC)""")
            # Distribution
            fig = px.histogram(filtered_data.query('platform == "XOne" or platform == "PC"'), x='user_score', color='platform', 
                               marginal='box', barmode='overlay',
                               title='Distribution of User Score for PC and Xbox One platform (Vertical line indicates mean)',
                               labels={'user_score':'User Score'},
                               color_discrete_map={'XOne': 'orange',  # Color for Action genre
                                                   'PC': 'lightblue'  # Color for Sports genre
                                                  }
                              )
    
            fig.add_vline(x=filtered_data.query('platform == "PC"')['user_score'].mean(), line_dash='dash', line_color='lightblue', )
                          #annotation_text="PC Mean", annotation_position="top left", )
            fig.add_vline(x=data.query('platform == "XOne"')['user_score'].mean(), line_dash='dash', line_color='orange', )
                          #annotation_text="Xbox One Mean", annotation_position="top left")
            
            fig.update_layout(yaxis_title='Number of Users',)
            st.plotly_chart(fig)
            
            st.write(f"Mean of Xbox One platform user score {filtered_data.groupby('platform')['user_score'].mean()['XOne']:.2f}")
            st.write(f"Mean of PC platform user score {filtered_data.groupby('platform')['user_score'].mean()['PC']:.2f}")
    
            # Perform two_sample t-tests
            xone_ratings = filtered_data.query('platform == "XOne"')['user_score'].dropna()
            pc_ratings = filtered_data.query('platform == "PC"')['user_score'].dropna()
            
            # t-test
            t_stat, p_value = stats.ttest_ind(xone_ratings, pc_ratings)
            
            st.write(f"T-statistics: {t_stat}")
            st.write(f"P-value: {p_value}")
            
            # Interpretation
            alpha = 0.05
            if p_value < alpha:
                st.write("Reject the null hypothesis: The average user ratings for Xbox One and PC are equal.")
            else:
                st.write("Failed to reject the null hypothesis, i.e., the average user ratings of Xbox One and PC are equal.")

        with tab2:
            st.write('''Null hypothesis: The mean of the user score distributions of Xbox One and PC gaming platforms are equal.
H0: Mean(Xbox One) = Mean(PC)

Alternative hypothesis: The mean of the user score distribution of Xbox One and PC gaming platforms are not equal.
H1: Mean(Xbox One) != Mean(PC)''')
            # Distribution
            fig = px.histogram(filtered_data.query('genre == "Action" or genre == "Sports"'), x='user_score', color='genre', 
                                   marginal='box', barmode='overlay',
                                   title='Distribution of User Score for Action and Sports Genre (Vertical line indicates mean)',
                                   labels={'user_score':'User Score'},
                                   color_discrete_map={'Action': 'orange',  # Color for Action genre
                                                       'Sports': 'lightblue'  # Color for Sports genre
                                                      }
                                          )
        
            fig.add_vline(x=filtered_data.query('genre == "Action"')['user_score'].mean(), line_dash='dash', line_color='orange', )
                          #annotation_text="PC Mean", annotation_position="top left", )
            fig.add_vline(x=filtered_data.query('genre == "Sports"')['user_score'].mean(), line_dash='dash', line_color='lightblue', )
                          #annotation_text="Xbox One Mean", annotation_position="top left")
            
            fig.update_layout(yaxis_title='Number of Users',)
            st.plotly_chart(fig)
            
            st.write(f"Mean of Action genre user score {filtered_data.groupby('genre')['user_score'].mean()['Action']:.2f}")
            st.write(f"Mean of Sports genre user score {filtered_data.groupby('genre')['user_score'].mean()['Sports']:.2f}")

            
            # Perform two_sample t-tests
            xone_ratings = filtered_data.query('genre == "Action"')['user_score'].dropna()
            pc_ratings = filtered_data.query('genre == "Sports"')['user_score'].dropna()
            
            # t-test
            t_stat, p_value = stats.ttest_ind(xone_ratings, pc_ratings)
            
            st.write(f"T-statistics: {t_stat}")
            st.write(f"P-value: {p_value}")
            
            # Check p_value against alpha
            alpha = 0.05
            if p_value < alpha:
                st.write("Reject the null hypothesis: The average user ratings for Action and Sports genre are significantly different.")
            else:
                st.write("Failed to reject the null hypothesis: There is no significant different in average user ratings between Action and Sports genre.")
        
        

    ############################################################################
    elif page == "Models":
        st.header("Global Sales Prediction")
        st.write("""
        A Random Forest Regressor model is trained to predict the global sales of video games, using features like platform, genre, year of release, critic score, and user score. This predictive model can help stakeholders make informed decisions about which factors contribute most to the success of a game.
        """)

        st.subheader('Train Model')
        # Encode features
        data_encoded = preprocess_data(filtered_data)

        # Split train-test data
        X_train, X_test, y_train, y_test = split_data(data_encoded)

        # Standardize data
        X_train, X_test, scaler = standardize_data(X_train, X_test)

        # Train model
        # tab1, tab2 = st.tabs('Random Forest', 'XGBoost')
        model_name = st.selectbox('Select a model', ['Random Forest', 'XGBoost'])
        if model_name == 'Random Forest':
            col1, col2, col3 = st.columns(3)
            n_estimators = col1.number_input("Enter the Number of Estimators", value=100,) #placeholder="Type a number...")
            max_depth = col2.number_input("Enter Max Depth", value=20)
            min_samples_split = col3.number_input("Enter Minimum Sample Split", value=10)
            model = train_rf_model(X_train, y_train, n_estimators, max_depth, min_samples_split)
        elif model_name == 'XGBoost':
            col1, col2, col3 = st.columns(3)
            n_estimators = col1.number_input("Enter the Number of Estimators", value=100,) #placeholder="Type a number...")
            learning_rate = col2.number_input("Enter Learning Rate", value=0.1)
            max_depth = col3.number_input("Enter Maximum Depth", value=6)
            model = train_xgb_model(X_train, y_train, n_estimators, learning_rate, max_depth)

        # Evaluate model
        y_train_pred, y_test_pred, train_mae, train_r2, test_mae, test_r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Plot model prediction
        train_predictions_df = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred, 'Set': 'Training'}).reset_index(drop=True)
        test_predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred, 'Set': 'Testing'}).reset_index(drop=True)
        combined_predictions_df = pd.concat([train_predictions_df, test_predictions_df])

        fig = px.scatter(combined_predictions_df, x='Actual', y='Predicted', color='Set',
                         title='Actual vs Predicted Global Sales (Training and Testing Sets)',
                         labels={'Actual': 'Actual Sales', 'Predicted': 'Predicted Sales'},
                         trendline='ols')
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

        st.write(f'Mean Absolute Error - Train: {train_mae} | Test: {test_mae}')
        st.write(f'R² Score - Train: {train_r2} | Test: {test_r2}')
        
        
        st.markdown('---')
        st.subheader('Feature Importance')
        importance_df = feature_importance(model, X_train)
        importance_df = importance_df.loc[:10]

        # Plot feature importance
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                title='Feature Importance', labels={'Importance': 'Importance', 'Feature': 'Feature'},
                                color_discrete_sequence=['skyblue'],)
        fig.update_traces(marker=dict(line=dict(color='black', width=1)))
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)
        
if __name__ == "__main__":
    main()