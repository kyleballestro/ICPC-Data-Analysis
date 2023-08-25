#!/usr/bin/env python
# coding: utf-8

# # **CS433-740 Final Project** 
# ## Group 1: Kyle Ballestro, Phillip Dunning, Collin Rosborg, Joe Scanga, Trenton Berck
# ## Data Analysis of ICPC World Finals Ranking Results 1999-Present
# ### https://www.kaggle.com/code/dekomorisanae09/icpc-exploratory-data-analysis

# ## Data Loading
# ##### Import the appropriate libraries and packages. The primary library and package used will be Matplotlib for plotting and Pandas for data analysis.

# In[ ]:


import string
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import datetime
import textwrap
from statistics import mean
import pdb
import csv
import re
import math
import seaborn as sns
import sys
import operator
from decimal import Decimal as D

df = pd.read_csv('icpc-full.csv')


# ### Display first 10 rows of unclean data
# ##### The initial columns of the data consist of Year, Date, Host, City, Venue, Rank, University, Country, Team, Contestant 1, Contestant 2, Contestant 3, Gold, Silver, Bronze, Honorable, Score, Total, Score Percentage, Penalty, and Prize. Many of these columns are irrelevant, difficult to wrangle, and missing information.

# In[ ]:


print(df.head(5))


# ## Data Cleaning

# ### Drop unnecessary columns
# ##### Some of the columns in the dataset are not necessary for our purposes and therefore need to be dropped.

# In[ ]:


df = df.drop(['City', "Venue", "Contestant 1", "Contestant 2", "Contestant 3"], axis='columns')


# ### Change True and False to 1 and 0, respectively
# ##### Gold, Silver, Bronze, and Honorable columns currently have True/False values, however we will change them to 1 or 0 in order to handle them more easily.

# In[ ]:


df['Gold'].mask(df['Gold'] == True, 1, inplace=True)
df['Gold'].mask(df['Gold'] == False, 0, inplace=True)
df['Silver'].mask(df['Silver'] == True, 1, inplace=True)
df['Silver'].mask(df['Silver'] == False, 0, inplace=True)
df['Bronze'].mask(df['Bronze'] == True, 1, inplace=True)
df['Bronze'].mask(df['Bronze'] == False, 0, inplace=True)
df['Honorable'].mask(df['Honorable'] == True, 1, inplace=True)
df['Honorable'].mask(df['Honorable'] == False, 0, inplace=True)


# ### Handle the ranks that are shown as ranges in the CSV
# ##### Some of the ranks of countries are stored as a range (i.e. 40-43) if they have the same statistics. We will rework this so that we can perform math later on the ranks.

# In[ ]:


'''
If a country's rank is "qualified" in the csv, they're rank is automatically set to 50; if it's "finalist", it's set to 10. 
Since there are so few of these types of string entries (and these constants hover around what their true rank number probably was in those instances), 
this is an adequately accurate solution. If a country's rank is a range of numbers (ex.: 46 - 64), their new rank number will be the average of the two (in this example, 55).
'''
rankList = df['Rank'].to_list()
for i in range(len(rankList)):
    if '-' in str(rankList[i]):
        rankList[i] = mean([int(j) for j in rankList[i].split() if j.isdigit()])
    elif 'qualified' in str(rankList[i]):
        rankList[i] = 50
    elif 'finalist' in str(rankList[i]):
        rankList[i] = 10
    else:
        rankList[i] = float(rankList[i])
df['Rank'] = pd.Series(rankList)


# ### Rename columns, change Date to actual datetime
# ##### Some of the column names are long or ambiguous. We will rename these in order to make them easier to work with.

# In[ ]:


df.rename(columns = {'University':'Uni', 'Score Percentage':'Score %', 'Total':'Total Score'}, inplace = True)
df['Date'] = pd.to_datetime(df['Date'])


# ### Display first 10 rows of the newly cleaned data
# ##### After totally cleaning the data, this is its current state.

# In[ ]:


print(df.head(5))


# ## Data Wrangling/Aggregation

# ### The top 10 universities with the most medals
# ##### This shows the top 10 performing universities were based on how many total medals they received.

# In[ ]:


uni_medals = df.groupby('Uni')[['Gold', 'Silver', 'Bronze']].sum()
uni_medals['Total Medals'] = uni_medals['Gold'] + uni_medals['Silver'] + uni_medals['Bronze'] 
print("The top 10 universities with the most medals\n", uni_medals.nlargest(10, 'Total Medals'))


# ### The top 10 countries that got penalized the most
# ##### This shows the top 10 countries that received the highest total penalty. This data is interesting because most of the top performing countries are also among this list of the top penalized countries.

# In[ ]:


penalty = df.groupby('Country')[['Penalty']].sum()
print("The top 10 countries that got penalized the \n", penalty.nlargest(10, 'Penalty'))


# ### Penalty outliers along with the countries that received them and the year they were received. Outliers are found using the IQR method.
# ##### This uses the IQR method of detecting outliers in order to find any countries that had an outstanding penalty for every year. This data shows that some years, there were multiple outlying penalized countries whereas other years consisted of no outliers.

# In[ ]:


outliers = df.loc[:, ['Year', 'Uni', 'Penalty']]
outliers['Penalty'] = df['Penalty'].fillna(0)
out = pd.DataFrame(columns = ['Year', 'Outlier Penalty', 'Uni'])
i = 0
years = df['Year']
years = years.drop_duplicates()
outlierDict = {}
for year in years:
    curYearSet = outliers.loc[df['Year'] == year, ['Penalty', 'Uni']]
    curYearSet = curYearSet.sort_values(by = 'Penalty')
    med = curYearSet['Penalty'].median()
    q1 = curYearSet['Penalty'].quantile(0.25)
    q3 = curYearSet['Penalty'].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + (1.5 * iqr)
    lower_fence = q1 - (1.5 * iqr)
    for index, penaltyVal in curYearSet.iterrows():
        if penaltyVal['Penalty'] > upper_fence or penaltyVal['Penalty'] < lower_fence:
            out.loc[i] = [year, penaltyVal['Penalty'], penaltyVal['Uni']]
            i += 1
print('Universities that had outlying (exceptionally high) penalties compared to the other universities:\n', out.to_string())
for i in outlierDict:
    print(i, 'had', outlierDict[i], 'penalty value that was an outlier.') if outlierDict[i] == 1 else print(i, 'had', outlierDict[i], 'penalty values that were outliers.')
    outlierDict[i] = 0


# ### Every country and their amount of gold, silver, and bronze medals. This is shown per year and over the entire course of time.
# ##### This shows what countries received any medals at all, year by year.

# In[ ]:


fig,ax = plt.subplots()
all_country = pd.unique(df['Country']).tolist()
all_country_not_unique = df['Country'].tolist()
all_medals_gold = []
all_medals_silver = []
all_medals_bronze = []

all_gold = df['Gold'].tolist()
all_sil = df['Silver'].tolist()
all_bron = df['Bronze'].tolist()


for e in all_country:
    num_gold = 0
    num_sil = 0
    num_bron = 0
    for idx, c in enumerate(all_country_not_unique):
        if (e == c):
            if (all_gold[idx] == 1):
                num_gold = num_gold + 1
            if (all_sil[idx] == 1):
                num_sil = num_sil + 1
            if (all_bron[idx] == 1):
                num_bron = num_bron + 1
    all_medals_gold.append(num_gold)
    all_medals_silver.append(num_sil)
    all_medals_bronze.append(num_bron)


cur_medal_gold = []
cur_medal_silver = []
cur_medal_bronze = []
cur_Country = []

for index, yy in enumerate(all_country):
    if ((all_medals_gold[index] != 0) or (all_medals_silver[index] != 0) or (all_medals_bronze[index] != 0)):
        cur_Country.append(all_country[index])
        if(all_medals_gold[index] != 0):
            cur_medal_gold.append(all_medals_gold[index])
        else:
            cur_medal_gold.append(0)
        if(all_medals_silver[index] != 0):
            cur_medal_silver.append(all_medals_silver[index])
        else:
            cur_medal_silver.append(0)
        if(all_medals_bronze[index] != 0):
            cur_medal_bronze.append(all_medals_bronze[index])
        else:
            cur_medal_bronze.append(0)
info = {}
for idx, hh in enumerate(cur_Country):
    info[hh] = [cur_medal_gold[idx], cur_medal_silver[idx], cur_medal_bronze[idx]]
dfff = pd.DataFrame(info).T
width = .5
dfff.plot(kind="bar", figsize=(15,5), title='Gold, Silver, and Bronze Medals From 1999 - 2021', xlabel="Country", ylabel="# of Medals", ax=ax)
ax.legend(["Gold", "Silver", "Bronze"])
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, 1))


#Medals broken up for that year and country
start_year = 1999
final_dict = {1999: {}, 2000:  {}, 2001:  {}, 2002:  {}, 2003:  {}, 2004:  {}, 2005:  {},
              2006:  {}, 2007:  {}, 2008:  {}, 2009:  {}, 2010:  {}, 2011:  {}, 2012:  {},
              2013:  {}, 2014:  {}, 2015:  {}, 2016:  {}, 2017:  {}, 2018:  {}, 2019:  {}, 
              2020:  {}, 2021:  {}}
for i in range(23):
    test_df = df.loc[df['Year'] == start_year]

    unique_uni = pd.unique(test_df['Country']).tolist()
    all_uni_test = test_df['Country'].tolist()
    all_gold = test_df['Gold'].tolist()
    all_sil = test_df['Silver'].tolist()
    all_bron = test_df['Bronze'].tolist()

    all_medals_gold = []
    all_medals_silver = []
    all_medals_bronze = []
    
    for e in unique_uni:
        num_gold = 0
        num_sil = 0
        num_bron = 0
        for idx, c in enumerate(all_uni_test):
            if (e == c):
                if (all_gold[idx] == 1):
                    num_gold = num_gold + 1
                if (all_sil[idx] == 1):
                    num_sil = num_sil + 1
                if (all_bron[idx] == 1):
                    num_bron = num_bron + 1
        all_medals_gold.append(num_gold)
        all_medals_silver.append(num_sil)
        all_medals_bronze.append(num_bron)
    
    for idx, bb in enumerate(unique_uni):
        if (all_medals_gold[idx] != 0 or all_medals_silver[idx] != 0 or all_medals_bronze[idx] != 0):
            final_dict[start_year][bb] = ["Gold: " + str(all_medals_gold[idx]), "Silver: " + str(all_medals_silver[idx]), "Bronze: " + str(all_medals_bronze[idx])]
    start_year = start_year + 1
for k, v in final_dict.items():
    print(k, ':')
    for key, value in v.items():
        print(key, '-->', value)


# ### Store the universities with medals each year in a dictionary, then display those statistics
# ##### This shows what universities received any medals as well as the total amount of medals. It is displayed in text for each year, as well as a bar graph with each university and their total number of medals.

# In[ ]:


all_unies = pd.unique(df['Uni']).tolist()
all_unies_not_unique = df['Uni'].tolist()
all_medals = []
fig = plt.figure(figsize=(50,25))

all_gold = df['Gold'].tolist()
all_sil = df['Silver'].tolist()
all_bron = df['Bronze'].tolist()


for e in all_unies:
    num_gold = 0
    num_sil = 0
    num_bron = 0
    for idx, c in enumerate(all_unies_not_unique):
        if (e == c):
            if (all_gold[idx] == 1):
                num_gold = num_gold + 1
            if (all_sil[idx] == 1):
                num_sil = num_sil + 1
            if (all_bron[idx] == 1):
                num_bron = num_bron + 1
    all_medals.append(num_gold + num_sil + num_bron)


cur_medal = []
cur_uni = []

for index, yy in enumerate(all_unies):
    if(all_medals[index] != 0):
        cur_medal.append(all_medals[index])
        cur_uni.append(all_unies[index])

res = dict(zip(cur_uni,cur_medal))
sorted_res = sorted(res.items(), key=lambda x:x[1], reverse=True)
good_dict = dict(sorted_res)


names = list(good_dict.keys())
values = list(good_dict.values())
indexval = 1
for ll in range(0,99,20):
    ax1 = fig.add_subplot(5,1,indexval)
    ax1.bar(['\n'.join(textwrap.wrap(n, 20)) for n in names[ll: ll + 20]], values[ll: ll + 20], color="maroon", width=0.1)
    ax1.set_title("Number of Medals Won by Universities From 1999 - 2021")
    ax1.set_ylabel("Total Medal Count (Gold + Silver + Bronze)")
    ax1.set_xlabel("Universities")
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, 1))
    ax1.grid()
    indexval += 1
plt.tight_layout()
plt.show()

start_year = 1999
final_dict = {1999: [], 2000: [], 2001: [], 2002: [], 2003: [], 2004: [], 2005: [],
              2006: [], 2007: [], 2008: [], 2009: [], 2010: [], 2011: [], 2012: [],
              2013: [], 2014: [], 2015: [], 2016: [], 2017: [], 2018: [], 2019: [], 
              2020: [], 2021: []}
#Universities that have won a medal for that year. (Gold, Silver, Bronze ONLY)
for i in range(23):
    test_df = df.loc[df['Year'] == start_year]

    unique_uni = pd.unique(test_df['Uni']).tolist()
    medal_count = []
    all_uni_test = test_df['Uni'].tolist()
    all_gold = test_df['Gold'].tolist()
    all_sil = test_df['Silver'].tolist()
    all_bron = test_df['Bronze'].tolist()

    

    for e in unique_uni:
        num_gold = 0
        num_sil = 0
        num_bron = 0
        for idx, c in enumerate(all_uni_test):
            if (e == c):
                if (all_gold[idx] == 1):
                    num_gold = num_gold + 1
                if (all_sil[idx] == 1):
                    num_sil = num_sil + 1
                if (all_bron[idx] == 1):
                    num_bron = num_bron + 1
        medal_count.append(num_gold + num_sil + num_bron)
    
    for idx, bb in enumerate(unique_uni):
        if (medal_count[idx] != 0):
            final_dict.get(start_year).append(bb)
    start_year = start_year + 1
for k, v in final_dict.items():
    print (k, '-->', v)


# ### Record when a country hosted the competition and also won the World Champion prize
# ##### This shows if there is any home field advantage by wrangling the data to find out how many times a country has hosted and also won the World Champion prize. It is concluded that there is a slight home field advantage given that 21.74% of the time, this outcome has happened. We displayed this using a line plot.

# In[ ]:


fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(1,1,1)
all_host = pd.unique(df['Host']).tolist()
all_country_not_unique = df['Country'].tolist()
all_prize = df['Prize'].tolist()
world_champ = []
world_champ_count = 0
good_countries = []
for ll in all_prize:
    cc = ll
    if (type(cc) == str):
        result = [bb.strip() for bb in cc.split(',')]
        if (result[0] == 'World Champion'):
            world_champ_count += 1

good_dict = {}

for index, row in df.iterrows():
    if (row['Host'] == row['Country']):
        x = row['Prize']
        if (type(x) == str):
            result = [bb.strip() for bb in x.split(',')]
            if (result[0] == 'World Champion'):
                allKeys = good_dict.keys()
                if row['Host'] in allKeys:
                    #key found
                    good_dict[str(row['Host'])] = good_dict.get(row['Host']) + 1
                else:
                    #add new
                    good_dict[str(row['Host'])] = 1
            else:
                continue

res = good_dict
sorted_res = sorted(res.items(), key=lambda x:x[1], reverse=True)
good_dict = dict(sorted_res)


names = list(good_dict.keys())
values = list(good_dict.values())

good_Val = sum(values)/world_champ_count
percentage = D(good_Val) * 100
print("The percentage of host countries winning the world championship: {0:.2f} %".format(percentage))
ax1.stem(['\n'.join(textwrap.wrap(n, 20)) for n in names], values)
ax1.set_title("Number of World Champions Won by a Country When Hosting From 1999 - 2021")
ax1.set_ylabel("Total World Champion Count While Hosting")
ax1.set_xlabel("Country")
plt.tight_layout()
plt.show()


# ### Wrangle the countries and their scores, then display their total scores versus their average scores in a double bar graph
# ##### This shows what each country received as their total score versus what their average score is for each year; we can clearly see whether each country is performing well over the years.

# In[ ]:


years = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
all_country = pd.unique(df['Country']).tolist()
indexval = 1
fig = plt.figure(figsize=(15,350))

for e in all_country:
    ax1 = fig.add_subplot(76,1,indexval)
    score_avg = []
    total_score_avg = []
    start_year = 1999
    for i in range(23):
        test_df = df.loc[df['Year'] == start_year]
        cou = test_df['Country'].tolist()
        score = test_df['Score'].tolist()
        total_score = test_df['Total Score'].tolist()
        score_num = 0
        score_run = 0
        total_score_num = 0
        total_score_run = 0
        for idx, c in enumerate(cou):
            if (e == c):
                score_num = score_num + 1
                total_score_num = total_score_num + 1
                score_run = score_run + score[idx]
                total_score_run = total_score_run + total_score[idx]
        if (score_num == 0):
            score_avg.append(0)
        else:
            score_avg.append(score_run/score_num)
        if (total_score_num == 0):
            total_score_avg.append(0)
        else:
            total_score_avg.append(total_score_run/total_score_num)
        start_year = start_year + 1
    ax1.bar(years,total_score_avg)
    ax1.bar(years,score_avg, color='black')
    ax1.set_title(e)
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Score/Total Score")
    ax1.legend(['Total Score', 'Avg. Score'])
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start, end, 1))
    ax1.xaxis.set_ticks(np.arange(1999, 2022, 1))
    indexval = indexval + 1
plt.tight_layout()
plt.show()


# ### Observe each university's performance by marking their score percentage each year to determine if any universities have been improving over time
# ##### This shows each university's score percentage for each year, as depicted by the line graphs. This allows us to see general trends for each university and determine how their performance has improved or worsened.

# In[ ]:


years = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
all_uni = pd.unique(df['Uni']).tolist()
fig_index = 1
indexval = 1
running_tally = 0
for e in all_uni:
    fig = plt.figure(fig_index, figsize=(25,150))
    ax1 = fig.add_subplot(20,1,indexval)
    score_percentage = []
    start_year = 1999
    for i in range(23):
        test_df = df.loc[df['Year'] == start_year]
        Uni = test_df['Uni'].tolist()
        score_val = test_df['Score %'].tolist()
        score_percentage_num = 0
        score_percentage_val = 0
        for idx, c in enumerate(Uni):
            if (e == c):
                score_percentage_num = score_percentage_num + 1
                score_percentage_val = score_percentage_val + score_val[idx]
        if (score_percentage_num == 0):
            score_percentage.append(0)
        else:
            score_percentage.append(score_percentage_val/score_percentage_num)
        start_year = start_year + 1
    ax1.set_title(e)
    ax1.stem(years,score_percentage)
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Score %")
    ax1.yaxis.set_ticks(np.arange(0.0, 1.1, 0.1))
    ax1.xaxis.set_ticks(np.arange(1999, 2022, 1))
    indexval = indexval + 1
    running_tally = running_tally + 1
    if (running_tally % 20 == 0):
        fig_index = fig_index + 1
        indexval = 1
plt.tight_layout()
plt.show()


# ### Every 3 years, determine the total medal count for each country
# ##### This shows each country's total number of medals per 3 year periods. This gives us a good overview to easily see overall trends.

# In[ ]:


years = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
all_country = pd.unique(df['Country']).tolist()
indexval = 1
fig = plt.figure(figsize=(15,250))

for e in all_country:
    ax1 = fig.add_subplot(76,1,indexval)
    medal_count = []
    year_chunk = []
    for i in range(0,22,3):
        if (i + 2 == 23):
            test_df = df.loc[(df['Year'] == years[i]) | (df['Year'] == years[i + 1])]
        else:
            test_df = df.loc[(df['Year'] == years[i]) | (df['Year'] == years[i + 1]) | (df['Year'] == years[i + 2])]
        cou = test_df['Country'].tolist()
        all_gold = test_df['Gold'].tolist()
        all_sil = test_df['Silver'].tolist()
        all_bron = test_df['Bronze'].tolist()
        num_gold = 0
        num_sil = 0
        num_bron = 0
        total_count = 0
        for idx, c in enumerate(cou):
            if (e == c):
                if (all_gold[idx] == 1):
                    num_gold = num_gold + 1
                    total_count += 1
                if (all_sil[idx] == 1):
                    num_sil = num_sil + 1
                    total_count += 1
                if (all_bron[idx] == 1):
                    num_bron = num_bron + 1
                    total_count += 1
        medal_count.append(num_bron + num_sil + num_gold)
        if (i + 2 == 23):
            year_chunk.append(str(years[i]) + " - " + str(years[i + 1]))
        else:
            year_chunk.append(str(years[i]) + " - " + str(years[i + 2]))
    ax1.bar(year_chunk, medal_count)
    ax1.set_title(e)
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Number Of Medals")
    indexval = indexval + 1
plt.tight_layout()
plt.show()


# ### The percent of times that a country has won at least one medal of any kind
# ##### If a country wins at least one medal of any kind, that is considered a "win", and is therefore counted toward the win percentage. This essentially calculates a win percentage for each country based on that metric.

# In[ ]:


final_dict = {}
total_games = 23
all_country = pd.unique(df['Country']).tolist()
#Universities that have won a medal for that year. (Gold, Silver, Bronze ONLY)
for e in all_country:
    start_year = 1999
    wins = 0
    for i in range(23):
        test_df = df.loc[df['Year'] == (start_year + i)]
        countries_in_one_year = test_df['Country'].tolist()
        all_gold = test_df['Gold'].tolist()
        all_sil = test_df['Silver'].tolist()
        all_bron = test_df['Bronze'].tolist()
        for idx, c in enumerate(countries_in_one_year):
            if (e == c):
                if (all_gold[idx] == 1):
                    wins += 1
                    break
                if (all_sil[idx] == 1):
                    wins += 1
                    break
                if (all_bron[idx] == 1):
                    wins += 1
                    break
    final_dict[e] = ((wins/total_games) * 100)
sorted_dict = dict(sorted(final_dict.items(), key = operator.itemgetter(1)))
for k, v in sorted_dict.items():
    print (str(k) + ' --> ' + "{0:.2f} %".format(v))


# ## Data Plotting

# ### Number of times that each country has won a World Champion prize
# ##### This shows how many times each country has won a World Champion prize in a bar graph in sorted order.

# In[ ]:


df['Prize'] = df['Prize'].fillna('')
world_count = df['Prize'].str.contains('World Champion').groupby(df['Country']).sum()
world_count_sort = world_count.sort_values()
world_count_sort.plot.bar(figsize=(14, 3), color = 'red')
plt.xlabel('Country')
plt.ylabel('Number of World Champion Prizes')
plt.title('Number of World Champion Prizes by Country')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()


# ### Average score for each country
# ##### This shows the total average score that was obtained by each country over the entire time period in a bar graph in sorted order.

# In[ ]:


world_scoreperc = df['Score %'].groupby(df['Country']).mean()
world_scoreperc_sort = world_scoreperc.sort_values()
world_scoreperc_sort.plot.bar(figsize=(14, 3), color = '#2E9962')
plt.xlabel('Country')
plt.ylabel('Avg Score')
plt.title('Average Score Per Country')
plt.legend(loc='upper left')
plt.grid()
plt.show()


# ### Amount of teams that entered each year
# ##### This shows the progression in the number of teams that entered the competition each year through the use of a bar graph. It shows that there was a consistent increase in the number of teams joining the competition.

# In[ ]:


team_count = df.groupby('Year')[['Team']].count()
team_count.plot.bar(figsize = (14, 3), color = '#900C3F')
plt.xlabel('Year')
plt.ylabel('Amount of entries')
plt.title('Number of teams that entered each year')
plt.legend(loc='upper left')
plt.grid()
plt.show()


# ### Amount of medals each country has won
# ##### This shows the total amount of medals that each country won across the entire time period in a bar graph. This excludes the honorable column.

# In[ ]:


country_total_medals = (df.groupby('Country')['Gold'].sum() + df.groupby('Country')['Silver'].sum() + df.groupby('Country')['Bronze'].sum())
country_total_medals_sort = country_total_medals.sort_values()
country_total_medals_sort.plot.bar(figsize = (14, 3), color = '#EA8D50')
plt.xlabel('Country')
plt.ylabel('Amount of medals')
plt.title('Number of medals each country has won')
plt.legend(['Number of Medals'], loc='upper left')
plt.grid()
plt.show()


# ### Average rank of universities
# ##### This shows the average rank of each university across the entire time period in a bar graph. It is shown in groups of 40 universities at a time.

# In[ ]:


school_avg_rank = (df.groupby('Uni')['Rank'].mean())
school_avg_rank = school_avg_rank.to_frame()
ranks = school_avg_rank['Rank']
j = 0
for i in range(len(school_avg_rank)):
    if j >= len(school_avg_rank):
        break
    tempDF = tempDF2 = pd.DataFrame(columns = ['Uni', 'Rank'])
    for j in range(j, j + 40):
        if j >= len(school_avg_rank):
            break
        row = school_avg_rank.iloc[j]
        tempDF2 = pd.DataFrame({'Uni': row.name, 'Rank': ranks[j]}, index = [0])
        tempDF = pd.concat([tempDF, tempDF2], ignore_index = True)
    tempDF = tempDF.sort_values('Rank')
    tempDF.plot.bar(figsize = (100, 20), color = '#5C3487')
    plt.xlabel('Uni')
    plt.ylabel('Average rank')
    plt.title('Average rank of each university')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xticks(tempDF.index, tempDF['Uni'])
    plt.show()
    j += 1


# ### Average rank of countries
# ##### This shows the average rank of each country across the entire time period in a bar graph. It is shown in groups of 40 countries at a time.

# In[ ]:


country_avg_rank = (df.groupby('Country')['Rank'].mean())
country_avg_rank = country_avg_rank.to_frame()
ranks = country_avg_rank['Rank']
j = 0
for i in range(len(country_avg_rank)):
    if j >= len(country_avg_rank):
        break
    tempDF = tempDF2 = pd.DataFrame(columns = ['Country', 'Rank'])
    for j in range(j, j + 40):
        if j >= len(country_avg_rank):
            break
        row = country_avg_rank.iloc[j]
        tempDF2 = pd.DataFrame({'Country': row.name, 'Rank': ranks[j]}, index = [0])
        tempDF = pd.concat([tempDF, tempDF2], ignore_index = True)
    tempDF.plot.bar(figsize = (100, 20), color = '#AF2A2A')
    plt.xlabel('Country')
    plt.ylabel('Average rank')
    plt.title('Average rank of each Country')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xticks(tempDF.index, tempDF['Country'])
    plt.show()
    j += 1


# ### Penalty of #1 ranked university versus the highest penalty from that year
# ##### This shows what the #1 ranked university's penalty was for a year as well as the highest penalty of that year in a scatterplot with lines drawn between the two datasets. This scatterplot shows that, generally, the #1 ranked university has a penalty close to that of the highest penalty each year, if not the highest penalty.

# In[ ]:


# Get a dataframe of the years and the penalties of each rank 1 uni
rank1_penaltyS = df.loc[df['Rank'] == 1, 'Penalty']
years = df['Year']
years = years.drop_duplicates()
rank1_penalty = pd.DataFrame({'Year': years, 'Penalty': rank1_penaltyS})

# Get the highest penalty of each year and add it to the first dataframe
highest_penaltyS = df['Penalty'].groupby(df['Year']).max()
highest_penaltyS = highest_penaltyS.reset_index(drop = True)
highest_penalty = highest_penaltyS.to_frame()

# Create scatterplot with both datasets
fig, axis = plt.subplots()
x = 1
axis.scatter(rank1_penalty['Year'], rank1_penalty['Penalty'])
axis.scatter(rank1_penalty['Year'], highest_penalty['Penalty'])
plt.xticks(rank1_penalty['Year'], rotation = 'vertical')
plt.plot(rank1_penalty['Year'], rank1_penalty['Penalty'])
plt.plot(rank1_penalty['Year'], highest_penalty['Penalty'])
plt.title('Penalty of Top Ranked University vs Highest Penalty Each Year')
plt.legend(['Penalty For Rank 1 University', 'Highest Penalty This Year'])
plt.show()


# ### Average score percentage compared to the median score percentage every 5 years
# ##### This shows the distinction between the average score percentage and the median score percentage in five year increments using a line graph.

# ### Average score percentage compared to the median score percentage every 5 years
# ##### This shows the distinction between the average score percentage and the median score percentage in five year increments using a line graph.

# In[ ]:


df_avgScore = df.groupby('Year').mean()['Score %']
df_avgScore.tail()
df_medianScore = df.groupby('Year').median()['Score %']
df_medianScore.tail()
index_avg = df_avgScore.index
index_median = df_medianScore.index
sns.set_style('whitegrid')
x1, y1 = index_avg, df_avgScore
x2, y2 = index_median, df_medianScore
plt.figure(figsize=(9, 4))
plt.plot(x1, y1, color = 'b', label = 'Average')
plt.plot(x2, y2, color = 'r', label = 'Median')
plt.title('Score % Over All Years')
plt.xlabel('Year')
plt.ylabel('Score %')
plt.legend(loc = 'upper left')
plt.show()


# ### Double bar graph for the amount of teams a country sends and the amount of medals won
# ##### This is useful because it shows if there is any correlation to amount of teams sent and medals won.

# In[ ]:


country_teams_stats_df = df[['Country', 'Team', 'Gold', 'Silver', 'Bronze']]
country_total_teams = (country_teams_stats_df.groupby('Country')['Team'].count())
country_total_medals = (df.groupby('Country')['Gold'].sum() + df.groupby('Country')['Silver'].sum() + df.groupby('Country')['Bronze'].sum())
countries = pd.unique(df['Country']).tolist()
sortedCountries = sorted(countries)
X = sortedCountries
X_axis = np.arange(len(X))
fig = plt.figure(figsize=(14,6))
plt.bar(X_axis, country_total_teams, 0.4, label = 'Total Teams')
plt.bar(X_axis, country_total_medals, 0.4, label = 'Total Medals')
plt.xticks(X_axis, X, rotation='vertical')
plt.xlabel("Countries")
plt.ylabel("Total")
plt.title("Number of Teams each Countries sends Vs How Many Medals each Country Won")
plt.legend()
plt.grid()
plt.show()


# ### Top ranked universities every 3 years
# ##### This graph is practical because it allows us to see the top performing universities over broad spans of time.

# In[ ]:


years = range(1999, 2021 + 1)

for i in range(0, len(years), 3):
    top10 = []
    ax1 = fig.add_subplot(76, 1, indexval)
    if (i + 2 == 23):
        test_df = df.loc[(df['Year'] == years[i]) | (df['Year'] == years[i + 1])]
    else:
        test_df = df.loc[(df['Year'] == years[i]) | (df['Year'] == years[i + 1]) | (df['Year'] == years[i + 2])]
    top10 = test_df.groupby('Uni')['Rank'].sum().sort_values().head(n=10)
    top10.plot.bar(figsize = (14, 3), color = '#EA8D50')
    plt.ylabel("Total rank within 3 years")
    plt.xlabel("University")
    plt.title(str(1999 + i) + " Through " + str(1999 + i + 3))
    plt.grid()

    plt.show()

