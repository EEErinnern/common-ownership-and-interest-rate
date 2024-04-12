#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

import statsmodels.formula.api as smf

import seaborn as sns
import pyhdfe

import matplotlib.pyplot as plt


# In[2]:


# merge 
df_bank = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_bank.parquet').to_pandas()
display(df_bank.head())

# bhcid is the rssd_id used for match with the link table!!!!


# In[3]:


# dataset with bank information including deposit rate and bank characteristics
df_deposit_sample = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_bank_deposit.parquet').to_pandas()
display(df_deposit_sample.head())


# In[4]:


df_kappa = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_kappa.parquet').to_pandas()
df_ihhi = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_ihhi.parquet').to_pandas()

display(df_kappa.head())


# In[99]:


sns.boxplot(after_ihhi['ihhi'])


# In[14]:


fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter( df_kappa['date'],df_kappa['kappa'])
ax.set_xlabel('(date)')
ax.set_ylabel('(ihhi)')
plt.show()


# In[5]:


from scipy.stats import zscore

# Calculate Z-scores for kappa
z_scores = zscore(df_kappa['kappa'])

# Define a threshold for identifying outliers (e.g., Z-score greater than 3 or less than -3)
threshold = 3.5

# Identify outliers
outliers = np.abs(z_scores) > threshold

# Remove outliers from the DataFrame
df_kappa = df_kappa[~outliers]

# Check the shape of the DataFrame after removing outliers
print("Original shape:", df_kappa.shape)
print("Shape after removing outliers:", df_kappa.shape)


# In[6]:


from scipy.stats import zscore

# Calculate Z-scores for kappa
z_scores = zscore(df_ihhi['ihhi'])

# Define a threshold for identifying outliers (e.g., Z-score greater than 3 or less than -3)
threshold = 3

# Identify outliers
outliers = np.abs(z_scores) > threshold

# Remove outliers from the DataFrame
df_ihhi = df_ihhi[~outliers]

# Check the shape of the DataFrame after removing outliers
print("Original shape:", df_ihhi.shape)
print("Shape after removing outliers:", df_ihhi.shape)


# In[13]:



display(df_ihhi.head())


# In[8]:


# descriptive figures of common ownership and ihhi
# code/plots2_kappa_official.py


# In[7]:


# # combine ownership data and bank data 
df_regression = pd.merge(
    df_kappa.loc[(df_kappa['firm_j'] != df_kappa['firm_k']) & (
        df_kappa['date'] <= '2020-01-01')],
    df_deposit_sample,
    left_on=['firm_j', 'year', 'quarter'],
    right_on=['permno', 'year', 'quarter'],
    how='left'
).reset_index(drop=True).dropna(subset = ['permno'])

df_regression['rssdid'] = df_regression['rssdid'].astype(int)
df_regression['permco'] = df_regression['permco'].astype(int)
df_regression['permno'] = df_regression['permno'].astype(int)
df_regression['rssd_id'] = df_regression['rssd_id'].astype(int)
df_regression['termlength'] = df_regression['termlength'].astype(int)
df_regression['cnty_fps'] = df_regression['cnty_fps'].astype(int)
df_regression['state_fps'] = df_regression['state_fps'].astype(int)

df_regression = df_regression.drop(columns = ['bhcid', 'dt_start','dateq','dt_end','month_x','branches'])

display(df_regression.head())


# In[211]:


print("Number of rows (Option 2):", df_regression.shape[0])
print("Number of unique rssdid values:", df_regression['firm_j'].unique())


# In[15]:


print("Number of unique rssdid values:", df_regression['firm_j'].nunique())
print("Number of unique rssdid values:", df_regression['firm_k'].nunique())


# In[33]:


filter_df = df_regression[(df_regression['rssd_id'] == 485559) & (df_regression['year'] == 2010) & (df_regression['month'] == 3 ) & (df_regression['termlength']==12)]
pd.set_option('display.max_columns', None)
display(filter_df)


# In[32]:


filter_df = filter_df.drop_duplicates(subset=['cnty_fps', 'state_fps','firm_j','firm_k','year','month','quarter'])
pd.set_option('display.max_columns', None)
display(filter_df)


# In[12]:


# controls 
df_controls = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_control_variable.parquet').to_pandas()

quarter_mapping = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
df_controls['quarter'] = df_controls['month'].map(quarter_mapping)

df_controls.rename(columns={'County FIPS Code': 'cnty_fps'}, inplace=True)
df_controls.rename(columns={'State FIPS Code': 'state_fps'}, inplace=True)
df_controls.rename(columns={'Median Household Income': 'median_income'}, inplace=True)
df_controls.rename(columns={'Postal Code': 'stcode'}, inplace=True)
df_controls.rename(columns={'Name': 'name'}, inplace=True)

df_controls = df_controls.drop_duplicates(subset=['cnty_fps', 'state_fps','population','median_income','dgs10'])

df_controls = df_controls[df_controls['median_income'] != 0]
df_controls = df_controls[df_controls['population'] != 0]
df_controls['log_income'] = np.where(df_controls['median_income'] > 0, np.log(df_controls['median_income']), np.nan)
df_controls['log_population'] = np.where(df_controls['population'] > 0, np.log(df_controls['population']), np.nan)

df_controls['fps'] = df_controls['cnty_fps'].astype(str) + df_controls['state_fps'].astype(str)

df_controls = df_controls.drop(columns = ['population','median_income','month','date'])

display(df_controls.head())


# In[13]:


# merge controls with the regression data
df_regression = pd.merge(df_regression, df_controls, left_on=['cnty_fps','state_fps', 'year','quarter'], right_on=['cnty_fps','state_fps','year','quarter'], how='left').reset_index(drop=True)

display(df_regression.head())


# In[14]:


# deposit rate: 12-, 24-, and 36-month CD rate spreads; 
df_cd12 = df_regression[(df_regression['termlength'] == 12)]
df_cd24 = df_regression[(df_regression['termlength'] == 24)]
df_cd36 = df_regression[(df_regression['termlength'] == 36)]

display(df_cd12.head())


# In[25]:


# the information of banks identified
banks = df_regression[['firm_j','rssdid','assets','deposits','name_y','inst_type']].drop_duplicates(subset=['rssdid'])
sorted_banks = banks.sort_values(by='assets', ascending=False)

largest_10_banks = sorted_banks.head(10)

display(largest_10_banks['firm_j'])


# In[22]:


# descriptive grapf of deposit rates 
# Group by month, year, and county to calculate the average deposit rate
avg_rate_12 = df_cd12.groupby(['year', 'quarter'])['rate'].mean().reset_index()
avg_rate_24 = df_cd24.groupby(['year', 'quarter'])['rate'].mean().reset_index()
avg_rate_36 = df_cd36.groupby(['year', 'quarter'])['rate'].mean().reset_index()
us_10_year_treasury = df_regression.groupby(['year', 'quarter'])['dgs10'].mean().reset_index()

avg_rate_12['date'] = avg_rate_12['year'].astype(str) + 'Q' + avg_rate_12['quarter'].astype(str)
avg_rate_24['date'] = avg_rate_24['year'].astype(str) + 'Q' + avg_rate_24['quarter'].astype(str)
avg_rate_36['date'] = avg_rate_36['year'].astype(str) + 'Q' + avg_rate_36['quarter'].astype(str)

us_10_year_treasury['date'] = us_10_year_treasury['year'].astype(str) + 'Q' + us_10_year_treasury['quarter'].astype(str)

plt.figure(figsize=(10, 6))

# Plot each average rate
plt.plot(avg_rate_12['date'], avg_rate_12['rate'], label='Avg Rate 12', marker='o', linestyle='-')
plt.plot(avg_rate_24['date'], avg_rate_24['rate'], label='Avg Rate 24', marker='o', linestyle='-')
plt.plot(avg_rate_36['date'], avg_rate_36['rate'], label='Avg Rate 36', marker='o', linestyle='-')
plt.plot(us_10_year_treasury['date'], us_10_year_treasury['dgs10'], label='US 10-Year Treasury', marker='o', linestyle='-')

# Add labels and title
plt.title('Average Rates')
plt.xlabel('Date')
plt.ylabel('Average Rate')
plt.xticks(avg_rate_12['date'][::3], rotation=45)  # Display every 3rd label and rotate for better readability
plt.legend()  # Add legend
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/ave_rate_graph.png')
plt.show()


# In[113]:


def weighted(x, cols):
    a1 = np.average(x[cols].values, weights=x['deposit_share'].values, axis=0)[0]
    return pd.Series({'kappa_mean': a1})

# Apply the weighted averages
y = df_cd12.groupby(['year','quarter']).apply(weighted, ["kappa"])

qtr_mean = pd.concat([df_cd12.groupby(['quarter']).mean(), y], axis=1)

df_cosine = df_cd12.groupby(
    ['year','quarter'])['cosine'].describe(
        percentiles=[
            0.05, 0.25, 0.5, 0.75, 0.95])

kappa_pct = df_cd12.groupby(
    ['year','quarter'])['kappa'].describe(
        percentiles=[
            0.05,
            0.25,
            0.5,
            0.75,
            0.95])

ihhi_pct = df_ihhi[~df_ihhi.ihhi.isnull()].groupby(['year','quarter'])['ihhi'].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/Kappa_graph.png')


# In[23]:


kappa_pct = df_cd12.groupby(
    ['year','quarter'])['kappa'].describe(
        percentiles=[
            0.05,
            0.25,
            0.5,
            0.75,
            0.95])

plt.clf()
kappa_pct[['95%', '75%', '50%', '25%', '5%']].plot(figsize=(20, 10))
plt.legend(['95th percentile',
            '75th percentile',
            '50th percentile',
            '25th percentile',
            '5th percentile'])
plt.ylabel(r"$\kappa$ Quantiles")
plt.xlabel("")
plt.ylim(0, 1.4)


# In[11]:


ihhi_pct = df_ihhi[~df_ihhi.ihhi.isnull()].groupby(['year','quarter'])['ihhi'].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

ihhi_pct[['95%', '75%', '50%', '25%', '5%']].plot(figsize=(20, 10))
plt.legend(['95th percentile',
            '75th percentile',
            '50th percentile',
            '25th percentile',
            '5th percentile'])
plt.ylabel("Investor HHI")
plt.xlabel("")
plt.ylim(0, 800)

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/ihhi_graph.png')


# In[11]:


df_allkappa = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_allkappa.parquet').to_pandas()

plt.figure(figsize=(20, 10))
df_kappa = df_kappa[df_kappa['firm_j'] != df_kappa['firm_k']]
df_allkappa = df_allkappa[df_allkappa['firm_j'] != df_allkappa['firm_k']]
                               
df_kappa.groupby(['year', 'quarter'])['kappa'].mean().plot(ax=plt.gca())
df_allkappa.groupby(['year', 'quarter'])['kappa'].mean().plot(ax=plt.gca())

# Plot the second line on the same graph

# Customize the plot
plt.title('Trend of common ownerhsip of banks and S&P500')
plt.xlabel("")  # No x-label
plt.ylim(0.2, 0.9)  # Set y-axis limits
plt.legend([r'$\kappa_{j,k}$ $ of $ $ Banks $', r'$\kappa_{j,k}$ $ of $ $ S&P 500$'], fontsize='large')  # Add legend
plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust layout

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/profit_weight_graph.png')

plt.show()


# In[23]:


df_allihhi = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_allihhi.parquet').to_pandas()

plt.figure(figsize=(20, 10))

df_allihhi.groupby(['year', 'quarter'])['ihhi'].mean().plot(ax=plt.gca())
df_ihhi.groupby(['year', 'quarter'])['ihhi'].mean().plot(ax=plt.gca())

# Customize the plot
plt.title('Trend of Investor HHI of banks and S&P500')
plt.xlabel("")  # No x-label
plt.ylim(100, 400)  # Set y-axis limits
plt.legend([r'$IHHI$ $ of $ $ S&P 500 $', r'$IHHI$ $ of $ $ Banks$'], fontsize='large')  # Add legend
plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust layout

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/ihhi_500_graph.png')

plt.show()


# In[59]:


plt.figure(figsize=(25, 12))

# Plot the first line
grouped = df_cd24.groupby(['year', 'quarter'])[['kappa', 'rate']].median()

# Customize the plot
fig, ax1 = plt.subplots(figsize=(25, 12))
ax2 = ax1.twinx()
grouped['rate'].plot(ax=ax1, color='purple', marker='s', label='rate')
grouped['kappa'].plot(ax=ax2, color='orange', marker='o', label='kappa')

# Set labels and title
ax1.set_ylabel('Rate')
ax2.set_ylabel('Profit Weight')
ax1.set_xlabel('Time Period')
plt.title('Profit weight and Interest rate Over Time')

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust layout

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/kappa_rate_graph.png')

plt.show()


# In[56]:


df_regression = df_regression[(df_regression['firm_j'] != df_regression['firm_k'])]
plt.figure(figsize=(20, 10))

df_kappa.groupby(['year','quarter'])[['kappa', 'cosine']].mean().plot(ax=plt.gca())
plt.xlabel("")
#plt.title("Cosine Similarity and $\kappa$")
plt.ylim(0.2, 0.9)
plt.legend([r'$\kappa_{j,k}$',
            r'$similarity$ $cos(\beta_j,\beta_k)$'
            ], fontsize='large')
plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust layout

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/kappa_cosine_graph.png')

plt.show()


# In[184]:


var_list = df_cd12[['cnty_fps','state_fps','year','quarter']]
var_list = var_list.drop_duplicates(subset = ['cnty_fps','state_fps','year','quarter'])
print(var_list.shape)


# In[188]:


print(df_cd_12[['firm_j','firm_k', 'year','cnty_fps','state_fps']].nunique())


# In[131]:


display(df_cd12)


# In[15]:


group_12 = df_cd12.groupby(['firm_j','year','quarter','cnty_fps','state_fps','fps'])['rate'].mean().reset_index()
df_cd12 = df_cd12.drop(columns = ['rate'])
df_cd_12 = pd.merge(df_cd12, group_12, on=['firm_j','year','quarter','cnty_fps','state_fps','fps'], how = 'inner').drop_duplicates(['firm_j','firm_k', 'year','quarter','cnty_fps','state_fps'])
df_cd_12 = df_cd_12[['firm_j', 'firm_k','year','quarter','cnty_fps','state_fps','fps','kappa', 'rate','deposit_share','log_income',
                     'log_population','bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']].dropna().drop_duplicates(['firm_j','firm_k', 'year','quarter','cnty_fps','state_fps'])
display(df_cd_12.head())


# In[205]:


# simple plot of average kappa and average deposit rate change over the years 


# In[16]:


group_24 = df_cd24.groupby(['firm_j','year','quarter','cnty_fps','state_fps','fps'])['rate'].mean().reset_index()
df_cd24 = df_cd24.drop(columns = ['rate'])
df_cd_24 = pd.merge(df_cd24, group_24, on=['firm_j','year','quarter','cnty_fps','state_fps','fps'], how = 'inner').drop_duplicates(['firm_j','firm_k', 'year','quarter','cnty_fps','state_fps'])
df_cd_24 = df_cd_24[['firm_j', 'firm_k','year','quarter','cnty_fps','state_fps','fps','kappa', 'rate','deposit_share','log_income',
                     'log_population','bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']].dropna().drop_duplicates(['firm_j','firm_k', 'year','quarter','cnty_fps','state_fps'])
display(df_cd_24.head())
print(df_cd_24.shape)


# In[17]:


group_36 = df_cd36.groupby(['firm_j','year','quarter','cnty_fps','state_fps','fps'])['rate'].mean().reset_index()
df_cd36 = df_cd36.drop(columns = ['rate'])
df_cd_36 = pd.merge(df_cd36, group_36, on=['firm_j','year','quarter','cnty_fps','state_fps','fps'], how = 'inner').drop_duplicates(['firm_j','firm_k', 'year','quarter','cnty_fps','state_fps'])
df_cd_36 = df_cd_36[['firm_j', 'firm_k','year','quarter','cnty_fps','state_fps','fps','kappa', 'rate','deposit_share','log_income',
                     'log_population','bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']].dropna().drop_duplicates(['firm_j','firm_k', 'year','quarter','cnty_fps','state_fps'])
display(df_cd_36.head())
print(df_cd_36.shape)


# In[18]:


# var lis
# var list
var_list = ['kappa', 'rate','deposit_share','log_income','log_population',
            'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']
controls = ['bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

# Branch and year fixed effects

df_cd_12['pair_fe'] = df_cd_12.groupby(['firm_j', 'firm_k']).ngroup()
df_cd_12['county_fe'] = df_cd_12.groupby(['fps']).ngroup()
df_cd_12['year_fe'] = df_cd_12.groupby(['year','quarter']).ngroup()

# Drop any missings
df_cd_12 = df_cd_12.dropna(subset=var_list + ['pair_fe', 'year_fe','county_fe'])
df_cd_12 = df_cd_12.drop_duplicates(subset=var_list + ['pair_fe', 'year_fe','county_fe'])

# Create the entity and time indices
alg_pa = pyhdfe.create(df_cd_12[['pair_fe', 'year_fe','county_fe']].values, drop_singletons=False)
resid_pa = alg_pa.residualize(df_cd_12[var_list].values)

re_vars = pd.DataFrame(resid_pa, columns=['kappa', 'rate','deposit_share','log_income','log_population',
                                          'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income'])
# Regression
reg_1 = smf.ols(
    formula = f'rate ~ kappa + log_population + log_income + {" + ".join(controls)}', 
    data=re_vars).fit(cov_type='cluster', cov_kwds={'groups': [df_cd_12['fps']]})

# Print regression summary
print(reg_1.summary())


# In[64]:


# var list
var_list = ['kappa', 'rate','deposit_share','log_income','log_population',
            'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']
controls = ['bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

# Branch and year fixed effects

df_cd_24['pair_fe'] = df_cd_24.groupby(['firm_j', 'firm_k']).ngroup()
df_cd_24['county_fe'] = df_cd_24.groupby(['cnty_fps', 'state_fps']).ngroup()
df_cd_24['year_fe'] = df_cd_24.groupby(['year','quarter']).ngroup()

# Drop any missings
df_cd_24 = df_cd_24.dropna(subset=var_list + ['pair_fe', 'year_fe','county_fe'])
df_cd_24 = df_cd_24.drop_duplicates(subset=var_list + ['pair_fe', 'year_fe','county_fe'])

# Create the entity and time indices
alg_pa = pyhdfe.create(df_cd_24[['pair_fe', 'year_fe','county_fe']].values, drop_singletons=False)
resid_pa = alg_pa.residualize(df_cd_24[var_list].values)

re_vars = pd.DataFrame(resid_pa, columns=['kappa', 'rate','deposit_share','log_income','log_population',
                                          'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income'])
# Regression
reg_2 = smf.ols(
    formula = f'rate ~ kappa + deposit_share + log_population + log_income + {" + ".join(controls)}', 
    data=re_vars).fit(cov_type='cluster', cov_kwds={'groups': [df_cd_24['fps']]})

# Print regression summary
print(reg_2.summary())


# In[62]:


var_list = ['kappa', 'rate','deposit_share','log_income','log_population',
            'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']
controls = ['bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

# Branch and year fixed effects

df_cd_36['pair_fe'] = df_cd_36.groupby(['firm_j', 'firm_k']).ngroup()
df_cd_36['year_fe'] = df_cd_36.groupby(['year','quarter']).ngroup()
df_cd_36['county_fe'] = df_cd_36.groupby(['cnty_fps', 'state_fps']).ngroup()

# Drop any missings
df_cd_36 = df_cd_36.dropna(subset=var_list + ['pair_fe', 'year_fe','county_fe'])
df_cd_36 = df_cd_36.drop_duplicates(subset=var_list + ['year','pair_fe', 'year_fe','county_fe'])


# Create the entity and time indices
alg_pa = pyhdfe.create(df_cd_36[['pair_fe', 'year_fe','county_fe']].values, drop_singletons=False)
resid_pa = alg_pa.residualize(df_cd_36[var_list].values)

re_vars = pd.DataFrame(resid_pa, columns=['kappa', 'rate','deposit_share','log_income','log_population',
                                          'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income'])
# Regression
reg_3 = smf.ols(
    formula = f'rate ~ kappa + deposit_share + log_population + log_income + {" + ".join(controls)}', 
    data=re_vars).fit(cov_type='cluster', cov_kwds={'groups': [df_cd_36['fps']]})

# Print regression summary
print(reg_3.summary())


# In[20]:


# cross-sectional regression of year 2018
cd12_2018 = df_cd_12[df_cd_12['year'] == 2018]
cd24_2018 = df_cd_24[df_cd_24['year'] == 2018]
cd36_2018 = df_cd_36[df_cd_36['year'] == 2018]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter( cd24_2018['kappa'],cd24_2018['spread'])
ax.set_xlabel('(kappa)')
ax.set_ylabel('(spread)')
plt.show()


# In[21]:


display(cd24_2018.head())


# In[82]:


# var list
var_list = ['kappa', 'spread','deposit_share','log_income','log_population',
            'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']
controls = ['bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

# Branch and year fixed effects

df_cd12['pair_fe'] = df_cd12.groupby(['firm_j', 'firm_k']).ngroup()
df_cd12['year_fe'] = df_cd12.groupby(['year']).ngroup()

# Drop any missings
df_cd12 = df_cd12.dropna(subset=var_list + ['pair_fe', 'year_fe'])
df_cd12 = df_cd12.drop_duplicates(subset=var_list + ['pair_fe', 'year_fe'])

# Create the entity and time indices
alg_pa = pyhdfe.create(df_cd12[['pair_fe', 'year_fe']].values, drop_singletons=False)
resid_pa = alg_pa.residualize(df_cd12[var_list].values)

re_vars = pd.DataFrame(resid_pa, columns=['kappa', 'spread','deposit_share','log_income','log_population',
                                          'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income'])
# Regression
reg1 = smf.ols(
    formula = f'spread ~ kappa + deposit_share + log_population + log_income + {" + ".join(controls)}', 
    data=re_vars).fit(cov_type='HC3')

# Print regression summary
print(reg1.summary())


# In[24]:


# simple correlations between profit weight kappa and bank deposit rate

# controls: market characteristics such as log median household income and log population
# Banks’market share: a proxy for differences across banks in the level and and changes over time in variable costs

# var list
var_list = ['kappa', 'spread','deposit_share','log_income','log_population',
            'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']
controls = ['bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

# Branch and year fixed effects
df_cd24['pair_fe'] = df_cd24.groupby(['firm_j', 'firm_k']).ngroup()
df_cd24['quarter_fe'] = df_cd24.groupby(['year','quarter']).ngroup()

# Drop any missings
df_cd24 = df_cd24.dropna(subset=var_list + ['pair_fe', 'quarter_fe'])
df_cd24 = df_cd24.drop_duplicates(subset=var_list + ['pair_fe', 'quarter_fe'])

# Create the entity and time indices
alg_pa = pyhdfe.create(df_cd24[['pair_fe', 'quarter_fe']].values, drop_singletons=False)
resid_pa = alg_pa.residualize(df_cd24[var_list].values)

re_vars = pd.DataFrame(resid_pa, columns=['kappa', 'spread','deposit_share','log_income','log_population',
                                          'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income'])
# Regression
reg2 = smf.ols(
    formula = f'spread ~ kappa + deposit_share + log_population + log_income + {" + ".join(controls)}', 
    data=re_vars).fit(cov_type='HC3')

# Print regression summary
print(reg2.summary())


# In[182]:


# Plot residuals
sns.set(style="whitegrid")
sm.graphics.plot_regress_exog(reg_1, "kappa", fig=plt.figure(figsize=(12, 8)))

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/residual.png')
plt.show()


# In[23]:


controls = ['bank_size','loan_to_asset','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

df_cd24['pair_fe'] = df_cd24.groupby(['firm_j', 'firm_k']).ngroup()
df_cd24['year_fe'] = df_cd24.groupby(['year']).ngroup()

# Drop any missings
df_cd24 = df_cd24.dropna(subset=var_list + ['pair_fe', 'year_fe'])

# Regression with fixed effects using C() to specify categorical variables
reg3 = smf.ols(
    formula=f'spread ~ kappa + deposit_share + log_population + log_income + {" + ".join(controls)}',
    data=df_cd24
).fit(cov_type='HC3')

# Print regression summary
print(reg2.summary())


# In[48]:


# controls: market characteristics such as log median household income and log population
# Banks’market share: a proxy for differences across banks in the level and and changes over time in variable costs

# var list
var_list = ['kappa', 'spread','deposit_share','log_income','log_population',
            'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']
controls = ['bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

# Branch and year fixed effects
df_cd36['pair_fe'] = df_cd36.groupby(['firm_j', 'firm_k']).ngroup()
df_cd36['year_fe'] = df_cd36.groupby(['year']).ngroup()

# Drop any missings
df_cd36 = df_cd36.dropna(subset=var_list + ['pair_fe', 'year_fe'])
df_cd36 = df_cd36.drop_duplicates(subset=var_list + ['pair_fe', 'year_fe'])

# Create the entity and time indices
alg_pa = pyhdfe.create(df_cd36[['pair_fe', 'year_fe']].values, drop_singletons=False)
resid_pa = alg_pa.residualize(df_cd36[var_list].values)

re_vars = pd.DataFrame(resid_pa, columns=['kappa', 'spread','deposit_share','log_income','log_population',
                                          'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income'])
# Regression
reg3 = smf.ols(
    formula = f'spread ~ kappa + deposit_share + log_population + log_income + {" + ".join(controls)}', 
    data=re_vars).fit(cov_type='HC3')

# Print regression summary
print(reg3.summary())


# In[33]:


controls = ['bank_size','loan_to_asset','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']

df_cd36['pair_fe'] = df_cd36.groupby(['firm_j', 'firm_k']).ngroup()
df_cd36['year_fe'] = df_cd36.groupby(['year']).ngroup()

# Drop any missings
df_cd36 = df_cd36.dropna(subset=var_list + ['pair_fe', 'year_fe'])

# Regression with fixed effects using C() to specify categorical variables
reg2 = smf.ols(
    formula=f'spread ~ kappa + deposit_share + log_population + log_income + {" + ".join(controls)}',
    data=df_cd36
).fit(cov_type='HC3')

# Print regression summary
print(reg2.summary())


# In[50]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming 'df' is your DataFrame with the independent variables
X = df_cd24[['kappa', 'spread','deposit_share','log_income','log_population',
            'bank_size','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']]
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# In[127]:


correlation_matrix = df_cd36[['kappa', 'spread','deposit_share','log_income','log_population',
            'bank_size','loan_to_asset','credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']].corr()
display(correlation_matrix)


# In[65]:


import re
from statsmodels.iolib.summary2 import summary_col

info_dict = {'Year FE': lambda x: "\u2713" if getattr(x, 'year_fe', False) else "",
             'Pair FE': lambda x: "\u2713" if getattr(x, 'pair_fe', False) else "",
             'Controls': lambda x: "\u2713"  if len(x.model.exog_names) > 1 else "",
             'R\sq': lambda x: f"{x.rsquared:.4f}",
             'N': lambda x: f"{int(x.nobs):d}",
             'F-statistic': lambda x: f"{x.fvalue:.4f}" if hasattr(x, 'fvalue') else ""
             }

dfoutput = summary_col(results=[reg_1, reg_2, reg_3],
                       float_format='%0.4f',
                       stars=True,
                       model_names=['12-Month CD rate',
                                    '24-Month CD rate',
                                    '36-Month CD rate'],
                       info_dict=info_dict,
                       regressor_order=['kappa',
                                        'deposit_share',
                                        'log_population',
                                        'log_income',
                                        ],
                       drop_omitted=True)

# Clean up the TeX by hand for the table
tab_reg2 = re.sub(r'\*\*\*', '*', dfoutput.as_latex())

# Display table and save
print(tab_reg2)


# In[37]:


# Check for NA values in the specified column
# Assuming df_cd12 is your DataFrame
missing_values_count = df_bank['loans'].isna().sum()

print(f"Number of missing values in 'loan': {missing_values_count}")
print(df_bank['loans'].shape)


# In[58]:


# Subset each dataset
subset_cd_12 = df_cd_12['rate']
subset_cd_24 = df_cd_24['rate']
subset_cd_36 = df_cd_36['rate']
subset_regression = df_regression[['kappa', 'deposit_share', 'log_income', 'log_population','bank_size', 'credit_risk', 'deposit_to_asset', 'roa', 'equity_to_income']]

# Concatenate or merge the subsets
combined_data = pd.concat([subset_cd_12, subset_cd_24, subset_cd_36, subset_regression], axis=1)
combined_data = combined_data.drop_duplicates()
# Generate descriptive statistics for the combined dataset
desc_stats = combined_data.describe().transpose().round(4)
desc_stats['count'] = desc_stats['count'].astype(int)
# Display the descriptive statistics
print(desc_stats)


# In[59]:


desc_stats_latex = desc_stats.to_latex()
print(desc_stats_latex)


# In[193]:


variable_names = ['kappa', 'spread', 'deposit_share','log_income','log_population','bank_size','credit_risk','deposit_to_asset','roa','equity_to_income']

# Assuming 'Definition 1', 'Definition 2', etc. are your variable definitions
variable_definitions = ['common ownership measurement: the profit weights is that each firm would place on the other', 
                        'the difference between the 10-year Treasury Constant Maturity average rate for each year and the deposit interest rates as a fraction of the Treasury rate',
                        'the deposit market share of each branch',
                       'log median household income',
                       'log population for each county',
                       'The natural logarithm of total assets, bank size',
                       'The ratio of loan loss provisions to interest income',
                       'The ratio of total deposits to total assets, liquidity risk/funding structure',
                       'Net income relative to total assets, profitability',
                       'Equity-to-asset ratio, capital buffer/leverage']

# Create a DataFrame with variable names and definitions
variable_table = pd.DataFrame({'Variable': variable_names, 'Definition': variable_definitions})

# Export to LaTeX
variable_table_latex = variable_table.to_latex(index=False, column_format='p{10cm}p{20cm}')
print(variable_table_latex)


# In[17]:


pip install geopandas matplotlib descartes


# In[18]:


# spatial distribution of kappa
import geopandas as gpd
import matplotlib.pyplot as plt

# Load US counties shapefile
us_counties = gpd.read_file("E:/Manh/5th semester/thesis/deposit_thesis/USA_Counties.zip")
df_controls.rename(columns={'CNTY_FIPS	': 'cnty_fps'}, inplace=True)
df_controls.rename(columns={'STATE_FIPS': 'state_fps'}, inplace=True)


# In[19]:


display(us_counties)


# In[ ]:


# Merge bank data with US counties GeoDataFrame
merged_data = us_counties.merge(df_cd_24, how="left", on="common_identifier")

# Plot US counties map
ax = us_counties.plot(figsize=(10, 10), edgecolor="black", color="white")

# Plot bank data on the map
merged_data.plot(ax=ax, color="blue", markersize=5)

# Customize plot
plt.title("Bank Information by US County")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

