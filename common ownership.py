#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import requests


# In[2]:


def matlab_sparse(i,j,s,compress=True):
    rows, row_pos = np.unique(i, return_inverse=True)
    cols, col_pos = np.unique(j, return_inverse=True)
    pivoted_arr = np.zeros((len(rows), len(cols)))
    pivoted_arr[row_pos, col_pos] = s
    if compress:
    	nz=(pivoted_arr.max(axis=1)>0)
    	pivoted_arr=pivoted_arr[nz,:]
    	rows=rows[nz]
    return pivoted_arr, rows, cols


# In[3]:


report_path = 'E:/Manh/5th semester/thesis/deposit_thesis/call_report.parquet'
ownership_path = 'E:/Manh/5th semester/thesis/deposit_thesis/common_ownership.parquet'
sample_path = 'E:/Manh/5th semester/thesis/deposit_thesis/deposit_sample.parquet'
link_id_path = 'E:/Manh/5th semester/thesis/deposit_thesis/permco_rssdid_2023.csv'
population_path = 'E:/Manh/5th semester/thesis/deposit_thesis/population_county.csv'

permno_co_path = 'E:/Manh/5th semester/thesis/deposit_thesis/permno_permco_link.parquet'


# In[4]:


income_0316 = 'E:/Manh/5th semester/thesis/deposit_thesis/median_income_us/median_income_2003_2016.xlsx'
income_17 = 'E:/Manh/5th semester/thesis/deposit_thesis/median_income_us/median_income_2017.xls'
income_18 = 'E:/Manh/5th semester/thesis/deposit_thesis/median_income_us/median_income_2018.xls'
income_19 = 'E:/Manh/5th semester/thesis/deposit_thesis/median_income_us/median_income_2019.xls'


# In[5]:


data_income0316 = pd.read_excel(income_0316)
df_income0316 = pd.DataFrame(data_income0316)

df_income0316['year'] = df_income0316['Year'].dt.year
df_income0316 = df_income0316.drop(columns = ['Year'])

df_income0316 =df_income0316 = df_income0316[df_income0316['County FIPS Code'] != 0]
df_income0316 = df_income0316.dropna(subset = ['Median Household Income'])
df_income0316['Median Household Income'] = df_income0316['Median Household Income'].astype(int)

display(df_income0316.head())


# In[5]:


print(df_income0316['year'].unique())


# In[7]:


data_income17 = pd.read_excel(income_17)
df_income17 = pd.DataFrame(data_income17)

data_income18 = pd.read_excel(income_18)
df_income18 = pd.DataFrame(data_income18)

data_income19 = pd.read_excel(income_19)
df_income19 = pd.DataFrame(data_income19)

df_income17 = df_income17.assign(year=2017)
df_income18 = df_income18.assign(year=2018)
df_income19 = df_income19.assign(year=2019)

display(df_income17.head())


# In[8]:


df_combined = df_income18.append(df_income17, ignore_index=True)
df_combined = df_combined.append(df_income19, ignore_index=True)
df_income = df_combined.append(df_income0316, ignore_index=True).dropna(subset = ['Median Household Income'])


# In[9]:


df_income = df_income[df_income['County FIPS Code'] != 0]
df_income = df_income[df_income['Median Household Income'] != '.']
df_income['Median Household Income'] = df_income['Median Household Income'].astype(int)


# In[10]:


table_income = pa.Table.from_pandas(df_income)
pa.parquet.write_table(table_income, 'E:/Manh/5th semester/thesis/deposit_thesis/df_income.parquet')


# In[12]:


# population

pop_2000_2010 = 'E:/Manh/5th semester/thesis/deposit_thesis/population_2000_2010.csv'
pop_2010_2019 = 'E:/Manh/5th semester/thesis/deposit_thesis/population_2010_2019.csv'

data_pop_10 = pd.read_csv(pop_2000_2010,encoding='latin-1')
df_pop10 = pd.DataFrame(data_pop_10)

data_pop_19 = pd.read_csv(pop_2010_2019,encoding='latin-1')
df_pop19 = pd.DataFrame(data_pop_19)

merged_pop = pd.merge(df_pop10, data_pop_19, on=['STATE', 'COUNTY'], how='inner').dropna()
merged_pop = merged_pop.drop(columns = ['POPESTIMATE2010','STNAME_y','CTYNAME_y','CENSUS2010POP_y'])

merged_pop.rename(columns={'STATE': 'State FIPS Code'}, inplace=True)
merged_pop.rename(columns={'COUNTY': 'County FIPS Code'}, inplace=True)
merged_pop.rename(columns={'STNAME_x': 'Postal Code'}, inplace=True)
merged_pop.rename(columns={'CTYNAME_x': 'Name'}, inplace=True)

merged_pop.rename(columns={'POPESTIMATE2000': '2000'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2001': '2001'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2002': '2002'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2003': '2003'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2004': '2004'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2005': '2005'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2006': '2006'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2007': '2007'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2008': '2008'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2009': '2009'}, inplace=True)
merged_pop.rename(columns={'CENSUS2010POP_x': '2010'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2011': '2011'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2012': '2012'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2013': '2013'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2014': '2014'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2015': '2015'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2016': '2016'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2017': '2017'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2018': '2018'}, inplace=True)
merged_pop.rename(columns={'POPESTIMATE2019': '2019'}, inplace=True)


# In[19]:


display(merged_pop.head())


# In[13]:


# Assuming 'Country' is your identifier column and the years are columns from 2000 to 2019
df_pop = pd.melt(merged_pop, id_vars=['State FIPS Code','County FIPS Code'], var_name='year', value_name='population')

# Convert 'Year' column to numeric
df_pop['year'] = pd.to_numeric(df_pop['year'], errors='coerce')

# Drop rows with NaN in the 'Population' column
df_pop = df_pop.dropna(subset=['population'])

# Sort the DataFrame by 'Country' and 'Year'
df_pop = df_pop.sort_values(by=['State FIPS Code','County FIPS Code', 'year'])

# Resetting the index
df_pop = df_pop.reset_index(drop=True).dropna(subset = ['year'])
df_pop['year'] = df_pop['year'].astype(int)

# Display the reshaped DataFrame
display(df_pop)


# In[21]:


print(df_pop['year'].unique())


# In[14]:


table_pop = pa.Table.from_pandas(df_pop)
pa.parquet.write_table(table_pop, 'E:/Manh/5th semester/thesis/deposit_thesis/df_population.parquet')


# In[15]:


df_pop_income = pd.merge(df_pop, df_income, on=['State FIPS Code', 'County FIPS Code','year'], how='inner').dropna()
display(df_pop_income)


# In[30]:


print(df_pop_income['year'].unique())


# In[103]:


display(df_pop_income[df_pop_income['County FIPS Code'] ==840 ])


# In[4]:


# link table of three identifiers: rssdid, permco and permno
df_link = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/permco_rssdid_link.parquet').to_pandas()
df_link = df_link.drop_duplicates(subset=['permno', 'rssdid','permco'])

df_link['dt_start'] = pd.to_datetime(df_link['dt_start'], format='%Y%m%d')
df_link['dt_end'] = pd.to_datetime(df_link['dt_end'], format='%Y%m%d')


# Drop rows where end date is before 2003
df_link = df_link[df_link['dt_end'] >= '2003-01-01']

display(df_link.head())


# In[6]:



df_report = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/call_report.parquet').to_pandas()
df_report['rssdid'] = df_report['rssdid'].astype(int)

display(df_report.head(10))

# bhcid is the rssd_id used for match with the link table!!!!


# In[4]:


print("Number of values with bhcid equals to 0:", df_report[df_report['bhcid'] == 0].shape[0])


# In[19]:


data_sample = pd.read_stata(sample_path)
table_sample = pa.Table.from_pandas(data_sample)
pa.parquet.write_table(table_sample, 'E:/Manh/5th semester/thesis/deposit_thesis/deposit_sample.parquet')


# In[4]:


df_sample = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/deposit_sample.parquet').to_pandas().dropna(subset=['rssd_id'])
df_sample['rssd_id'] = df_sample['rssd_id'].astype(int)

display(df_sample.head())


# In[7]:


# market share of each branch of banks

# Group by Quarter and Bank, calculate deposits share of each branch
df_report = df_report[df_report['bhcid'] != 0].dropna(subset=['bhcid'])
df_report = df_report[df_report['rssdid'] != 0].dropna(subset=['rssdid'])
df_report['bhcid'] = df_report['bhcid'].astype(int)

grouped_df_report = df_report.groupby(['year','month','dateq', 'name','rssdid','bhcid'])[['deposits']].sum().reset_index()
grouped_df_report['deposit_share'] = grouped_df_report['deposits'] / grouped_df_report.groupby('dateq')['deposits'].transform('sum') * 100

grouped_df_report = grouped_df_report[grouped_df_report['year'] >= 2003]
grouped_df_report = grouped_df_report[['bhcid','deposit_share','dateq']]

display(grouped_df_report.head())
# identification of the banks: rssdid


# In[8]:


# call report: equity, debt, EBIT, loans, ROA
# rssdid,name,equity,loans,liabilities,deposits,netinc,year,month,quarter,dateq
df_cr = df_report[['rssdid','bhcid', 'name','assets', 'equity', 'loans','loanleaselossprovision','liabilities', 'deposits', 'netinc','intincnet', 'year', 'month', 'quarter', 'dateq']].copy()

df_cr['year'] = df_cr['year'].astype(int)
df_cr['month'] = df_cr['month'].astype(int)
df_cr['quarter'] = df_cr['quarter'].astype(int)

df_cr = df_cr[df_cr['year'] >= 2003]

# bank size: logarithm of assets
df_cr = df_cr[df_cr['assets'] != 0]
df_cr['bank_size'] = np.where(df_cr['assets'] > 0, np.log(df_cr['assets']), np.nan)
df_cr = df_cr[df_cr['bank_size'].notnull()]
df_cr = df_cr[df_cr['bank_size'] != 0]


# asset composition: ration of loans to assets
df_cr['loan_to_asset'] = df_cr['loans']/df_cr['assets']
# credit risk: the ration of loan loss provisions to interest income
df_cr['credit_risk'] = df_cr['loanleaselossprovision']/df_cr['intincnet']
# liquidity risk/funding structure: ratio of deposit to assets
df_cr['deposit_to_asset'] = df_cr['deposits']/df_cr['assets']
# ROA: profitability,net income to assets
df_cr['roa'] = df_cr['netinc']/df_cr['assets']
# Capital buffer/leverage: equity to asset ratio
df_cr['equity_to_income'] = df_cr['equity']/df_cr['assets']

display(df_cr.head())


# In[9]:


merged_df = pd.merge(df_cr, grouped_df_report, on=['bhcid', 'dateq'], how='left')
merged_df.rename(columns={'rssdid': 'rssd_id'}, inplace=True)
merged_df = merged_df.drop_duplicates(subset=['bhcid', 'rssd_id','dateq'])


# In[97]:


display(merged_df.head())


# In[10]:


# Merge the link identifier to the bank characteristics
df_bank_factor = pd.merge(merged_df, df_link, left_on=['bhcid'], right_on=['rssdid'], how='left').dropna(subset =['rssdid'])

df_bank_factor['rssdid'] = df_bank_factor['rssdid'].astype(int, errors='ignore')
df_bank_factor['permco'] = df_bank_factor['permco'].astype(int, errors='ignore')
df_bank_factor['permno'] = df_bank_factor['permno'].astype(int, errors='ignore')

display(df_bank_factor.head())


# In[99]:


# bank branches characteristics
table_bank = pa.Table.from_pandas(df_bank_factor)
pa.parquet.write_table(table_bank, 'E:/Manh/5th semester/thesis/deposit_thesis/df_bank.parquet')


# In[11]:


df_bank = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/df_bank.parquet').to_pandas()

display(df_bank.head(10))


# In[4]:


# descriptive statistics of deposit rate: 12-, 24-, and 36-month CD rate spreads; 
# cross-sectional variation: map of the average interest rates in each county in 2013 for 12-month CDs and money market accounts

# time-variation: quarterly median, 20th percentile, and 80th percentile county-level average interest rates for 12-month CDs


# In[5]:


df_ownership = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/common_ownership.parquet').to_pandas()
df_ownership = df_ownership.drop_duplicates(subset=['permno', 'ownercik','rdate','fdate'])
df_ownership = df_ownership[df_ownership['permno'] != 0]
display(df_ownership.head(-10))


# In[9]:


print(df_bank['bhcid'].nunique())


# In[6]:


merged_ownership = pd.merge(df_ownership, df_link, on=['permno'], how='left').dropna(subset = ['permco'])

merged_ownership['rssdid'] = merged_ownership['rssdid'].astype(int)
merged_ownership['permco'] = merged_ownership['permco'].astype(int)

display(merged_ownership.head())


# In[17]:


owner_ids = [47896, 59408, 70519, 38703, 66157, 68144, 64995, 35554, 60442, 71563]

# Filter the DataFrame based on owner IDs
filtered_owners = merged_ownership[merged_ownership['permno'].isin(owner_ids)]
filtered_owners.loc[:, 'year'] = filtered_owners['rdate'].dt.year
data_2018 = filtered_owners[filtered_owners['year'] == 2018]

grouped_data = data_2018.groupby('permno')

# Step 3-4: Find the largest owner for each ID based on beta value
largest_owners = grouped_data.apply(lambda x: x.loc[x['beta'].idxmax()])
largest_owners = largest_owners.drop(columns = ['cusip','fdate','filed_13f','filed_f345','filed_13d','filed_13g','source','rssdid','permco','dt_start','dt_end','year'])

display(largest_owners)


# In[20]:


latex_table = largest_owners.to_latex(index=False)
print(latex_table)


# In[12]:


ownercik_id = [102909, 1364742]
filtered_owner = merged_ownership[merged_ownership['ownercik'].isin(ownercik_id)]
filtered_owner['year'] = filtered_owner['rdate'].dt.year

grouped_data = filtered_owner.groupby(['ownercik', 'ownername','rdate'])['beta'].mean().reset_index()

data_102909 = grouped_data[grouped_data['ownercik'] == 102909]
data_1364742 = grouped_data[grouped_data['ownercik'] == 1364742]

# Plot the average beta for each ownercik ID
plt.figure(figsize=(10, 6))

plt.plot(data_102909['rdate'], data_102909['beta'], label='Vanguard')
plt.plot(data_1364742['rdate'], data_1364742['beta'], label='Blackrock')

plt.title('Average Share of Banks ')
plt.xlabel('Year')
plt.ylabel('Average Ownership Percentage')
plt.legend()
plt.grid(True)
plt.tight_layout() 

plt.savefig('E:/Manh/5th semester/thesis/deposit_thesis/avg_share_graph.png')
plt.show()


# In[ ]:





# In[6]:


merged_ownership = merged_ownership[merged_ownership['beta'] != 0]
merged_ownership = merged_ownership[merged_ownership['beta'] <= 0.5]


# In[7]:


display(merged_ownership['rdate'].unique())


# In[25]:


# calculate measurement of common ownership: profit weights kappa 
# Betas: ownership shares of each bank: assume S shareholders and F banks,then the beta matrix should be 


def beta_to_kappa(merged_ownership):
    merged_ownership = merged_ownership[(merged_ownership.rdate >= '1980-01-01')]
    total_ownership = merged_ownership.groupby(['rdate']).apply(do_one_period)
    
    # merge and clean up missings
    total_ownership[['kappa', 'cosine']] = total_ownership[['kappa','cosine']].fillna(0)
    return total_ownership

# profit weights calculation
# Input beta: S x F matrix
# Output kappa: F x F matrix
def do_one_period(merged_ownership):
    [betas, owner_keys, permno_keys] = matlab_sparse(
        merged_ownership.ownercik, merged_ownership.permno, merged_ownership.beta)
    kappa = calc_kappa(betas)
    cosine = cosine_similarity(betas.transpose())
  
    idx = kappa.nonzero()
    return pd.DataFrame({
         'date': merged_ownership['rdate'].iloc[0],  # Assuming 'rdate' is constant within each group
         'firm_j': permno_keys[idx[0]], 'firm_k': permno_keys[idx[1]], 'kappa': kappa[idx].flatten(),'cosine': cosine[idx].flatten()})

def calc_ihhi(merged_ownership):
    merged_ownership['inv_total'] = merged_ownership.groupby(['permno', 'rdate'])['beta'].transform(sum)
    y = merged_ownership[['permno', 'rdate','beta', 'inv_total']].groupby(['permno', 'rdate']).apply(agg_ihhi)
    return y.reset_index()

def agg_ihhi(y):
    beta = y['beta']
    gamma = (beta**1)
    adj = 10000 * ((beta.sum() / gamma.sum())**2)
    ihhi = (gamma**2).sum() * adj
    return pd.Series({'date': y['rdate'].iloc[0], 'ihhi': ihhi})

# proportional control
def calc_kappa(betas, gamma_type='default'):
    gamma = betas  # /betas.sum(axis=0)
    return raw_kappa(betas, gamma)

# the ratio of inner products for kappas
def raw_kappa(betas, gamma):
    gamma = betas 
    # F x F matrix
    numer = gamma.T @ betas
    # F x 1 vector
    denom = np.diag(numer)
    # this is a F x F matrix
    return numer / denom[:, None]


# In[11]:


df_kappa = beta_to_kappa(merged_ownership)
df_ihhi = calc_ihhi(merged_ownership)


# In[28]:


df_all_kappa = beta_to_kappa(df_ownership)
df_all_ihhi = calc_ihhi(df_ownership)


# In[29]:


quarter_mapping = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
df_all_kappa['year'] = df_all_kappa['date'].dt.year
df_all_kappa['month'] = df_all_kappa['date'].dt.month
df_all_kappa['quarter'] = df_all_kappa['month'].map(quarter_mapping)

table_allkappa = pa.Table.from_pandas(df_all_kappa)

df_all_ihhi['year'] = df_all_ihhi['date'].dt.year
df_all_ihhi['month'] = df_all_ihhi['date'].dt.month
df_all_ihhi['quarter'] = df_all_ihhi['month'].map(quarter_mapping)

table_allihhi = pa.Table.from_pandas(df_all_ihhi)

pa.parquet.write_table(table_allkappa, 'E:/Manh/5th semester/thesis/deposit_thesis/df_allkappa.parquet')
pa.parquet.write_table(table_allihhi, 'E:/Manh/5th semester/thesis/deposit_thesis/df_allihhi.parquet')


# In[112]:


quarter_mapping = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
df_kappa['year'] = df_kappa['date'].dt.year
df_kappa['month'] = df_kappa['date'].dt.month
df_kappa['quarter'] = df_kappa['month'].map(quarter_mapping)


df_ihhi['year'] = df_ihhi['date'].dt.year
df_ihhi['month'] = df_ihhi['date'].dt.month
df_ihhi['quarter'] = df_ihhi['month'].map(quarter_mapping)
df_ihhi = df_ihhi.drop(columns = ['rdate'])


# In[113]:



display(df_kappa['date'].unique())


# In[114]:


table_kappa = pa.Table.from_pandas(df_kappa)
pa.parquet.write_table(table_kappa, 'E:/Manh/5th semester/thesis/deposit_thesis/df_kappa.parquet')

table_ihhi = pa.Table.from_pandas(df_ihhi)
pa.parquet.write_table(table_ihhi, 'E:/Manh/5th semester/thesis/deposit_thesis/df_ihhi.parquet')


# In[38]:


print(merged_ownership['per'].unique())


# In[47]:


print("Number of unique rssdid values:", merged_ownership['rssdid'].nunique())


# In[12]:


# df_sample: prd_typ_join,prod_nm, termlength, termtype, rate, apy, datesurveyed,inst_nm,rssd_id,asset_sz,city,state,county,branches,institutiondeposits,branchdeposits,cnty_fps
df_sp = df_sample[['prd_typ_join', 'prod_nm', 'termlength', 'termtype', 'rate', 'apy', 'datesurveyed','year','inst_nm', 'rssd_id', 'asset_sz', 'city', 'state', 'county', 'branches', 'institutiondeposits', 'branchdeposits', 'cnty_fps','state_fps']].copy()
df_sp['month'] = df_sample['datesurveyed'].dt.month

df_sp = df_sp[df_sp['year'] >= 2003]
df_sp = df_sp[df_sp['rssd_id'] != 0]

df_sp = df_sp[df_sp['termlength'] != 0]
df_sp['cnty_fps'] = df_sp['cnty_fps'].astype(int)
df_sp['state_fps'] = df_sp['state_fps'].astype(int)

display(df_sp.head())


# In[17]:


pd.set_option('display.max_columns', None)
display(df_bank_sample[(df_bank_sample['rssd_id'] == 485559) & (df_bank_sample['year'] == 2010) & (df_bank_sample['month_x'] == 5)])


# In[13]:


quarter_mapping = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
df_sp['quarter'] = df_sp['month'].map(quarter_mapping)

display(df_sp.head())


# In[19]:


# merge sample data and call report-- bank characteristics
df_bank_sample = pd.merge(df_sp, df_bank, on = ['rssd_id','year','quarter'], how='left').dropna(subset = ['rssdid'])

df_bank_sample['rssdid'] = df_bank_sample['rssdid'].astype(int)
df_bank_sample['permco'] = df_bank_sample['permco'].astype(int)
df_bank_sample['permno'] = df_bank_sample['permno'].astype(int)

#The problem of multiple reported prices for a branch within a quarter

## replace the rate with the median rate among different periods
grouped_avg_rate = df_bank_sample.groupby(['cnty_fps', 'state_fps', 'rssdid', 'year', 'month_x', 'quarter', 'termlength'])['rate'].mean().reset_index()
df_bank_sample = pd.merge(df_bank_sample, grouped_avg_rate, on=['cnty_fps', 'state_fps', 'rssdid', 'year', 'month_x', 'quarter', 'termlength'], suffixes=('', '_avg'))
df_bank_sample['rate'] = df_bank_sample['rate_avg'].fillna(df_bank_sample['rate'])

##drop the duplicates if the reported prices remains the same within a quarter
df_bank_sample = df_bank_sample.drop_duplicates(subset=['cnty_fps', 'state_fps','rssdid','year','month_x','quarter','termlength','rate'])

df_bank_sample = df_bank_sample.drop(columns=['rate_avg'])

df_bank_sample = df_bank_sample.drop(columns=['month_y'])
display(df_bank_sample.head())


# In[10]:


display(df_bank_sample.head())


# In[20]:


table_bank_sample = pa.Table.from_pandas(df_bank_sample)
pa.parquet.write_table(table_bank_sample, 'E:/Manh/5th semester/thesis/deposit_thesis/df_bank_deposit.parquet')


# In[64]:





# In[36]:


df_ihhi = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/ihhi.parquet').to_pandas()
df_kappa = pq.read_table('E:/Manh/5th semester/thesis/deposit_thesis/kappa.parquet').to_pandas()
display(df_ihhi.head())


# In[37]:


display(df_kappa.head())


# In[32]:


treasury_path = 'E:/Manh/5th semester/thesis/deposit_thesis/DGS10.csv'
data_dgs = pd.read_csv(treasury_path)
df_dgs = pd.DataFrame(data_dgs)

df_dgs['date'] = pd.to_datetime(df_dgs['date'])
df_dgs['year'] = df_dgs['date'].dt.year
df_dgs['month'] = df_dgs['date'].dt.month

display(df_dgs)


# In[33]:


df_control = pd.merge(df_pop_income, df_dgs, on='year',how='left').dropna()
df_control = df_control[df_control['year'] >= 2003]
display(df_control)


# In[34]:


print(df_control['year'].unique())


# In[35]:


table_control = pa.Table.from_pandas(df_control)
pa.parquet.write_table(table_control, 'E:/Manh/5th semester/thesis/deposit_thesis/df_control_variable.parquet')


# In[ ]:




