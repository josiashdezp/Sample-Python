#Import libraries required for this project
import pandas as pd
import sklearn.feature_selection as feature_selection
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plotter
import numpy as np
import seaborn as sns
from mysql.connector import connection
import csv as csv_handler


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# ASSIGNMENT 1
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


'''
----------------------------------------------
Task 1. import files and generate some summary
----------------------------------------------
'''
print('\n---------------------------------------------- \n Task 1. import files and generate some summary \n ----------------------------------------------')

# Open the CSV file
flights = pd.read_csv(r'C:/Users/josia/PycharmProjects/PythonProject_Week1/flights.csv')

#List number of rows and columns
print("The file 'Flights.csv' has " + str(flights.shape[0])  + " columns and " + str(flights.shape[1])   + " rows.\n")

#Display the data types
print("These are the data types of each column in the Dataset:")
print(flights.dtypes)

#Calculate number of flights per carrier and display them
quantity = flights.groupby("carrier")["flight"].count()
print("\nThere are " + str(quantity.size) + " types of carriers in the Dataset\nwith the following number of flights (carrier size):")
print(quantity)

'''
----------------------------------------------
Task 2. Data aggregation, filtering, and sorting 
----------------------------------------------
'''
print('\n---------------------------------------------- \n Task 2. Data aggregation, filtering, and sorting \n ----------------------------------------------\n')

# Calculate the mean of the delays for departures and arrivals for each flight of each carrier
# sort the results in descending order by Average Departure Delay
avg_delays = flights.groupby(['carrier', 'flight'],as_index=False).agg(
    avg_dep_delay=("dep_delay", "mean"),
    avg_arr_delay=("arr_delay", "mean")).sort_values(by=['carrier','avg_dep_delay'],ascending=[True,False])
print('The following table contains the mean of the delays for departures\n and arrivals for each flight of each carrier. \nThe results are sorted in descending order by Average Departure Delay\n')
print(avg_delays)

# Compare the average arr_delay in January and February vs in June and July to see if more delays happen
# in winter months or summer months

winter_average = round(flights[(flights['month'].isin([1,2]))]['arr_delay'].mean(),2)
summer_average = round(flights[(flights['month'].isin([6,7]))]['arr_delay'].mean(),2)

if summer_average > winter_average:
    print('The average number of delays in the arrival of flights during Summer was ' + str(summer_average))
    print('The average number of delays in the arrival of flights during Winter was ' + str(winter_average))
    print('Thus there are more delays during Summer than during the Winter season.')
elif summer_average < winter_average:
    print('The average number of delays in the arrival of flights during Summer was ' + str(summer_average))
    print('The average number of delays in the arrival of flights during Winter was ' + str(winter_average))
    print('Thus there are more delays during Winter than during the Summer season.')
else:
    print('The average number of delays in the arrival of flights during Summer was ' + str(summer_average))
    print('The average number of delays in the arrival of flights during Winter was ' + str(winter_average))
    print('Thus there are no differences in the number of delays between the Winter and Summer season.')


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# ASSIGNMENT 2
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# Open the CSV file
anal_data = pd.read_csv(r'C:/Users/josia/PycharmProjects/PythonProject_Week1/2017CHR_CSV_Analytic_Data-new.csv')

'''
--------------------------------------------------------------------------------------------
1)	Check if any columns have missing value (show the result). If there are columns
 having missing value, impute the columns (with missing values). 
 If there is no missing value, leave the data frame as is. 
--------------------------------------------------------------------------------------------
'''

# Initialize the counter for the missing values and then verify each row in the array
counter = 0
for titles,values in anal_data.isnull().sum().items():
    if values:
        counter+= 1     # count each missing value

# display a message in the console displaying the results
if counter > 0:
    print('\n'+'There are ' + str(counter) + ' columns with missing values in the dataset.\n')
else:
    print('\n'+'There are no columns with missing values in the dataset. We''ll leave the data frame as is.\n' )



'''
--------------------------------------------------------------------------------------------
2)	Drop (remove) the identifier columns: 5-Digit FIPS Code, statecode, countycode, county
--------------------------------------------------------------------------------------------
'''
new_anal_data = anal_data.drop(['5-Digit FIPS Code', 'statecode', 'countycode', 'county'], axis=1)
print('\n'+'The identifier columns: 5-Digit FIPS Code, statecode, countycode, county were successfully removed from the dataframe.' )
print('Original columns count:' + str(len(anal_data.columns)))
print('New columns count:' + str(len(new_anal_data.columns))+'\n')




'''
--------------------------------------------------------------------------------------------
3)	Use z-score normalization to normalize these columns: Poor physical health days Value, Poor mental health days Value, Food environment index Value (5 points)
--------------------------------------------------------------------------------------------
'''
# Initialize the instance of the Standard Scaler class
scaler = StandardScaler()

# Let's print a sample the data to have and idea of their current values
print('This is a sample of the current values of the variables to normalize \n')
print(anal_data[['Poor physical health days Value', 'Poor mental health days Value', 'Food environment index Value']].head())

# Replace the values of the selected columns with the normalized ones and print them in the console
anal_data[['Poor physical health days Value', 'Poor mental health days Value', 'Food environment index Value']] = scaler.fit_transform(anal_data[['Poor physical health days Value', 'Poor mental health days Value', 'Food environment index Value']])
print('\n'+'These are the new values after the z-score normalization:')
print(anal_data[['Poor physical health days Value', 'Poor mental health days Value', 'Food environment index Value']].head())


'''
--------------------------------------------------------------------------------------------
4)	Create a new column “Diabetes-level” by coding the “Diabetes Value” into four groups, and label them as low, median low, median high, and high
--------------------------------------------------------------------------------------------
'''
# Define the diabetes levels names / labels
diabetes_group = ['Low','Median Low','Median High', 'High']

# Define the filters for each group
my_bins = [0, 0.08, 0.12, 0.18,0.24]

# Let's print a sample the data to have and idea of their current values
print('This is a sample of the current Diabetes Values in the dataset. \n')
print(anal_data[['Diabetes Value']].head())

# generate diabetes groups using the bins
anal_data['Diabetes Level'] = pd.cut(anal_data['Diabetes Value'], bins=my_bins, labels=diabetes_group)

#print the resulting groups and values
print('Now, those values have been grouped in 4 Diabetes categories by its Diabetes Value \n')
print(anal_data[['Diabetes Value','Diabetes Level']].head())



'''
--------------------------------------------------------------------------------------------
5)	Apply feature selection to find the top 5 features relevant to “Diabetes-level”.
--------------------------------------------------------------------------------------------
'''
# In this case there is no need to encode the input columns because their values are already numerical
# The target column Y is the Diabetes Level which is categorical.
# Here we use the DataFrame from who we dropped the
# identifier columns: 5-Digit FIPS Code, statecode, countycode, county
# because those columns are not meaningful and do not add valuable data for this analysis

# generate diabetes groups using the bins
new_anal_data['Diabetes Level'] = pd.cut(new_anal_data['Diabetes Value'], bins=my_bins, labels=diabetes_group)

# define the target column Y
y = new_anal_data['Diabetes Level']

# Define the Input columns
x = new_anal_data.drop(['Diabetes Level', 'Diabetes Value'],axis=1)

# apply information gain as feature selection
results = dict(zip(x.columns,feature_selection.mutual_info_classif(x, y)))

# Sort the dictionary by values
sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

# Convert the dictionary to a DataFrame
features_selection_results = pd.DataFrame(list(sorted_results.items()), columns=['Feature', 'Mutual Information'])

# Print the DataFrame
print(features_selection_results)




# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# ASSIGNMENT 3
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


#Opening the file to process as a DataFrame
wines = pd.read_csv(r'Wine.csv')

'''
--------------------------------------------------------------------------------------------
 Draw a bar chart using 'quality' and 'alcohol' column
 to see if the quality level has
 any relationship with the alcohol amount (5 points)
--------------------------------------------------------------------------------------------
'''


#Convert the quality column to NUMERIC using the Ordinal Encoder
wine_cat = [['cero', 'one', 'two', 'three', 'four','five','six','seven','eight','nine','ten',]]
label = OrdinalEncoder(categories=wine_cat)
wines['rated_quality'] = label.fit_transform(wines[['quality']])

#Assign the data values to the plot
plotter.bar(x=wines['alcohol'],height=wines['rated_quality'],width=0.5,edgecolor='black')

#Configure the parameters for the plot and display it
plotter.xticks(np.arange(8, 16, step=0.5))
plotter.title('Quality of the Wine vs Percentage of Alcohol')
plotter.xlabel('Alcohol')
plotter.ylabel('Quality')
plotter.show()

#Create another bar chart with seaborn
plot1 = sns.catplot(x='quality', hue='alcohol', data=wines, kind="count")

#Configure the parameters for the plot and display it
plot1.set_titles('Number of Wines per Alcohol Percentage in each Quality Category ')
plot1.set_xlabels('Alcohol')
plot1.set_ylabels('Quality')
plotter.show()

'''
--------------------------------------------------------------------------------------------
# Draw a histogram plot using 'total sulfur dioxide', and
# explain what you learn from the figure (5 points)
--------------------------------------------------------------------------------------------
'''


# Calculate total sulfur dioxide median
wines_median = wines['total sulfur dioxide'].median()


# Calculate cumulative distribution
wines_sorted = wines.sort_values(by='total sulfur dioxide')
wines_sorted['cumulative'] = wines_sorted['total sulfur dioxide'].cumsum()
cum_wines_median = wines_sorted['cumulative'].median()

# Create a figure with two subplots side by side
fig, (plot2, plot3) = plotter.subplots(nrows=1,ncols=2, figsize=(14, 6))

#Plot distribution Plot1
sns.histplot(x='total sulfur dioxide',data=wines,ax=plot2)
plot2.set_title('Histogram of Total Sulfur Dioxide ')
plot2.set_xlabel('Total Sulfur Dioxide Quantity')
plot2.set_ylabel('Frequency')
plot2.set_xticks(np.arange(0,200,step=10))
plot2.axvline(wines_median, color='red', linestyle='--', label=f'50th Percentile (Median): {wines_median}')
plot2.legend()

# Plot cumulative distribution: Plot2
sns.set_style("darkgrid")
sns.histplot(x='cumulative',data=wines_sorted,kde=True,label='Cumulative Sulfur',ax=plot3,cumulative=True)
plot3.set_title('Cumulative Distribution of Total Sulfur Dioxide')
plot3.set_xlabel('Total Sulfur Dioxide')
plot3.set_ylabel('Cumulative Value')
plot3.set_xticks(np.arange(0,50000,step=2000))
plot3.tick_params(axis='x',rotation=80,labelsize=9,)
plot3.set_autoscalex_on(True)
plot3.axvline(cum_wines_median, color='red', linestyle='--', label=f'50th Percentile (Median): {cum_wines_median}')
plot3.legend()

# Add title and labels
fig.align_labels()
plotter.tight_layout()
plotter.show()

'''
--------------------------------------------------------------------------------------------

# (3) Draw a scatter plot using 'residual sugar' and 'quality', and explain what you learn
# from the figure (5 points)

# (4)	Draw a hexbin plot using `residual sugar' and 'alcohol', and explain what you learn
# from the figure (5 points)
--------------------------------------------------------------------------------------------
'''



fig, (plot4, plot5) = plotter.subplots(nrows=1,ncols=2, figsize=(14, 6))

# Draw a scatter plot using `residual sugar' and 'quality',
plot4 = sns.scatterplot( data=wines.sort_values(by='rated_quality',ascending=False), x='residual sugar',y='quality',ax=plot4)
plot4.set_title('Scatter plot Wine Quality vs. Residual Sugar')
plot4.set_xlabel('Residual Sugar')
plot4.set_ylabel('Wine Quality')

# draw hexbin plot
my_hb_plot = plot5.hexbin(wines.sort_values(by='rated_quality', ascending=False)['residual sugar'], wines.sort_values(by='rated_quality', ascending=False)['alcohol'], gridsize=20, cmap='Greens', edgecolors='black', linewidths=0.5)
plot5.set_title('Hexbin plot Wine Quality vs. Percentage of Alcohol')
plot5.set_xlabel('Residual sugar')
plot5.set_ylabel('Alcohol')

# Adding a color bar to show the color scale
cb = fig.colorbar(my_hb_plot, ax=plot5)
cb.set_label('counts')

#display the graphics
plotter.show()


'''
--------------------------------------------------------------------------------------------
# Task 2. Use country_population_historic.csv to complete the following task (use the countries.csv
# file in the “dataset” folder on canvas to get the actual countries.)
# (1)	Select the 10 countries with the largest population in the year 1960, use the heatmap to
# show the changes of the populations of these 10 countries from 1960 to 1970. (15 points)
--------------------------------------------------------------------------------------------
'''

#Opening the files to process as a DataFrame
population_df = pd.read_csv(r'country_population_historic.csv')
countries = pd.read_csv(r'countries.csv')

# Retrieve only the data for the countries whose names are in the country list.
# (The reason is that the population's file has also data about geographical regions,
# and we want only data about countries)
population_df = population_df[population_df['Country Name'].isin(countries['name'])]

# Now we get records for the top 10 countries in the 60s.
top_country_60s: DataFrame = population_df.nlargest(n=10, columns='1960', keep="first")[['Country Name', '1960', '1970']]

# Adjust the numbers for displaying them in millions then add the change in a new column and order the table alphabetically
top_country_60s['1960'] = top_country_60s['1960']/1000000
top_country_60s['1970'] = top_country_60s['1970']/1000000
top_country_60s['Change'] =  top_country_60s['1970'] - top_country_60s['1960']

# draw the heatmap using the Country Name as the Index
plotter.figure(figsize=(12,8),edgecolor='Black',frameon=False)
plot6 = sns.heatmap(top_country_60s.set_index('Country Name'), annot=True,cmap='Greens',linewidths=0.30,fmt='.1f')
plotter.title('Population Change from 1960 to 1970')
plotter.xlabel('Population Change (Millions)')
plotter.ylabel('Country Name')
plotter.tick_params(labelsize=9)
plotter.show()




# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# ASSIGNMENT 4
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


'''
--------------------------------------------------------------------------------------------
#Connect to the Northwind database in the cloud
# and finish the following tasks. Please submit your code and screenshots of the output of each question in the Word
# file to submit to Canvas.


#Task 1. Export tables Product and Category to two CSV files (10 pts)
--------------------------------------------------------------------------------------------
'''

# Open connection with database.
my_connection = connection.MySQLConnection(user='readonlyuser',
                                 password='Fall2024Msis5193!',
                                 host='34.66.134.201',
                                 database='Northwind')

# initialize a cursor to execute SQL statement
cursor = my_connection.cursor()

# select the data from a table to be exported
cursor.execute("select * from Product")

#save the table to a file customers.csv
with open('Exported_Products.csv', 'w', newline='')  as csv_file1:
    writer1 = csv_handler.writer(csv_file1)
    writer1.writerow(cursor.column_names)
    for each_row in cursor:
        print(list(each_row))
        writer1.writerow(list(each_row))

# select the data from a table to be exported
cursor.execute("select * from Category")

#save the table to a file customers.csv
with open('Exported_Category.csv', 'w', newline='')  as csv_file2:
    writer2 = csv_handler.writer(csv_file2)
    writer2.writerow(cursor.column_names)
    for each_row in cursor:
        print(list(each_row))
        writer2.writerow(list(each_row))

# close DB connection
my_connection.close()



'''
--------------------------------------------------------------------------------------------
#Task 2.
# Use the exported CSV files to find the number of products in each category,
# and save the result into a new CSV file (20 pts)
--------------------------------------------------------------------------------------------
'''

# Open the files to start extracting the data
product_df = pd.read_csv(r'Exported_Products.csv')
category_df = pd.read_csv(r'Exported_Category.csv')

# Open/Create the csv file. if it is filled, delete its content.
with open('Exported_Prod_vs_Categ.csv', 'w', newline='') as csv_file3:
    writer3 = csv_handler.writer(csv_file3)
    csv_file3.truncate()

    #Write the header (columns names in row 1)
    writer3.writerow(['Category Id','Category Name','Quantity of Products'])

    #Iterate over each row in the DataFrame
    for index, category in category_df.iterrows():
        #Count the number of products whose categoryId is equal to the categoryId
        number_category = product_df[product_df['categoryId'] == category.iloc[0]].groupby(by='categoryId')['productId'].count()
        # Write on each row the CategoryID, CategoryName, Quantity of Products under that category
        writer3.writerow([category.iloc[0],category.iloc[1],number_category[category.iloc[0]]])