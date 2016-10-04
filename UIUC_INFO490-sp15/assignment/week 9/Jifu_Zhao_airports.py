
# coding: utf-8

# ## Week 9 Assignment
# 
# - The template for this problem is [airports.ipynb](airports.ipynb).
# - When you are done, rename your file to `FirstName_LastName_airports.ipynb`
#   and submit it via Moodle.
# - In this problem, you will visualize the total number of flights
#   and current temperature of the top 20 airports 
#   using the 2001 flight data.
# - This week's assignment is one long, continuous problem,
#   but I split it up into three sections for easier grading.

# In[1]:

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from IPython.display import SVG, display_svg


# This is where we are going:

# In[2]:

# run this cell if there's no image
resp = requests.get("https://rawgit.com/INFO490/spring2015/master/week09/images/top20.svg")
SVG(resp.content)


# The circles are the top 20 airports in the U.S.
#   The size of each circle is proportional to the total number of
#   arrivals and departures in 2001.
#   The redder the color, the higher the temperature;
#   the bluer the color, the lower the temperature.
#   Thus, you will visualize three pieces of information in one plot:
#   the location of major airports,
#   the size of the airports,
#   and the current temperature.

# ### Problem 9.1. Top 20 Airports.
# 
# In the first part of this week's assignment,
#   you will add the number of departures and the number of arrivals in 2001
#   to find which 20 airports had the most number of flights.
# 
# You should use the `Dest` and `Origin` columns of the 2001 flight data `2001.csv`.
#   Note that each airport is identified by [IATA codes](http://en.wikipedia.org/wiki/International_Air_Transport_Association_code).

# In[3]:

dest_origin = pd.read_csv('/data/airline/2001.csv', encoding='latin-1', usecols=('Dest', 'Origin'))
print(dest_origin.head())


# #### Function: get_flights()
# 
# Count the total number of departures from and arrivals to each airport.
#   In other words, first count the number of times each airport appears in the `Origin` column
#   to get
#   
#     Dest
#     ABE      5262
#     ABI      2567
#     ABQ     36229
#     ACT      2682
#     ADQ       726
#     
#   (only the first 5 columns are shown).
#   Then, count the number of times each airport apears in the `Dest` column to get
#   
#     Origin
#     ABE        5262
#     ABI        2567
#     ABQ       36248
#     ACT        2686
#     ACY           1
# 
#   Add them up get the total number,
#   and store the result in a column named `flights`,
#   
#          flights
#     ABE    10524
#     ABI     5134
#     ABQ    72477
#     ACT     5368
#     ACY        1
#     
#   Note that some airports have no departures or arrivals (e.g. `ACY`).
#   
# Hint: I would use `groupby(...).size()` on `Dest` and `Origin` columns to get
#   the number of departures and arrivals, respectively.
#   I would then use
#   [`pandas.DataFrame.add()`](http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.add.html)
#   (with `fill_value` and `columns` parameters).

# In[6]:

def get_flights(df):
    '''Takes a dataframe that has two columns Dest and Origin
    and returns a dataframe that has a column named flights
    and is indexed by IATA codes.
    
    Parameters
    ----------
    df: pandas.DataFrame
    
    Returns
    -------
    pandas.DataFrame
    '''
    result = pd.DataFrame()
    
    # your code goes here
    
    departure = df.groupby('Dest').size()
    origin = df.groupby('Origin').size()
    result = origin.add(departure, fill_value = 0, axis = 0)
    result = pd.DataFrame(result, columns = ['flights'])
    
    return result


# In the end, when you execute
# 
# ```python
# flights = get_flights(dest_origin)
# print(flights)
# ```
# 
# you should get
# 
# ```text
#      flights
# ABE    10524
# ABI     5134
# ABQ    72477
# ACT     5368
# ACY        1
# ADQ     1452
# AKN      568
# ALB    32713
# AMA    12267
# ANC    42381
# APF      725
# ATL   503163
# AUS    85809
# AVL     3172
# AVP     2893
# AZO     5290
# BDL    71983
# BET     2306
# BFL     3338
# BGM      751
# BGR     7417
# BHM    37566
# BIL     6249
# BIS     2779
# BMI     2869
# BNA   112603
# BOI    24152
# BOS   266032
# BPT     3481
# BQN      518
# ..       ...
# SHV    12011
# SIT     2758
# SJC   144653
# SJT     4505
# SJU    52957
# SLC   152859
# SMF    80394
# SNA    86871
# SPS     3985
# SRQ     9044
# STL   324477
# STT     6723
# STX     1817
# SUX      546
# SWF     2386
# SYR    22281
# TLH     2957
# TOL     4483
# TPA   137286
# TRI     1095
# TUL    45562
# TUS    39101
# TVC     5067
# TXK     3475
# TYR     6361
# TYS    11131
# VPS     3455
# WRG     1452
# XNA    11749
# YAK     1450
# 
# [231 rows x 1 columns]
# ```

# In[7]:

flights = get_flights(dest_origin)
print(flights)


# To keep the problem simple, we will use only the top 20 airports.
# 
# ```text
#      flights
# ORD   682636
# DFW   624361
# ATL   503163
# LAX   450019
# PHX   368631
# STL   324477
# DTW   297522
# MSP   284955
# LAS   272293
# BOS   266032
# DEN   265184
# IAH   257193
# CLT   256626
# SFO   243473
# EWR   241016
# PHL   239390
# LGA   232964
# PIT   212738
# SEA   205486
# BWI   199674
# ```

# In[8]:

top20 = flights.sort('flights', ascending=False)[:20]
print(top20)


# ### Problem 9.2. XML Scrapping.
# 
# As mentioned earlier, our airports are identified by IATA codes in the dataframe.
#   However, the circles in the SVG file are identified by the city names
#   (e.g. Chicago or San Francisco).
#   For example, the circle that corresponds to O'Hare Airport is represented
#   by the following code in
#   [`airports.svg`](https://raw.githubusercontent.com/INFO490/spring2015/master/week09/images/airports.svg):
#   
# ```xml
# <circle cx="367.68198" cy="118.095" stroke="black" stroke-width="1" r="5" fill="red"
#   id="Chicago"/>
# ```
#   So we need to match the IATA codes with the city names.
#   You could use a supplementary data set such as
#   [airports.csv](http://stat-computing.org/dataexpo/2009/supplemental-data.html)
#   that contains all this information,
#   but let's pretend that no such file exists and
#   we have to gather this information ourselves.
#   
# FAA provides an [XML service](http://services.faa.gov/docs/services/airport/),
#   which lets us get the airport status, including known delays and weather data.
#   You should read http://services.faa.gov/docs/services/airport/ and
#   try a few sample XML request to make sure you understand how it works.
#   
# From the XML response,
#   we will extract the city name of the airport as well as the current temperature.
#   
# #### Functions: xml\_city() and xml\_temp()
# 
# - Write a function named `xml_city` that takes a url of an XML request (str)
#   and returns the city name (str).
# - Write a function named `xml_temp` that takes a url of an XML request (str)
#   and returns the current temperature (float).

# In[9]:

def xml_city(url):
    '''
    Takes a url (str) with IATA code and returns the city name (str).
    
    Parameter
    ---------
    url: A str, e.g. "http://services.faa.gov/airport/status/SFO?format=application/xml"
    
    Returns
    -------
    A str.
    
    Example
    -------
    >>> print(xml_city("http://services.faa.gov/airport/status/SFO?format=application/xml"))
    San Francisco
    '''
    result = ""
    
    # your code goes here
    
    resp = BeautifulSoup(requests.get(url).content)
    city = resp.body.airportstatus.city.string
    result += city
    
    return result

def xml_temp(url):
    '''
    Takes a url (str) with IATA code and returns the current temperature (float).
    
    Parameter
    ---------
    url: A str, e.g. "http://services.faa.gov/airport/status/SFO?format=application/xml"
    
    Returns
    -------
    A float.
    
    Example
    -------
    >>> print(xml_temp("http://services.faa.gov/airport/status/SFO?format=application/xml"))
    71.0
    '''
    result = ""
    
    # your code goes here
    
    resp = BeautifulSoup(requests.get(url).content)
    tmp = resp.body.airportstatus.weather.temp.string.split(' ')[0]
    result += tmp
    result = float(result)
    
    return result


# When you run the following cell, you should get
# 
# ```text
# The current temperature at San Francisco (<class 'str'>) is 71.0 (<class 'float'>) degrees Fahrenheit.
# ```
# 
# (The temperature may be different.)

# In[10]:

sfo_city = xml_city("http://services.faa.gov/airport/status/SFO?format=application/xml")
sfo_temp = xml_temp("http://services.faa.gov/airport/status/SFO?format=application/xml")
print("The current temperature at {0} ({1}) is {2} ({3}) degrees Fahrenheit."
      "".format(sfo_city, type(sfo_city), sfo_temp, type(sfo_temp)))


# #### Function: add\_city\_temp()
# 
# - Write a function that takes a Pandas DataFrame (indexed by IATA codes)
#   and appends two additional columns `city` and `temp`.
#   
# If you had a dataframe that looks like 
# 
# ```python
# print(top20.head(3))
# ```
# ```text
#      flights
# ORD   682636
# DFW   624361
# ATL   503163
# ```
# 
# then `add_city_temp()` should return
# 
# ```python
# top20 = add_city_temp(top20)
# print(top20.head(3))
# ```
# ```text
#      flights             city  temp
# ORD   682636          Chicago    54
# DFW   624361  Dallas-Ft Worth    68
# ATL   503163          Atlanta    63
# ```
# 
# (The temperatures will be different in your case.)
# 
# Hint: Create a new column (e.g. `url`) and use the indices of the DataFrame
#   to create appropriate links `http://services.faa.gov/airport/status/XXX?format=application/xml`
#   for each airport.
#   Use [`pandas.DataFrame.apply()`](http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.apply.html)
#   and the `xml_city()` function to create a column named `city`.
#   Similary, use `pandas.DataFrame.apply()` with `xml_temp()` to create a column named `temp`.
#   Finally, use
#   [`pandas.DataFrame.drop()`](http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.drop.html)

# In[11]:

def add_city_temp(df):
    '''Takes a Pandas DataFrame that has a column named flights
    and is indexed by IATA codes.
    Returns a DataFrame with two additional columns: 
    city (str) and temp (float).
    
    Parameter
    ---------
    df: A pandas.DataFrame
    
    Returns
    -------
    A pandas.DataFrame
    '''
    
    result = pd.DataFrame()
    
    # your code goes here
    
    url = "http://services.faa.gov/airport/status/SFO?format=application/xml"
    df['airport'] = df.index.to_series()
    link = []
    for item in df['airport']:
        link.append(url.replace("SFO", item))
    df['url'] = link
    
    df['city'] = df['url'].apply(xml_city)
    df['temp'] = df['url'].apply(xml_temp)
    result = df.drop(['airport', 'url'], axis = 1) 
    
    return result


# When you run the following cell, you should get
# 
# ```text
#      flights             city  temp
# ORD   682636          Chicago    54
# DFW   624361  Dallas-Ft Worth    68
# ATL   503163          Atlanta    63
# ```
# 
# (The temperatures may be different.)

# In[12]:

top20 = add_city_temp(top20)
print(top20.head(3))


# In Problem 3, it will be easier if the dataframe is indexed by
#   `city` instead of IATA codes.
#   
# When you run the following cell, you should get
# 
# ```text
#                 index  flights  temp
# city                                
# Chicago           ORD   682636    54
# Dallas-Ft Worth   DFW   624361    68
# Atlanta           ATL   503163    63
# ```

# In[13]:

top20 = top20.reset_index().set_index('city')
print(top20.head(3))


# ### Problem 9.3. Data Visualization.
# 
# - Use the blank U.S. map with 20 major cities
#   [airports.svg](https://rawgit.com/INFO490/spring2015/master/week09/images/airports.svg)
#   to visualize the `top20` dataframe.
#   
# Run the following cell to display the SVG template.

# In[14]:

resp = requests.get("https://rawgit.com/INFO490/spring2015/master/week09/images/airports.svg")
usairports = resp.content
display_svg(usairports, raw=True)


# You should use `BeautifulSoup` to adjust the size and color of the circles
#   to create a figure similar to the following.
#   Again, the radius of a circle is proportional to the total traffic
#   of that airport.
#   The color of a circle depends on the current temperature at that airport.

# In[15]:

resp = requests.get("https://rawgit.com/INFO490/spring2015/master/week09/images/top20.svg")
SVG(resp.content)


# (Your figure does not have to look exactly the same.
#   Choose a different set of sizes and colors if you wish.)
# 
# Lesson 3 covers how to use `BeautifulSoup` to parse an SVG file (which is in XML format),
#   search for the FIPS code of each county, and replace the colors using a dataframe (unemployment data).
#   Although we are searching for different objects (`circle`) and
#   using a different data set (`top20`), the same principle applies.
#   
# Remember that each circle is represented by the following XML code:
# 
# ```xml
# <circle cx="367.68198" cy="118.095" stroke="black" stroke-width="1" r="5" fill="red"
#   id="Chicago"/>
# ```
# 
# So you should use `BeautifulSoup` to parse the `usairports` (the contents of `airports.svg` file)
#   and search for all instances of `circle`.
#   Then, iterate over each instance of `circle`
#   and replace the `r` variable with the number of flights `flights` in the `top20` dataframe
#   (multiplied/divided by some proportionaly constant
#   since a circle with a radius `r="682636"` would not work).
#   You should also replace the `fill` variable with a color the current temperature `temp`.
#   After replacing all instances of `circle`,
#   display the result with `display_svg(soup.prettify(), raw=True)`.

# In[16]:

# your code goes here

soup = BeautifulSoup(usairports)
circle = soup.findAll('circle')
colors = ['#edf8fb', '#ccece6', '#99d8c9', '#66c2a4', '#2ca25f', '#006d2c']

for item in circle:
    number = float(top20[top20.index == item['id']]['flights'])
    temp = float(top20[top20.index == item['id']]['temp'])
    item['r'] = number*0.00003
    
    if temp > 70:
        color_n = 5
    elif temp > 60:
        color_n = 4
    elif temp > 50:
        color_n = 3
    elif temp > 40:
        color_n = 2
    elif temp > 30:
        color_n = 1
    elif temp > 20:
        color_n = 0
        
    color = colors[color_n]
    item['fill'] = color
    
display_svg(soup.prettify(),  raw=True)


# In[ ]:



