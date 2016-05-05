import sqlite3
import csv
from datetime import datetime as dt

# This program reads in the items from the training set contained
# in 'data/train.csv' and adds them into the database 'data/sf_crimes.sqlite'

conn = sqlite3.connect('data/sf_crimes.sqlite')
print('Database connection to sf_crimes established')
c = conn.cursor()
c.execute("""CREATE TABLE crimes (X REAL, Y REAL, Category TEXT,
    Year TEXT, Month TEXT, Day TEXT, Hour TEXT, Minute TEXT)""")

with open('data/train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        y = float(row['Y'])
        if y > 40.0:
            print('\nSkipping bad data point')
            continue
        date = row['Dates']
        cat = row['Category']
        x = float(row['X'])
        date = dt.strptime(date, '%Y-%m-%d %H:%M:%S')
        year = str(date.year)
        month = str(date.month)
        day = str(date.weekday())
        hour = str(date.hour)
        minute = str(date.minute)
        insert = (x, y, cat, year, month, day, hour, minute)
        c.execute("INSERT INTO crimes VALUES(?,?,?,?,?,?,?,?)", insert)

conn.commit()
print('\nCommiting changes')
conn.close()
print('\nClosing connection')    
