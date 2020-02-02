# Regression-for-Interview

This program aims to predict the "Market Share_total" of Montreal citizen's film taste.

At first, I imported the test and training data as a data frame then I needed to change the format of some columns from string to integer to make the prediction much easier for "sklearn" library. 

Unfortunately, some columns were not complete and some rows in the dataset had null values. I could give them a number like mean of features but I decided to delete those rows.

At last, I used "sklearn" library to use linear regression and printed the predicted "Market Share_total". 
 
