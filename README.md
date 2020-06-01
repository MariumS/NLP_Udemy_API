

<center>Problem Statement:</center>
<br>
The question this research attempts to answer is if there is a strong connection between the words used in a Udemy course headline and the price of the course. This could help those listing courses on Udemy to write headlines, and further, pick specific topics, that best fit in with the price bracket they are aiming to sell their course in. 
<br> 

Based on the guidelines put out by Udemy, the lowest an instructor can price a course is 19.99USD, and the highest is 199.99USD. Courses can also be listed as free, and are only allowed to switch from paid to free (or vice versa) once. Courses are often put on promotional discounts, influenced by market factors. The fact that courses are not always in the same price bracket still allows for using the results of this research to be valid at the point of API call. 
<br><br>
<center>Data:</center>
<br>
The data comes from the Udemy courses API. I did a call for 5000 records each from 5 course categories. The course categories are Design, Marketing, Development, Music and Lifestyle. Not all categories had 5000 courses listed at the point of API call, so some returned less rows (see 4.1.3). 
<br>

The point of interests are the course headline, course category, and course price. The category is not included in the API result, so it is appended seperately, based on the category specified in the API call. 
In total the dataset includes 21251 rows and 4 columns: id, price, headline and category. 
<br><br>
<center>Hypothesis:</center>
<br>
I predict that there will be certain words that correlate to higher prices in courses. These words will likely correspond to hot topics at the time of the API call. There might be futher research to be done to see if these words also see a recent spike on google trends. 

ALTERNATIVE HYPOTHESIS:
The price bracket of a course can be predicted by the words in the course headline

NULL HYPOTHESIS:
The price bracket of a course cannot be predicted by the words in the course headline

Threshold: We will reject the null hypothesis if the accuracy of any model for the dataset as a whole is 25% over the baseline. (75% accurate if the baseline is 50%, for example)

<br><br>
<center>Methodology:</center>
    <br>
For this research, the course prices are segmented into 2 categories, high and low. Low courses are priced between 19.99 to (max price/2-.01)USD and high courses are priced between (max price/2-.01 USD) and 199.99USD. The category of low courses includes the midpoint cutoff, while the higher courses category does not. 
<br>

For the modeling, the course headlines are processed, cleaned of stopwords, and tokenized. Naive bayes and logistic regression classification is done on the data as a whole, and on each category individually.
<br>

<br>
<center>Results:</center>
<br>
(these results reflect the state at writing, June 1, 2020, however, running the models with the data from different api calls will give different results)
<br>
The models do not preform well, none of them surpassing 15% above the baseline, either on the data as a whole or on individual categories. 

The highest preforming model is Naive Bayes on the Development category, at 7.1% higher than baseline.
The second highest preforming model is LogisticRegression on the Development category, at 6.4% higher than baseline.

The lowest preforming model is the Multinomial Naive Bayes model for the lifestyle category, at -2.9% lower than baseline. 

For the overall dataset the logistic regression outpreforms the naive bayes in terms of accuracy above the baseline. For logisitic regression the accuracy is 4.6% higher than baseline, and for naive bayes the accuracy is 2.8% higher than baseline

For full results, see section 9.

<br>
<br>
<center>Conclusions:</center>
    <br>
There is some relation between the words in a course headline and the price bracket of that headline, however, it cannot be said to be a highly predictive variable, based on the methodology and results of this notebook. It is more predictive for some categories over other.
<br>
We do not reject the null hypothesis. 

<br>
<br>
<center>Next steps:</center>
    <br>
Refine price bracketing make ratio of price brackets more even. 
<br>Figure out how to get free courses included in API call (they were as of May 29 2020, and made the division between 1 and 0 almost 50/50 when segmenting the data at ((max/2) -.01)). 
<br>
Refine model parameters
<br>
Add more categories
<br>
Look into running nonbinary models
