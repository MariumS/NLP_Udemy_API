# NLP_Udemy_API

Problem Statement:
<br>
The question this research attempts to answer is if there is a strong connection between the words used in a Udemy course headline and the price of the course. This could help those listing courses on Udemy to write headlines, and further, pick specific topics, that best fit in with the price bracket they are aiming to sell their course in.
<br> Based on the guidelines put out by Udemy, the lowest an instructor can price a course is 19.99USD, and the highest is 199.99USD. Courses can also be listed as free, and are only allowed to switch from paid to free (or vice versa) once. Courses are often put on promotional discounts, influenced by market factors. The fact that courses are not always in the same price bracket still allows for using the results of this research to be valid at the point of API call.
<br><br>
Data:<br>
The data comes from the Udemy courses API. It includes 5000 (or the max) records each from 5 course categories. The course categories are Design, Marketing, Development, Music and Lifestyle. The point of interests are the course headline, course category, and course price. The category is not included in the API result, so it is appended seperately, based on the category specified in the API call.
In total the dataset includes 23912 rows and 4 columns: id, price, headline and category.
<br><br>
Hypothesis:<br>
I predict that there will be certain words that correlate to higher prices in courses. These words will likely correspond to hot topics at the time of the API call. There might be futher research to be done to see if these words also see a recent spike on google trends.
<br><br>
Methodology:<br>
For this research, the course prices are segmented into 2 categories, high and low. Low courses are priced between 0-199.9USD and high courses are priced between 100USD and 200USD.
<br>
For the Naive Bayes modelling, the course headlines are processed, cleaned of stopwords, and tokenized. Naive bayes classification is done on the data as a whole, and on each category individually. Then a logisitic regression is attempted.
<br>
<br>
Results:
<br>
(these results reflect the state at writing, however, running the models with the data from different api calls will give different results)
<br>
The models do not preform well, none of them surpassing 11% above the baseline, either on the data as a whole or on individual categories.

The highest preforming model is Naive Bayes on the Development category. The baseline accuracy would be 53.1%, the accuracy of this model is 64.00%, which is 10.9% higher than baseline.
The second highest preforming model is LogisticRegression on the Development category, at 9.2% higher than baseline.

The lowest preforming model is the Multinomial Naive Bayes model for the lifestyle category. The baseline accuracy would be 78.3%, and the accuracy of the model is 70.80%, -7.5% lower than baseline.

There seems to be a correlation between higher baselines (more unevenly segmented data) and worse preforming models. This should be looked into more.

For more details, see the last section of this notebook.

<br>
<br>
Conclusions:<br>
There is some relation between the words in a course headline and the price bracket of that headline, however, it cannot be said to be a highly predictive variable, based on the results of this notebook. It is more predictive for some categories over others.

<br>
<br>
Next steps:<br>
Refine price bracketing to lower skew towards either 1 or 0.
<br>Figure out how to get free courses included in API call (they were as of May 29 2020, and made the division between 1 and 0 almost 50/50 when segmenting the data at ((max/2) -.01)).
<br>
Refine model parameters
<br>
Add more categories
<br>
Look into running nonbinary models?
