# Linear Regression

Linear regression is the supervised learning method that performs the Regression task when the target / dependent variable is continuous.Regression yields the desired prediction value premised on independent variables. It is mainly employed in determining the relationship between variables and prediction tasks. Different regression models differ in the type of relationship they compute between dependent and independent variables and the number of independent variables.
      Linear Regression is a predictive method that yields a linear relationship between the input (called X) and the prediction (Y^). It enables quantifying the relationship between a predictor and an output variable.
Machine learning systems employ linear regression to determine future values. It is the most common Machine Learning algorithm for making predictions and constructing models. Predictive analytics and modeling are the most common applications of linear regression in machine learning. It is premised on the principle of ordinary least square (OLS) / Mean square error (MSE). In statistics, OLS is employed to predict the unknown parameters of the linear regression function. Its objective is to minimize the sum of square differences between the observed dependent variables in the given data set and those predicted by linear regression function.

Its objective is to minimize the sum of square differences between the observed dependent variables in the given data set and those predicted by linear regression function. In order to predict  the continuous target variable we employ the following  notation:
  X’s to denote the input variables also called input features.
  Y’s to denote the output or target variable that we have to predict.
This implies, (X, Y) will denote one training example.
For some specific  ith  point, a pair (x<sup>(i)</sup>, y<sup>(i)</sup>) is called the training example and the dataset contains a list of n such training examples  where {x<sup>(i)</sup>, y<sup>(i)</sup>; i=1,2,….n)  } specifies training set. In a regression problem, we try to predict the target variable which is continuous and our objective is to learn a function h: X→y   (also called a hypothesis) such that h(x) is a good predictor for the corresponding value of y. 
Thus, mathematically  we can write 

>Let **D={x<sup>(i)</sup>, y<sup>(i)</sup>)**  ∣ **x<sup>(i)</sup> ∈  X**,**y<sup>(i)</sup>∈ Y, i= 1,2,……N}**,<br>

> **X={ x | x ∈ ℝ<sub>d</sub> }** and **Y= { y  |  y ∈ ℝ}**
}

And x 's  are  the d-dimensional vectors and y is a real value to be predicted.The Linear regression model for d-dimensional data introduces a weight vector w =w<sub>1</sub>,w<sub>2</sub>,...,w<sub>d</sub> and bias value w0 to predict the output value as a linear combination of the input features x<sup>i</sup><sub>j</sub>  ( where x<sup>i</sup><sub>j</sub> denotes the j<sup>th</sup> feature of the i<sup>th</sup>  point )  of the input x<sup>i<sup>→</sup></sup>as
![image](https://user-images.githubusercontent.com/97376928/160909144-6d3e6a0f-1856-44ad-957d-8f123d6e2f17.png)<br>
Where w<sup>(k)</sup>’s are the parameters(also known as weights ) parameterizing the space of linear functions mapping from X to Y. Sometimes, we modify the  x<sup>→</sup> and  w<sup>→</sup>  vectors to introduce  x<sub>0</sub> = 1(intercept term )   and include  w<sub>0</sub>  into the weight vector  w<sup>→</sup>    such the above equation simplifies as: <br>
![image](https://user-images.githubusercontent.com/97376928/160911903-a26d727e-8c5e-4911-a9b2-b5081bcf441e.png)<br>
Here, the d is the number of input variables and limits of k change from 1 before modifying the vectors to 0 after modification.The elements of x<sup>→</sup>  are features of a data point and to find out how much a particular feature contributes towards the output i.e, their contribution is represented by the corresponding weight from   w<sup>→</sup> .
## Model Evaluation
To find out how well our model performs,we require a cost function. A good choice in this case is the squared error function (although the choice is not random and there is a reason behind choosing this function):
![image](https://user-images.githubusercontent.com/97376928/160912682-6027599a-7e32-427d-b14d-55eaa591253b.png)<br>
We can find the cost over all the dataset as:<br>
![image](https://user-images.githubusercontent.com/97376928/160913118-2bf7bd36-0013-41f0-85f5-b498f6bdf5d8.png)<br>
The scaling factor 1/2 before the summation  is to make the math easier as you will find out.
## Gradient Descent to find the minimum of a cost function
The cost function is<br>
![image](https://user-images.githubusercontent.com/97376928/160913118-2bf7bd36-0013-41f0-85f5-b498f6bdf5d8.png)<br>
and the goal is to minimize the J(w<sup>→</sup>,D).To do so we start with some initial guess for w,that repeatedly changes w to make J(w<sup>→</sup>,D) smaller,until we converge to a value of w that minimizes J(w<sup>→</sup>,D).
To achieve this we employ gradient descent to find the minimum of the cost function that starts with some initial value of w<sub>k</sub>. For a particular w<sub>k</sub> ,where  k=0,1,…,d , we have<br>
![image](https://user-images.githubusercontent.com/97376928/160915298-f62757ba-c3e0-451f-b2b2-5505b612aa89.png)<br>
Lets now first find <br>
![image](https://user-images.githubusercontent.com/97376928/160916773-67b7df9c-1abe-40f5-9583-c5877ab0281a.png)<br>and then substitute that in the above equation 1.<br>
![image](https://user-images.githubusercontent.com/97376928/160917253-892a1507-f548-4630-8387-6f302c938bee.png)<br>
Substituting this in the above equation 1   we get,<br>
![image](https://user-images.githubusercontent.com/97376928/160917533-962b5a2f-e35a-41c1-80b8-80a3ab6265f9.png)<br>
We will find the gradient with respect to all  w<sub>k</sub>'s  to form the gradient vector<br>
![image](https://user-images.githubusercontent.com/97376928/160917799-ef237415-f71d-4cd9-8c93-16ca970f5165.png)<br>
where  J=J(w<sup>→</sup>,D) for ease of notation.
Hence we have ,<br>
          <center>Loop until convergence</center><br>

![image](https://user-images.githubusercontent.com/97376928/160918871-393be701-dcda-4dd5-86ed-2c128a65ea78.png)<br>

## Vectorized Form
Matrices are computationally efficient for defining the linear regression model and performing the subsequent analyses.The above equation can be denoted in vectorized form which leads to ease in implementation.<br>Let us define the matrix  X  which contains all the input vectors  x<sup>→</sup>  along its rows as:<br>
![image](https://user-images.githubusercontent.com/97376928/160919232-0cc96b44-c96b-4fd7-b7c1-827090e94d10.png)<br>
For two vectors  x<sup>→</sup> , y<sup>→</sup>  ∈ ℝ<sup>d</sup>  , we have<br>
![image](https://user-images.githubusercontent.com/97376928/160919539-d01851f8-ebe2-477d-b917-158e7f16eee8.png)<br>
 Where x<sub>k</sub><sup>→</sup> is the k-th column X matrix. That is, the k-th entry from each input vector x<sub>k</sub><sup>→</sup>.<br>

![image](https://user-images.githubusercontent.com/97376928/160920278-834219cb-4d9c-4c83-b953-fd4f1f68c00a.png)<br>To find the 
![image](https://user-images.githubusercontent.com/97376928/160920567-98990b27-67a6-4731-9627-d128cdbdad6c.png)<br>
we have:
![image](https://user-images.githubusercontent.com/97376928/160920700-3c79a84a-0134-4c4f-8df2-36700ae70d89.png)
## Normal equations 
 
  We have derived above the expression for ![image](https://user-images.githubusercontent.com/97376928/160920936-a673db35-6f2e-489c-8d5c-648b6600e657.png)
<br>
  Now to find the minimum using normal equations, we have<br>
  ![image](https://user-images.githubusercontent.com/97376928/160921206-36d5d2ec-211a-4c08-891c-c52602196cbd.png)<br>
  ## Implementation of Linear Regression Using Python code




In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

### Importing Libraries
To import necessary libraries for this task, execute the following import statements:


###### import pandas as pd
###### import numpy as np
###### import matplotlib.pyplot as plt
###### %matplotlib inline

#### Dataset

The dataset being used for this example has been made publicly available and can be downloaded from this link:

https://drive.google.com/open?id=1oakZCv7g3mlmCSdv9J8kdSaqO5_6dIOw

Note: This example was executed on a Windows based machine and the dataset was stored in "D:\datasets" folder. You can download the file in a different location as long as you change the dataset path accordingly.

The following command imports the CSV dataset using pandas:

###### dataset = pd.read_csv('D:\Datasets\student_scores.csv')


Now let's explore our dataset a bit. To do so, execute the following script:

###### dataset.shape


After doing this, you should see the following printed out:

###### (25, 2)

This means that our dataset has 25 rows and 2 columns. Let's take a look at what our dataset actually looks like. To do this, use the head() method:

##### dataset.head()


The above method retrieves the first 5 records from our dataset, which will look like this:

![fig1](https://user-images.githubusercontent.com/4158204/160814627-96ab0b1f-28c4-47b6-a9ba-aa8a20dd282f.JPG)


##### dataset.describe()


![fig2](https://user-images.githubusercontent.com/4158204/160868466-b64b7b63-8d1c-4266-bab6-3c9733a9cf80.JPG)


And finally, let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

##### dataset.plot(x='Hours', y='Scores', style='o')
##### plt.title('Hours vs Percentage')
##### plt.xlabel('Hours Studied')
##### plt.ylabel('Percentage Score')
##### plt.show()

In the script above, we use plot() function of the pandas dataframe and pass it the column names for x coordinate and y coordinate, which are "Hours" and "Scores" respectively.

The resulting plot will look like this:

![fig3](https://user-images.githubusercontent.com/4158204/160868781-5cad28f4-4101-41b3-a3f7-aeadbe8e16e5.JPG)

From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

##

### Preparing the Data


##



Now we have an idea about statistical details of our data. The next step is to divide the data into "attributes" and "labels". Attributes are the independent variables while labels are dependent variables whose values are to be predicted. In our dataset we only have two columns. We want to predict the percentage score depending upon the hours studied. Therefore our attribute set will consist of the "Hours" column, and the label will be the "Score" column. To extract the attributes and labels, execute the following script:

#### X = dataset.iloc[:, :-1].values
#### y = dataset.iloc[:, 1].values

The attributes are stored in the X variable. We specified "-1" as the range for columns since we wanted our attribute set to contain all the columns except the last one, which is "Scores". Similarly the y variable contains the labels. We specified 1 for the label column since the index for "Scores" column is 1. Remember, the column indexes start with 0, with 1 being the second column. In the next section, we will see a better way to specify columns for attributes and labels.

Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

#### from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


The above script splits 80% of the data to training set while 20% of the data to test set. The test_size variable is where we actually specify the proportion of test set.




## Training the Algorithm




We have split our data into training and testing sets, and now is finally the time to train our algorithm. Execute following command:

#### from sklearn.linear_model import LinearRegression
#### regressor = LinearRegression()
#### regressor.fit(X_train, y_train)

With Scikit-Learn it is extremely straight forward to implement linear regression models, as all you really need to do is import the LinearRegression class, instantiate it, and call the fit() method along with our training data. This is about as simple as it gets when using a machine learning library to train on your data.

In the theory section we said that linear regression model basically finds the best value for the intercept and slope, which results in a line that best fits the data. To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.

To retrieve the intercept:

##### print(regressor.intercept_)

The resulting value you see should be approximately 2.01816004143.

For retrieving the slope (coefficient of x):

##### print(regressor.coef_)

The result should be approximately 9.91065648.

This means that for every one unit of change in hours studied, the change in the score is about 9.91%. Or in simpler words, if a student studies one hour more than they previously studied for an exam, they can expect to achieve an increase of 9.91% in the score achieved by the student previously.


## Making Predictions




Now that we have trained our algorithm, it's time to make some predictions. To do so, we will use our test data and see how accurately our algorithm predicts the percentage score. To make pre-dictions on the test data, execute the following script:

##### y_pred = regressor.predict(X_test)


The y_pred is a numpy array that contains all the predicted values for the input values in the X_test series.

To compare the actual output values for X_test with the predicted values, execute the following script:

#### df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#### df

The output looks like this:
![fig4](https://user-images.githubusercontent.com/4158204/160877632-39fa20d9-b431-44b6-9f1f-e0cbc89f1ba4.JPG)


Though our model is not very precise, the predicted percentages are close to the actual ones.

Note:

The values in the columns above may be different in your case because the train_test_split function randomly splits data into train and test sets, and your splits are likely different from the one shown in this article.




## Evaluating the Algorithm




The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For regression algorithms, three evaluation metrics are commonly used:

* Mean Absolute Error (MAE) is the mean of the absolute value of the errors. It is calculated as:


![image](https://s3.amazonaws.com/stackabuse/media/linear-regression-python-scikit-learn-3.png)



* Mean Squared Error (MSE) is the mean of the squared errors and is calculated as:

![image](https://s3.amazonaws.com/stackabuse/media/linear-regression-python-scikit-learn-4.png)


* Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

![image](https://s3.amazonaws.com/stackabuse/media/linear-regression-python-scikit-learn-5.png)

Luckily, we don't have to perform these calculations manually. The Scikit-Learn library comes with pre-built functions that can be used to find out these values for us.

Let's find the values for these metrics using our test data. Execute the following code:

#### from sklearn import metrics
#### print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#### print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#### print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

The output will look similar to this (but probably slightly different):

Mean Absolute Error: 4.183859899

Mean Squared Error: 21.5987693072

Root Mean Squared Error: 4.6474476121

You can see that the value of root mean squared error is 4.64, which is less than 10% of the mean value of the percentages of all the students i.e. 51.48. This means that our algorithm did a decent job.





### This is a header

#### Some T-SQL Code

```tsql
SELECT This, [Is], A, Code, Block -- Using SSMS style syntax highlighting
    , REVERSE('abc')
FROM dbo.SomeTable s
    CROSS JOIN dbo.OtherTable o;
```

#### Some PowerShell Code

```powershell
Write-Host "This is a powershell Code block";

# There are many other languages you can use, but the style has to be loaded first

ForEach ($thing in $things) {
    Write-Output "It highlights it using the GitHub style"
}
```
