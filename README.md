# Machine Learning - An Enron Scandal POI Classifier

#### 1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

146 data points in total.

18 POI's and 128 non-POI's.

20 existing features.

One person named **Eugene Lockhart** (where the key is "LOCKHART EUGENE E") doesn't contain any non-NaN value, he is not a POI either. Since this dataset is already quite skewed toward the non-POI class, I consider this data point carries too little information and decide to disregard it. According to the "enron61702insiderpay.pdf", there is a payments made by Enron employees on account of business-related travel to **The Travel Agency in the Park**. Since it's not an actual person, this data point is removed as well.

On the other hand, every feature comes with missing values. `loan_advances` has the most missing values (97.26% of the values are missing), followed by `director_fees` (88.36%), `restricted_stock_deferred` (87.67%), and `deferral_payments` (73.29%).

I performed missing values imputation by filling financial NaNs to be 0. Because in this case, an NaN values suggests there's no payment for one specific category (column name) to the person, namely, zero payment.

One outlier **TOTAL** is removed because it's the total amount of payments of all the people, and it distorts the distribution enormously.

#### 2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

I created 4 new features.
1. A boolean field indicating whether this person has an email address or not, then transform the data type to float.
2. The ratio of emails received from POI's to total emails received.
3. The ratio of emails sent to POI's to total emails sent.
4. The ratio of emails shared receipt with POI's to total emails received.

The final features are chosen by `SelectKBest`, different numbers (K) are tested against my algorithms and 15 of them ended up giving the optimal results:

|Feature|Score|
|---|---|
|exercised_stock_options| 24.815079733218202|
|total_stock_value| 24.182898678566875|
|bonus|20.792252047181531|
|salary| 18.289684043404495|
|**ratio_from_this_to_poi**| 16.409712548035785|
|deferred_income| 11.458476579280346|
|long_term_incentive| 9.9221860131898083|
|restricted_stock| 9.2128106219771002|
|**ratio_shared_receipt_with_poi**| 9.101268739193543|
|total_payments| 8.7727777300916703|
|shared_receipt_with_poi| 8.5894207316823685|
|loan_advances| 7.1840556582887256|
|**email_account**| 6.1069253327317909|
|expenses| 6.0941733106389364|
|from_poi_to_this_person| 5.2434497133749458|
|other| 4.1874775069953749|
|**ratio_from_poi_to_this**| 3.1280917481567343|
|from_this_person_to_poi| 2.3826121082276805|
|director_fees| 2.1263278020077054|
|to_messages| 1.6463411294420061|
|deferral_payments| 0.22461127473601086|
|from_messages| 0.16970094762175478|
|restricted_stock_deferred| 0.065499652909942599|

This is a full list of 23 features with influence scores from high to low. New features are in bold font.

Compare to the highest score of 24.82 ("exercised_stock_options"), and the lowest score of 0.07 ('from_messages'), 3 new features actually contribute considerable influences.

`StandardScale()` is performed before feature selection. Because I will be using SVM and regularized logistic regression, feature scaling is necessary for a robust performance.

#### 3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

Logistic regression works best for me. I also tried SVM, with moderate performance that overall evaluation metrics are 10% ~ 20% lower than logistic regression. The former gains optimal result with 15 best features while SVM performs best with 14 features, but still lower precision and recall.

Since this dataset is pretty small and skewed, I used `GridSerchCV` equipped with `StratifiedShuffleSplit` for algorithm selection.


#### 4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Without carefully tuning parameters, very poor performance (e.g., SVM with RBF kernel for this dataset) can turn out, or code even breaks and doesn't produce any output (e.g., ill-calculated precision/recall). Choosing the right parameters helps the algorithm fit real-world data as tight as possible.

I used GridSerchCV equipped with StratifiedShuffleSplit for algorithm selection. kernel, C, gamma are tuned for SVM, while solver, C, max_iter are tuned for logistic regression. The results were compared simultaneously.

#### 5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is performed to estimate how the classifier is trained, help tuning the parameters for optimal performance, which is independent from the dataset that's already been trained.

Not properly validated algorithm can easily overfit, resulting much worse performance on the test data than the training data.

I used 100-fold stratified split cross-validation when tuning the parameters for my models.

#### 6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

I focused on precision and recall scores when evaluating my algorithms. Mostly they both fall between 0.3 ~ 0.4.

A good precision means, whenever a POI is flagged, it's highly possible that it's a real POI rather than a false alarm. A good recall means everytime a POI shows up, we're able to identify it.
