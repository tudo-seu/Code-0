### Authors: Code0
### Names: Karl Johannes, Hannes Schmidt, Ahmed Barakaa

### Problem:
- Classify time series machine sensor data into one of four categories. These categories are "Non-production", "Power-up", "Power-down", and "Production". The given training data contains a fifth category, "unclassified", hence we are working with a semi-labeled dataset, where the task is to classify the "unclassified" datapoints into one of the four given power consumption classes.

### Our Solution:
1) We started by splitting our dataset into a training set and a holdout set.
2) Then we visualized the newly obtained sub-training set and examined the five key data statistics for all machine sensor features.
3) Data cleaning was done by correctly converting the time feature in combination with all machine sensor features, resampling at a lower frequency (each hour instead of every 15 minutes), feature standardization, and trimming the extreme 5% from each tail (keeping only the central 90% of the data) since the visualizations showed quite a few outliers and process experts told us that these were simply incorrect sensor recordings.
4) We experimented with one supervised approach, one unsupervised approach, and shortly with one semi-supervised approach:
    - a) The supervised approach trained a simple Random Forest Classifier on the cleaned data. Other tested models include HIVE-COTE ensemble with Random Interval Feature Extractor preprocessing.
    - b) The unsupervised approach used kMeans clustering. Interestingly, using 6 instead of 4 clusters performed better on the given 4 classes; our final unsupervised approach resulted in having 2 clusters for "Production" and "Non-production" rather than a one-to-one mapping for every class.
    - c) Our semi-supervised approach utilizing LabelPropagation with a knn kernel unfortunately did not work in time before the submission deadline.

### Our obtained scores vs 1'st place scores
| Unseen Test Dataset | iamai - 1'st place   | Code0 - 2'nd place (Our Team) |
|---------------------|----------------------|-------------------------|
| Body_AHU            | 0.8601190476190477   | 0.8257068452380952      |
| BS1                 | 0.8980392156862745   | 0.8723039215686275      |
| Feeder F06          | 0.6264880952380952   | 0.8541666666666666      |
| Feeder F08          | 0.8768601190476191   | 0.8656994047619048      |
