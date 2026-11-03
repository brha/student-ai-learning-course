# BRHA: lidt ekstra til seneste lektion

## Ekstra 2026-10-20

- Se mindst en video på https://www.youtube.com/playlist?list=PLE3ovFxnzNpblbg9cylEiOnctg0D24rKz
- Se gennem listen af videoer for at se om der kan være interessant at se noget på et senere tidspunkt
- Se også på listen af videoer her: https://www.youtube.com/@DeepLearningAI

# Week 2: Machine learning basics

## Meeting 3

Lesson 7: Supervised learning introduction
- Goal: Explain classification and regression 
- Activity: Use, understand and modify spam_detector_starter.py
- Activity: Use, understand and modify house_price_predictor.py
- Add more messages to the training data
- Try a different model from scikit-learn
- Look at https://pypi.org/project/scikit-learn/
- Look at https://scikit-learn.org/stable/

Lesson 8: Data preprocessing
- Goal: Teach cleaning and feature preparation
- Topics: missing values, text cleaning, train/test split
- Activity: Use, understand and modify preperation.py
  - The code teaches:
    - how to identify missing values
    - how to clean a dataset
    - how to convert categories to numbers
    - how to prepare features for machine learning
    - how to train a simple regression model
  - Modify some of these
    - Change the dataset
      - add more students
      - add new features like sleep_hours, homework_done, or motivation
    - Change the target
      - instead of final_score, predict exam_grade, hours_needed, or future_score
    - Try different missing value strategies
      - replace missing values with mean
      - replace with median
      - remove rows with missing values
    - Use a different feature encoding
      - map categories manually
      - use pandas.get_dummies()

Lesson 9: Intro to scikit-learn
- Goal: Show a complete ML workflow and learn how to use scikit-learn
- Activity: Train a logistic regression model. Use explore_scikit-learn.py
  - The code teaches
    - practice using the scikit-learn documentation
    - where to find the right classifier
    - to look up functions and parameters
  - Good things to find on the scikit-learn site:
    - what load_iris does
    - what train_test_split does
    - what LogisticRegression does
    - what max_iter does
    - what accuracy_score does
    - how to read the docstrings and examples
  - Assignments you can do:
    - Replace LogisticRegression with another model they find on the scikit-learn site
    - Change test_size from 0.25 to 0.2 or 0.3
    - Try a different dataset from scikit-learn
    - Read the docs and explain what random_state does
    - Look up how to print the model coefficients or probabilities
  - Examples of models you can search for:
    - DecisionTreeClassifier
    - KNeighborsClassifier
    - RandomForestClassifier
    - SVC
    - GaussianNB




## Meeting 4

```python
#Denne springes over: Lesson 10: Build a simple spam detector or sentiment model 
#- Goal: Use text data for classification*
#- Activity: Create a tiny dataset and train a model*
```

Lesson 11: Model evaluation
- Goal: Explain accuracy and confusion matrix
- Activity: Discuss what bad accuracy means
  - Use, understand and modify accuracy_and_confusion.py
  - Read the pages:
    - Accuracy score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    - Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    - Model evaluation overview: https://scikit-learn.org/stable/modules/model_evaluation.html
  - What happens, when the data are good and when they are bad?


Lesson 12: Code lab
- Goal: Test your own ideas
- Activity: Tune parameters and compare results
  - Use and modify own_idea_starter.py
  - Read the nessesary docs in the links
  - Answer the questions in the code as comments, like this: # Answer: ...
  - Modify several of these
    - Change the dataset
    - Add more features
    - Remove a feature
    - Change test_size
    - Change random_state
    - Replace LogisticRegression with:
      - DecisionTreeClassifier
      - RandomForestClassifier
      - KNeighborsClassifier
      - SVC
    - Try a different problem:
      - spam or not spam
      - pass or fail
      - cat or dog
      - happy or sad review
    - Add more rows and check whether the model improves
  - Upload your python-file to Studie+




# Week 2 Starter Code

This folder contains machine learning examples for Week 2:
- spam_detector_starter.py
- house_price_predictor.py
- preperation.py
- explore_scikit-learn.py
- accuracy_and_confusion.py
- own_idea_starter.py

