
## Pursuit Of Happiness

## Introduction

  The overall goal of the CL-Aff Shared Task is to understand what makes people happy, and the factors contributing towards such happy moments. Related work has centered around understanding and building lexicons that focus on emotional expressions [5,9], while Reed et al. learn lexico-functional linguistic patterns as reliable predictors for first-person affect, and constructed a First-Person Sentiment Corpus of positive and negative first-person sentences from blog journal entries. Wu et al. propose a synthetic categorization of different sources for well-being and happiness targeting the private micro-blogs in Echo, where users rate their daily events from 1 to 9. These work aim to identify specific compositional semantics that characterize the sentiment of events, and attempt to model happiness at a higher level of generalization, however finding generic characteristics for modeling well-being remains challenging. In this project, we aim to find generic characteristics shared between different affective  classification tasks. Our approach is to compare state-of-the-art methods for linguistic modeling to prior lexicons’ predictive power. While this body of work is broader in scope than the goals we are trying to address, they do include annotated sets of words associated with happiness as well as additional categories of psychological significance.

  The aim of this work is to address the two tasks that are part of the CL-Aff Shared Task. The data provided for this task comes from the HappyDB dataset [1]. Task 1 focuses on binary prediction of two different labels, social and agency. The intention is to understand the context surrounding happy moments and potentially find factors associated with these two labels. Task 2 is fairly open-ended, leaving it to the participant’s imagination to model happiness and derive insights from their models. Here, we predict the concepts label using multi-class classification. We explore various approaches to determine which models work best to characterize contextual aspects of happy moments. Though the predictions of agency and social sound simpler than concepts, we expect that the best models for agency and social prediction could generate similarly optimal performance for concepts, assuming that the classes of social, agency, and concepts share common characteristics. To validate our assumptions, we build different models for general affective classification tasks and then try to gain a deeper understanding of the characteristics of happy moments by interpreting such models with the Riloff’s Autoslog linguistic-pattern learner 




## Dataset

  HappyDB is a dataset of about 100,000 ‘happy moments’ crowd-sourced via Amazons Mechanical Turk where each worker was asked to describe in a complete sentence “what made them happy in the past 24 hours”. Each user was asked to describe three such moments. In particular, we exploit the agency and sociality annotations provided on the dataset as part of the recent CL-Aff shared task 2, associated with the AAAI-19 workshop of affective content analysis 3. For this particular shared task, 10,560 moments are labelled for agency and sociality and were available as labeled training data. 4 Then, there were 17,215
moments used as test data. Test labels were not released and teams were expected to submit the predictions based on their systems on the test split. For our models, we split the labeled data into 80% training set (8,448 moments) and 20% development set (2112 moments). We train our models on train and tune parameters on dev. For our system runs, we submit labels from the models trained only on the 8,448 training data points. The distribution of labeled data is as follows: agency (‘yes’=7,796; ‘no’= 2,764), sociality (‘yes’=5,625; ‘no’= 4,935).



## Methodology

#### Model 1:
We started using the Profile features [Meta-data of a happy moment such as age, category, gender etc., etc.,] without considering the actual moment. This model is an attempt to understand if everything conveyed in the moment has been captured and would it suffice to predict the accuracy of social & agency. If the profile features are sufficient to predict social & agencies then we may get rid of the complexities involved in text understanding & rather focus on generating/capturing the profile features.

| Preprocessing \(50 dimensions\)  | LABEL ENCODING  
| Features \(100 dimensions\) | Profile Features  
| Classifier \(100 dimensions\)    | XGBoost  
| Social Accuracy \(100 dimensions\)  |  61.2%  
| Agency Accuracy \(100 dimensions\)  | 72.83%  

#### Conclusion & Insights:
  Profile features alone are not sufficient to build the model & we need to consider the moments data. However, profile features may act well to support the model built on moments data. We may use, model built on profile features as a complementary model i.e. an ensemble model can be built with profile features model as one of them.


#### Model 2
  Text (moment) based feature engineering. As an exploratory analysis we initially adopted one of the approach suggested by one of the state of the art papers i.e. to use 4grams to build features. We do not have any intuition behind the same, however did this as an exploratory learning experiment. Below are the results wrt the same.


| Model                                | Preprocessing | Features | Feature \-Representation | Social | Agency |
|--------------------------------------|---------------|------------------------------------|--------|--------|
| Naive    Bayes \(50 dimensions\)     | 0\.59         | 0\.43    |                         | 0\.70  |        |
| Logistic Regres. \(100 dimensions\)  | 0\.50         | 0\.39    |                         | 0\.62  |        |
| XGBoost Forest \(100 dimensions\)    | 0\.45         | 0\.36    |                         | 0\.58  |        |
| Neural Network   \(100 dimensions\)  | 0\.45         | 0\.36    |                         | 0\.58  |        |










![lstm_internal](imgs/LSTM_onTheInside.png)

#### RNN cell

![rnn_internal](imgs/RNN_onTheInside.png)

### Binary Classification

For binary classification, we used a combination of recurrent networks followed by a softmax classifier. The classifier design was not considered more intricate due to the constraints on the data as mentioned above.

A recurrent network can be either an RNN (recurrent neural network) or an LSTM (long short term memory). Recurrent models are used in this project as they capture features of the previous cell as well as the current input, weighted on a non-linearity, usually a tanh function. Here we use simple many-to-one recurrent model of size 100 dimensions. The difference in performance for RNNs and LSTMs comes from the fact that LSTMs have three gates which determine what information should be retained from the previous hidden states and what information should be discarded. LSTMs are preferred over RNNs in order to solve the vanishing gradient problem.

We run four experiments, a single LSTM of 50 and 100 dimensions, an two layer RNN of 100 dimensions and a two layer LSTM of 100 dimensions. The results of the experiments are given in the section below.

### Multiclass Classification Model

For multi-class classification, we use a slightly more complicated model of a stacked LSTM. A stacked LSTM has multiple sequences of LSTMs in a stack, such that for the second layer onwards, the input is not the embedding, but the hidden state of the previous layer. The diagram below shows this configuration of stacked LSTMs.

#### Architecture

![architecture](imgs/Model.png)

## Results and Analysis

### Binary Classification

We can see in table below where the precision, recall and F1 values of the binary classification experiment are provided. We see two important observations here. First, we see that the lower dimension single layer LSTM performs the best despite being the simplest model.

This is for two main reasons, which are as follows:

* The number of data points on training are quite few, and the ratio of positive to negative samples are quite skewed. This causes larger models to overfit, and because of that the larger the model in dimension size, the worse it performs.
* The data is skewed in more than one way. The comments which are sexist stereotypes tend to be much longer than those which are not sexist, specifically because the instagram scraping methodology only allows for scraping based on hashtags. 


| Model                                | Recall | Precision | F1 \-Score | Accuracy |
|--------------------------------------|--------|-----------|------------|----------|
| Single Layer LSTM \(50 dimensions\)  | 0\.59  | 0\.43     | 0\.49      | 0\.70    |
| Single Layer LSTM \(100 dimensions\) | 0\.50  | 0\.39     | 0\.44      | 0\.62    |
| Two Layer LSTM \(100 dimensions\)    | 0\.45  | 0\.36     | 0\.41      | 0\.58    |
| Single Layer RNN \(100 dimensions\)  | 0\.43  | 0\.37     | 0\.39      | 0\.57    |

We show the graphs of precision, recall, accuracy and F1-score of the binary classification experiment below. The effect of data overfitting is seen almost immediately. Further, note that sparsity and skew in the dataset requires better training data. Higher accuracies may be achieved by working with the better data.

![](results/sns_classfication/train_loss_allsns.png)

Recall            |  Precision
:-------------------------:|:-------------------------:
![](results/sns_classfication/rec_all_sns.png) | ![](results/sns_classfication/prec_all_sns.png)
Accuracy            |  F1-score
![](results/sns_classfication/acc_all_sns.png) | ![](results/sns_classfication/f1-score-sns.png)


### Mutli-class Classification

The table below shows the results of the multiclass classification experiment. Again here we see that the simplest model performs the best. We also see that using a stacked LSTM shows a slight increase in performance, but the model runs the risk of overfitting.


| Model                                | Recall | Precision | F1 \-Score | Accuracy |
|--------------------------------------|--------|-----------|------------|----------|
| Single Layer LSTM \(50 dimensions\)  | 0\.55  | 0\.49     | 0\.518     | 0\.605   |
| Single Layer LSTM \(100 dimensions\) | 0\.51  | 0\.44     | 0\.472     | 0\.535   |
| Two Layer LSTM \(100 dimensions\)    | 0\.484 | 0\.443    | 0\.462     | 0\.513   |


We also show the loss values of each of the models, binary and multi-class classification. We see that while the loss falls most quickly for the model that stabilizes quickest, based on which the local minima is achieved. While the local minima is not the best performing, the model learns certain charcteristics of the data, such as the use of certain terms, length of caption or comment and so on.

![](results/multiple/mutiple_train_loss_all.png)

Recall            |  Precision
:-------------------------:|:-------------------------:
![](results/multiple/multi_rec_all_sns.png) | ![](results/multiple/multi_prec_all_sns.png)
Accuracy            |  F1-score
![](results/multiple/mutlti_acc_all_sns.png) | ![](results/multiple/mutiple_f1_all.png)

## Conclusion

In this project, we performed a study into the classification of Instagram captions and comments. We first annotated the data using a set of well formed guidelines. The deprecating API provided by Instagram inhibits the process the scraping the data off the site and they delete any comments or posts that are reported within a short period. This lead to a small number of sexist posts in our dataset to start with.

With this dataset, we started off with a manual annotation of small number of posts, and using this seed data, we then used an active learning classifier in order to classify a large number of captions and comments. We cross verified the tags to see if the tags were right.

We then experimented with different classifiers, where an LSTM classifier with only $50$ hidden layer dimension performed the best compared to higher dimension, or multi-layer classifier, even though we made sure that the training set had equal distribution between the 2 classes, for both binary and hierarchical classifier. This can be attributed to the skewed dataset that we have for this task. Further work over here would be to expand the dataset to include more sexist posts/captions.

The further work in this would be first to better the dataset by including more sexist captions. We can also identify more classes in the sexist types in the dataset, such as slut shaming, mansplaining, etc. The next step would be to experiment with other classifiers like Bi-LSTM, CNN, CNN-biLSTM-Attention, Hierarchical-biLSTM-Attention, and BERT, and with GloVe Twitter embedding along with GloVe Wikipedia.

## Complete Report

The complete report can be found [here](https://drive.google.com/file/d/1ioXSm3dWoSF00Z3TjhdgCie4yebOwjra/view?usp=sharing).

## Video Presentation

[![IMAGE ALT TEXT](http://img.youtube.com/vi/okd5UwopDJE/0.jpg)](http://www.youtube.com/watch?v=okd5UwopDJE "Video Title")



