# CS521: Statistical Natural Language Processing Project
<hr/>

### Team Members -
Name: Samihan Nandedkar (667142409) <br/>
Name: Sambhav Jain (657939710)
<hr/>
<br/>
<br/>

# Problem Statement:
THe aim of this project is to develop a robust sentiment analysis system that can accurately analyze the sentiment of Hindi-English code-mixed text. The system needs to overcome several challenges, including the presence of code-mixed language, language ambiguity, and lack of labeled data for training. The system should be able to handle variations in the use of Hindi and English languages, such as different scripts (Devanagari and Roman), grammatical structures, and transliterated words.


## Motivation
The major driving factors to pursue this project are as follows
- India is a linguistically diverse country with a large population that uses both Hindi and English languages extensively in their day-to-day communication. Code-mixed text is a common way for people to convey their thoughts and feelings, which is indicative of India's dynamic language use. In order to meet the multilingual demands of the Indian population, businesses, governments, and scholars can benefit from knowing the mood of Hindi-English code-mixed text.
- Social media platforms are frequently utilized to convey ideas, feelings, and sentiments through text that has been code-mixed. Monitoring social media, analyzing trends, and tracking user sentiment can all benefit from analyzing the sentiment of code-mixed text. Understanding user feelings in other types of user-generated material, such reviews, comments, and feedback, can also be helpful.
- Sentiment analysis on Hindi-English code-mixed text can help businesses and brands monitor the sentiment of their products, services, or brand reputation among the Hindi-English bilingual audience. It can offer insights into client thoughts, grievances, and feedback, assisting businesses in making wise choices to enhance their products and customer satisfaction.

## Our Proposal & Previous Research

#### What has been done till date?

Previous work focused on limited-sized datasets, with similar and limited data preprocessing. The models used in previous work were LSTM, SVM, Naive Bayesian Classifier, MLP, and BERT.

#### Our Novel Contributions

We have achieved the following novelty in our work:

Accumulated larger and distinct datasets, such as BHAAV (Hindi text corpus for analyzing emotions), in addition to the SentiMix Dataset from SemEval Shared Task.
Implemented and compared multiple models, including BERT, IndicBERT, Roberta, and LSTM, to identify the best-suited model for sentiment analysis of code-mixed tweets.
Introduced additional preprocessing steps to handle code-mixed data effectively and improve model performance.

#### Contribution as per Claims in Project Proposal

We have successfully contributed to all aspects mentioned in our project proposal. We incorporated a larger and distinct dataset, trained and tested different models, and implemented additional steps in preprocessing to improve model performance in sentiment analysis of code-mixed tweets.


## Dataset

- The main code-mixed dataset is Sentimix created by [Patwa et al.](https://arxiv.org/pdf/2008.04277v1.pdf) containing 20k mannually annotated Code-mixed tweets.
- The distribution of the dataset can be seen in the chart shown below

![Data Distribution](images/data_distribution.png)

- Additionally the dataset is evenly balanced with following distribution.

![Label Distribution](images/label_distribution.svg)

- There is a requirement to restructure and clean the data so that models can trained effectively on the data.

- The dataset size was increased by adding manually annotated hindi text which was taken from [Kumar Y et al.](https://arxiv.org/ftp/arxiv/papers/1910/1910.04073.pdf) and annotated english tweets whose number equated to total number of added

- The pre-processing for each dataset file is performed using the `Basic_Data_Preprocessing.ipynb` script and dumped data/final_data/ 

The raw & preprocessed data are placed in the `data` directory of the repository.

## Methodology


![Steps](images/methodology.png)


### Data Gathering:

- The first step in gathering all the required datasets was to identify the ones which can most aptly be used in our problem.
- In order to increase the size of the data, additional Hindi and English data was be collected and incorporated into the dataset.
- We took BHAAV, a manually labeled corpus of hindi text to increase the size of dataset. 

### Dataset Restructure & Build

- The sentimix dataset was very crudely structured which would have required additional preprocessing before using it in code. 
- To avoid the redundancy of code, we performed additional step of data restructuring to data into a csv.
- This allowed us to directly use pandas dataframe to import the csv and perform necessary preprocessing tasks on the data.

The data restructuring  performed using `textToCsv.py` script

- Additionally to increase the size of dataset, BHAAV text corpus is used. However as the BHAAV dataset had text in Devnagiri script. Using python module `indic-transliteration`, the transliteration for the hindi text is obtained.
- The transliteration of hindi text & their labels are exported into a csv file with similar structure to that of other datasets.

The transliteration of hindi text is performed using `hinToEng.py` script

### Dataset Cleanup

- The first step in dataset cleanup is to validate the data to ensure it is accurate, complete, and consistent.
- The checking for missing values, incorrect formats, inconsistent data types, and other data integrity issues.
- This task includes removing the emoji, special characters from all the datasets.
- This task is perform apart from all the other preprocessing task as initially the tokens are labelled which makes it easier drop the incorrect data entries.

### Preprocess Data

- These steps were performed in preprocessing the data

1. Lowercase the text
2. Remove Null values
3. Stemming
4. Lemmetization
5. Removing English & Hindi stopwords
6. Remove username (words with numbers)
7. Vectorize the data

### Training the model

We have trained following models and evaluated the results.

1. LogisticRegression

Logistic Regression is a simple yet effective linear model for binary classification tasks. For sentiment analysis of code-mixed tweets, the model is trained using features extracted from the preprocessed text data. Logistic Regression can handle high-dimensional sparse data and is relatively faster to train compared to more complex models. We experimented with different feature extraction techniques, such as TF-IDF and word embeddings, to improve the model's performance on our dataset.

2. Decision Tree

Decision Tree is a non-linear model that recursively splits the dataset based on the most informative features. In the context of sentiment analysis of code-mixed tweets, Decision Tree is trained on features extracted from the preprocessed text data. This model offers interpretability, as the decision-making process can be visualized in the form of a tree. We experimented with various tree depths and splitting criteria to optimize the model's performance on our dataset.

3. Bidirectional LSTM

Bidirectional LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that can capture long-range dependencies in text data. For sentiment analysis of code-mixed tweets, the Bidirectional LSTM model processes the input text in both forward and backward directions to capture context from both past and future words. The model is trained on preprocessed text data using word embeddings as input features. We experimented with different hyperparameters, such as the number of hidden units, dropout rates, and learning rates, to optimize the model's performance on our dataset.

4. BERT

The model is trained for a specified number of epochs. The best value for the number of epochs should be determined based on the model's performance on the validation set. The model should not be overfitted to the training set, and a balance between underfitting and overfitting should be maintained.

5. Sequential

A Sequential model in the context of sentiment analysis of code-mixed tweets refers to a simple feedforward neural network architecture. The Sequential model is composed of a linear stack of layers, including input, hidden, and output layers. Each layer is fully connected to the next one, with activation functions applied in between. The model is trained on preprocessed text data using features such as word embeddings or other feature extraction techniques. We experimented with different layer sizes, activation functions, and optimization algorithms to optimize the model's performance on our dataset.

6. IndicBERT

IndicBERT is specifically designed to handle Indian languages, making it suitable for code-mixed tweets involving these languages. The model training process involves several steps, including data preprocessing, tokenization, and training loop setup. The IndicBERT model is fine-tuned on the pExperiment with different hyperparameter values for the learning rate, batch size, and number of epochs to optimize the model's performance on your dataset. rovided code-mixed tweets dataset for sentiment analysis. 


### Evaluate

The model's performance is evaluated using accuracy and F1-score. The test accuracy, test loss, and F1-score for the test dataset are reported after training the model.


## Steps to run the project:

### Required Modules:
```
pandas
numpy
Scikit-learn
TQDM
PyTorch 
Transformers (from Hugging Face)
nltk
tensorflow

```

### Steps

1. All the raw data files are present in the data/raw/. -> Sentimix, Bhaav
2. Perform the data restructuring on the raw data using `testToCsv.py` file with updating the filename in the script
3. Perform the transliteration on the Bhaav database using `hinToEng.ipynb` file and Bhaav database.
4. Perform the data cleanup on each of created csv using `Basic_Data_Preprocessing.ipynb` script.

Note: Already cleaned data is present in data/final/ directory. These files can be directly used and the steps 1-4 can be skipped.

5. Train & Evaluate LogisticRegression & DecisionTree models using the `basic_models.ipynb` script.

Nodt: All the script from Step 6 onwards will require a GPU enabled system to run as the models require cuda to train.
6. Train & Evaluate Bidirectional LSTM & Sequential models using `lstm_seq.ipynb`script.
7. Train & Evaluate BERT model using 
9. Train & Evaluate IndicBERT model using 



## References
