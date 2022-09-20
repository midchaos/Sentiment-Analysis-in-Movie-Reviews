# Sentiment-Analysis-in-Movie-Reviews

Preprocessing steps

Removal of noise

This step involved the removal of any HTML tags and square braces as it was seen all the input data was read as a list and contained several <br> tags for new paragraphs. Such text is noise for us and does not aid in our prediction of the sentiment of the review.

Removal of special characters

Special characters such as punctuation marks, brackets and other non-alphanumeric characters hold no importance for our model either. Since we are working with English language evaluations, we must ensure that any special characters are removed.

Removal of affixes from the words (Text lemmatization)

Text lemmatization is used to bring a word back to its base form if it has an affix. This helps us to get to the root of the word because that is what we are actually interested in. For instance, the root word for running is run. Hence, to sense the sentiment behind any word, all we need is its root word.

The main difference between text lemmatization and text stemming is that lemmatization takes into consideration the morphological analysis of the word while stemming works on the simple principle of cutting off common prefixes and suffixes. For example, the word studies becomes study when lemmatization is applied and studi when stemming is performed.

Therefore, we go with lemmatization in our case as it is not only more intuitive but keeps the original meaning of the words intact. 

Removal of stop words

Stopwords can be considered as the connecting or enhancing words which complete sentences. However, they provide no value when one is trying to figure out the sentiment of a sentence. Thus, we remove words such as ‘a’, ‘an’, ‘the’, ‘but’, etc.

Tokenize and add padding

Every sentence is broken up into smaller chunks to capture the words revealing the nature of the review. Furthermore, padding is added in order to make the inputs similar in shape and size.

Design choices

To build a solution for classifying a movie review as positive or negative, we would expect our machine learning model to persist its learning and be able to apply it to the next set of input. Recurrent neural networks immediately come to mind. However, there are certain limitations to using RNNs too. For example, if the gap between the present word and the last related word is very large, the network may not be able to predict the next word with respectable accuracy.

To learn such long term dependencies, we have Long Short Term Memory (LSTM) networks. They are able to pertain information for longer periods of time. This is achieved through a series of interactive neural network layers. Since we are trying to classify movies based on their reviews, we need to pick out certain words which show emotion and also understand the context in which they were used. Using a LSTM network seems wise in this case as it would be able to recall information and make an informed decision on the forthcoming words.

The architecture:



Embedding

The embedding layer converts all the words into a fixed length vector. All the words are thus represented in reduced dimensions as a dense vector with real values, instead of just 0’s and 1’s.

Dropout

To avoid overfitting, a dropout layer is added. Some of the less significant neurons are dropped which enhances the performance of the network. A dropout ratio of 0.2 is used.

LSTM

A LSTM layer with 100 units is added. The default activation function, tanh is used while sigmoid is used for the recurrent activation.

Dense

A fully-connected layer of neurons is added with ReLU (Rectified Linear Unit) activation function. ReLU outputs the input as it is if it is positive, otherwise 0.This is how it handles sparsity and the problem of vanishing gradient. Moreover, it does not activate all the neurons at the same time.

Dropout

Another dropout layer is added with a ratio of 0.2 to further improve the performance of the model.

Dense

The last layer of the model is chosen to be a dense layer with the sigmoid activation function. The output is between 0 and 1 which makes it more suitable for binary classification problems such as ours.

Further,

Binary cross-entropy is used as the loss function as we have only two target classes making it a binary classification.
Adam optimizer is used due to its training efficiency and ability to handle sparse gradients well.
Accuracy metric is used to track the accuracy at each epoch.

The training accuracy was found to be 90.67% as shown in the snapshot below:



The testing accuracy was reported to be 86.31% as seen in the following screenshot:




Comments on the output:
When the rating column was given as an input along with the reviews, the model seemed to overtrain on the input data, giving a training accuracy score of 99%. This can be attributed to the fact that the rating column already provided an intuition on the review and thus, would have been given the highest weightage. We can consider this as a bias because the model does not really learn on words alone because there is a direct relation between rating and the target label.
Moreover, when the input data is scaled and fed to the machine learning model, it was observed that the accuracy decreased to around 50%. One of the main reasons behind this is that when the reviews are tokenized and padded, each word is already assigned a number based on its importance in the review.
