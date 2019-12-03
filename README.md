# AI-Text-Classification
Implementation of machine learning algorithms to process text data.

To run the program: python3 classify.py farm-ads-train farm-ads-test "function"

Functions:
          
          
          tf - uses the training data to determine term frequencies for each word in the training set
          
          tfgrep - classifies a document based on the most discriminating term
          
          priors - computes the class priors for the training set and then classifies a document using 0-R             
          
          mnb - implements the multinomial naive bayes model to predict the most likely class
          
          ![Multinomial naive bayes matrix](/mnb.png)
          
          df - computes the document frequency for each termin the training set
          
          nb - implements the multivariate Bernoulli model to predict the most likely class. 
          
          mine - modified mnb function to limit complexity of model by recalculating term frequency, eliminate 50/50 split results, and construct confusion matrix  
