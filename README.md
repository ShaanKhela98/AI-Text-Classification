# AI-Text-Classification
Implementation of machine learning algorithms to process text data.

To run the program: python3 classify.py farm-ads-train farm-ads-test "function"

Functions:
          
   tf - uses the training data to determine term frequencies for each word in the training set
   
   ![Term Frequency](/tf.png)
       
          
   tfgrep - classifies a document based on the most discriminating term
   
   ![Term Frequency classifying most discriminating](/mnb.png)
   
   
   priors - computes the class and classifies a document using 0-R             
   
   ![0-R Classification](/priors.png)

          
   mnb - implements the multinomial naive bayes model to predict the most likely class
       
   ![Multinomial naive bayes matrix](/mnb.png)
          
   
   df - computes the document frequency for each term in the training set
   
   ![Document Frequency](/df.png)

   
   nb - implements the multivariate Bernoulli model to predict the most likely class. 
   ![Mulivariate Bernoulli model](/nb.png)
  
   
   mine - modified mnb function to limit complexity of model by recalculating term frequency, eliminate 50/50 split results, and construct confusion matrix  
   ![Personal function](/mine.png)

