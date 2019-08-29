Here I implemented a supervised model that uses a development set to learn  
different language models and compares them according to their perplexity on a test set.
I Implemented a Unigram language model wich compares 2 Lidstone and and HeldOut smoothing methods.
To compare between those methods I used the perplexity measurement. 

**Input:**
The script accepts the following 4 arguments in this exact order:  
< development set filename > < test set filename > < INP UT WORD > < output filename >  
INPUT WORD is a simple string representing either a seen or unseen word (such as ’the’,  
’honduras’, ’???’, etc.).  

Development set (develop.txt) and test set (test.txt) files are included in the data set provided. They are derived from a Reuters corpus, known as Reuters-21578 (since  
it contains 21,578 articles). They contain articles and their topics from a list of 9 topics (see  topics.txt for that list). 
I am assuming that the vocabulary size is 300,000.

**Output:**
An output file which describes and compares the elements of those smoothing methods.