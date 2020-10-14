# BiSOMs
Unsupervised distributed connectionist model of developmental bilingual speech processing using 2 SOMs connected via Hebbian links. Based on prior work by Li and Farkas (2002). The intention was to see if I could replicate their results with the least effort (e.g. using Python libraries). The training data was the Hong Kong Bilingual Corpus (which included Cantonese and English child-directed speech and code-switching) from the CHILDES dataset. 

Trains 2 SOMs, one for lexical/phonological representations and one for semantic, and a text file containing the coordinates for 400 most frequent words for each map. 
