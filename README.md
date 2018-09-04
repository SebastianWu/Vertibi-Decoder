# Vertibi Decoder
Train the WSJ_02-21, Implement Vertibi Decoder to generate POS-tag for sentences.

Author: Yuanxu Wu

## Prerequisite: 

python 3.6.4

## Predict POS-tag for unknown word

I implemented predicting the pos-tag of unknown word using a morphology method . 

In predicting unknown words’ pos-tag section, I generate six cases:  

•	Word starts with a particular prefix  
•	Word ends with a particular suffix  
•	Word first letter is capital  
•	Word contains a hyphen  
•	Word is all upper case  
•	Word is upper case, and has a digit and a dash  


## There are five input value you can change:  

  •	train_file_name = "WSJ_02-21.pos" # training file name  
  •	test_file_name = "WSJ_24.words" # testing file name  
  •	generate_pos_tag_file_name = "wsj_24.pos" # generate pos-tag file name  
  •	allow_to_use_pickled_data = 1 # allow to use pickled data or not  
  •	using_additional_training_file = 0 # add other training file or not  
  •	addition_training_file_name = "WSJ_24.pos" # additional training file name  

## How to compile and run the code

python3 MyVertibiDecoder.py
