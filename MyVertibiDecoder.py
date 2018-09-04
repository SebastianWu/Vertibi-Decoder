import pickle # to store trained data
import os # to check if a file exists

train_file_name = "WSJ_02-21.pos"
test_file_name = "WSJ_23.words"

generate_pos_tag_file_name = "wsj_23.pos"
allow_to_use_pickled_data = 0 # allow to use pickled data or not

using_additional_training_file = 1 # add other training file or not
addition_training_file_name = "WSJ_24.pos"

# class to store transfer dict and emit dict
class tag_state:
    def __init__(self):
        self.transfer_dict = {}
        self.emit_dict = {} 

# check do we already trained and stored in pickle
flag_trained = 0
if os.path.exists("states_dict.pickle"):
	flag_trained +=1
	states_dict = pickle.load(( open( "states_dict.pickle", "rb" ) ))

if os.path.exists("states_number_dict.pickle"):
	flag_trained +=1
	states_num_dict = pickle.load(( open( "states_number_dict.pickle", "rb" ) ))

if os.path.exists("tag_set.pickle"):
	flag_trained +=1
	tag_set = pickle.load(( open( "tag_set.pickle", "rb" ) ))

if allow_to_use_pickled_data == 0:
	flag_trained = 0
if using_additional_training_file == 1:
	flag_trained = 0

if flag_trained == 3:
	print("using last time trained data from pickle")

else:
	if using_additional_training_file == 0:
		print("Training corpus: "+train_file_name)
	else:
		print("Training corpus: "+train_file_name+", "+addition_training_file_name)
	#open training file and achieve the data
	training_file = open(train_file_name,"r")
	word_list = []
	tag_list = []
	tag_set = {}
	tag_set = set()
	for line in training_file:
	    if(line is not "\n"):
	        word, tag = line.split("\t")
	        tag = tag.rstrip("\n")
	        tag_set.add(tag)
	    else:
	        word = ""
	        tag = "__end__"
	    word_list.append(word)
	    tag_list.append(tag)
	training_file.close()

	if using_additional_training_file == 1:
		addition_training_file = open(addition_training_file_name)
		for line in addition_training_file:
		    if(line is not "\n"):
		        word, tag = line.split("\t")
		        tag = tag.rstrip("\n")
		        tag_set.add(tag)
		    else:
		        word = ""
		        tag = "__end__"
		    word_list.append(word)
		    tag_list.append(tag)
		addition_training_file.close()

	# function to generate tranfer dict and emit dict for each tag
	def create_state_dict(state):
	    arc_list = []
	    emit_list = []
	    word_set = {}
	    word_set = set()
	    for i in range(len(tag_list)):
	        if tag_list[i] == state:
	            arc_list.append(tag_list[i+1])
	            emit_list.append(word_list[i])
	            word_set.add(word_list[i]) 
	    arc_dict = {}
	    emit_dict = {}
	    for tag in tag_set:
	        arc_dict[tag] = arc_list.count(tag)
	    arc_dict["__end__"] = arc_list.count("__end__")
	    for word in word_set:
	        emit_dict[word] = emit_list.count(word)
	    sum_arc = sum(arc_dict.values())
	    sum_emit = sum(emit_dict.values())
	    for i in arc_dict:
	        arc_dict[i] = float(arc_dict[i])
	    for i in emit_dict:
	        emit_dict[i] = float(emit_dict[i])    
	    return [arc_dict, emit_dict]

	# test create_state_dict function
	#arc_d, emit_d = create_state_dict(tag_list[2])
	#print(sum(arc_d.values()), sum(emit_d.values()))
	#for k, v in arc_d.items():
	    #print(k, v)


	## start to train from training data
	print("Start to Training:")
	# create state dictionary for all tags
	states_dict = {}
	# add start state to states dict
	def create_start_state():
	    arc_list = []
	    arc_list.append(tag_list[0])
	    for i in range(len(tag_list)-1):
	        if tag_list[i] == "__end__":
	            arc_list.append(tag_list[i+1])
	    arc_dict = {}
	    for tag in tag_set:
	        arc_dict[tag] = arc_list.count(tag)
	    arc_dict["__end__"] = arc_list.count("__end__")
	    sum_arc = sum(arc_dict.values())
	    for i in arc_dict:
	        arc_dict[i] = float(arc_dict[i])
	    return arc_dict
	start = tag_state()
	start.transfer_dict = create_start_state()
	start.emit_dict = {}
	states_dict["start"] = start

	# add end state to states dict
	end = tag_state()
	end.transfer_dict = {}
	end.emit_dict = {}
	states_dict["__end__"] = end

	# add other state to states dict
	for tag in tag_set:
	    s = tag_state()
	    s.transfer_dict, s.emit_dict = create_state_dict(tag)   
	    states_dict[tag] = s

	print("Finish creating tranfer dictionary, emit dictionary for each tag.")

	#save states number dictionary
	states_num_dict = states_dict.copy()

	#convert states_dict to probability dictionary
	for state, v in states_dict.items():
	    if v.transfer_dict:
	        sum_t = sum(v.transfer_dict.values())
	        for k in v.transfer_dict.keys():
	            v.transfer_dict[k] = (float)(v.transfer_dict[k]/sum_t)
	    if v.emit_dict:
	        sum_e = sum(v.emit_dict.values())
	        for k in v.emit_dict.keys():
	            v.emit_dict[k] = (float)(v.emit_dict[k]/sum_e)

	#store state_dict into pickle
	pickle.dump( states_dict, open( "states_dict.pickle", "wb" ) )
	pickle.dump( states_num_dict, open( "states_number_dict.pickle", "wb"))
	pickle.dump( tag_set, open( "tag_set.pickle", "wb"))

## using a morphology method to train unknown word
m_file = open("MyPrefixSuffix.txt")
prefixsuffix_list = []
for line in m_file:
    prefixsuffix_list.append(line.rstrip("\n")) 

# train unknown word using prefix and suffix.  
prefix_suffix_tag_dict = {}
for suffix in prefixsuffix_list:
    prefix_suffix_tag_dict[suffix] = {}
    #initailization
    for tag in tag_set:
        prefix_suffix_tag_dict[suffix][tag] = 0
for tag in tag_set:
    s = states_dict[tag]
    for k, v in s.emit_dict.items():
        for fix in prefixsuffix_list:
            if fix[0] == "-":
                suffix = fix[1:]
                if k.endswith(suffix):
                    prefix_suffix_tag_dict[fix][tag] += v  
            if fix[-1:] == "-":
                prefix = fix[:-1]
                if k.startswith(prefix):
                    prefix_suffix_tag_dict[fix][tag] += v 
#change suffix_tag_dict to probability dictionary
for fix in prefixsuffix_list:
    sum_s = sum(prefix_suffix_tag_dict[fix].values())
    if sum_s == 0:
        del(prefix_suffix_tag_dict[fix])
        prefixsuffix_list.remove(fix)
    else:
        for tag in tag_set:
            prefix_suffix_tag_dict[fix][tag] = (float)(prefix_suffix_tag_dict[fix][tag]/sum_s)
prefix_suffix_set = set(prefixsuffix_list)

def is_number(s):
    s = s.replace(",","")
    s = s.replace(".","")
    s = s.replace(":","")
    s = s.replace("\/","")
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

# function to check word can be predicted or not
def check_word_can_be_predicted(word):
    flag_predicted = 0
    for fix in prefix_suffix_set:
        if fix[0] == "-":
            suffix = fix[1:]
            if word.endswith(suffix):
                flag_predicted = 1
        if fix[-1:] == "-":
            prefix = fix[:-1]
            if word.startswith(prefix):
                flag_predicted = 1
    
    if word[0].isupper():
        flag_predicted = 1
    if any(x == "-" for x in word):
        flag_predicted = 1
    if all(x.isupper() for x in word):
        flag_predicted = 1
    if is_number(word):
        flag_predicted = 1
    remove_digit = ''.join(i for i in word if not i.isdigit())
    remove_hypen = remove_digit.replace("_","")
    if all(x.isupper() for x in remove_hypen):
        flag_predicted = 1
    return flag_predicted

# generate guess unknown unpredictable word probability emit_dict
print("start to generate guess_unpredicatable_unknown_dict from trainin set")
unpredicatable_unknown_list = []
guess_unpredicatable_unknown_dict = {}
for state, v in states_num_dict.items():
    guess_unpredicatable_unknown_dict[state] = 0
    for k, c in v.emit_dict.items():
        if c < 10:
            if check_word_can_be_predicted(k) == 0:
                guess_unpredicatable_unknown_dict[state] += c

sum_g = sum(guess_unpredicatable_unknown_dict.values())
for tag in tag_set:
    guess_unpredicatable_unknown_dict[tag] = (float)(guess_unpredicatable_unknown_dict[tag]+1)/sum_g

print("Finish generating guess_unpredicatable_unknown_dict")
    
#word first letter is capital
capital_tag_dict = {}
for tag in tag_set:
    s = states_dict[tag]
    capital_tag_dict[tag] = 0
    for k, v in s.emit_dict.items():
        if k[0].isupper():
            capital_tag_dict[tag] += v
for tag in tag_set:
    capital_tag_dict[tag] = (float)(capital_tag_dict[tag]/sum(capital_tag_dict.values()))
    
#word contains a hyphen
one_hyphen_tag_dict = {}
for tag in tag_set:
    s = states_dict[tag]
    one_hyphen_tag_dict[tag] = 0
    for k, v in s.emit_dict.items():
        if any(x == "-" for x in k):
            one_hyphen_tag_dict[tag] += v
for tag in tag_set:
    one_hyphen_tag_dict[tag] = (float)(one_hyphen_tag_dict[tag]/sum(one_hyphen_tag_dict.values()))

#word is all upper case
all_capital_tag_dict = {}
for tag in tag_set:
    s = states_dict[tag]
    all_capital_tag_dict[tag] = 0
    for k, v in s.emit_dict.items():
        if all(x.isupper() for x in k):
            all_capital_tag_dict[tag] += v
for tag in tag_set:
    all_capital_tag_dict[tag] = (float)(all_capital_tag_dict[tag]/sum(all_capital_tag_dict.values()))

#word is upper case, and has a digit and a dash
word_hyphen_digit_tag_dict = {}
for tag in tag_set:
    s = states_dict[tag]
    word_hyphen_digit_tag_dict[tag] = 0
    for k, v in s.emit_dict.items():
        remove_digit = ''.join(i for i in k if not i.isdigit())
        remove_hypen = remove_digit.replace("_","")
        if all(x.isupper() for x in remove_hypen):
            word_hyphen_digit_tag_dict[tag] += v
for tag in tag_set:
    word_hyphen_digit_tag_dict[tag] = (float)(word_hyphen_digit_tag_dict[tag]/sum(word_hyphen_digit_tag_dict.values()))

print("Finish training for unknown word using morphology method")

# using operator to get the maximum value
import operator

# class to store pos tag with its probability
class predict_token:
    def __init__(self):
        self.states = ""
        self.prob = 0.0    

# predict unkown word return the emit probability dictionary
# morphology method
def predict_unkown_emit_prob(word):
    emit_dict = {}
    for tag in tag_set:
        emit_dict[tag] = 1
        
    flag_predicted = 0
    
    # word contains a particular suffix or prefix
    for fix in prefix_suffix_set:
        if fix[0] == "-":
            suffix = fix[1:]
            if word.endswith(suffix):
                for tag in tag_set:
                    flag_predicted = 1
                    emit_dict[tag] = emit_dict[tag]*prefix_suffix_tag_dict[fix][tag]
        if fix[-1:] == "-":
            prefix = fix[:-1]
            if word.startswith(prefix):
                for tag in tag_set:
                    flag_predicted = 1
                    emit_dict[tag] = emit_dict[tag]*prefix_suffix_tag_dict[fix][tag]
    
    # word contains a particular suffix from the set of all suffixes of length < 4
    #for suf in suffix_set:
    #    if word.endswith(suf):
    #        flag_predicted = 1
    #        for tag in tag_set:
    #            emit_dict[tag] = emit_dict[tag]*suffix_tag_dict[suf][tag]
                
    # word contains a particular prefix from the set of all prefixes of length < 4
    #for prefix in prefix_set:
    #    if word.startswith(prefix):
    #        for tag in tag_set:
    #            emit_dict[tag] = emit_dict[tag]*prefix_tag_dict[prefix][tag]
    
    # word contains a hyphen
    if any(x is "-" for x in word):
        flag_predicted = 1
        for tag in tag_set:
            emit_dict[tag] = one_hyphen_tag_dict[tag]
    
    # word is all upper case
    if all(x.isupper() for x in word):
        flag_predicted = 1
        for tag in tag_set:
            emit_dict[tag] = emit_dict[tag]*all_capital_tag_dict[tag]
    
    # word start with a upper letter
    if word[0].isupper():
        flag_predicted = 1
        for tag in tag_set:
            emit_dict[tag] = emit_dict[tag]*capital_tag_dict[tag]
            
    # word is upper case, and has a digit and a dash
    word_remove_digit = ''.join(i for i in word if not i.isdigit())
    word_remove_hypen = word_remove_digit.replace("_","")
    if all(x.isupper() for x in word_remove_hypen):
        flag_predicted = 1
        for tag in tag_set:
            emit_dict[tag] = emit_dict[tag]*word_hyphen_digit_tag_dict[tag]
            
    if is_number(word):
        flag_predicted = 1
        for tag in tag_set:
            emit_dict[tag] = 0
        emit_dict["CD"] = 1
    return emit_dict, flag_predicted

# function to check this word is unknown or not
def unknown_words(word):
    unknown_flag = 0
    if word == "__end__":
        unknown_flag = 1
    for tag in tag_set:
        s = states_dict[tag]
        if word in s.emit_dict:
            unknown_flag = 1
    
    return unknown_flag

## My Vertibi Decoder function
def predict_first_word_unkown_prob_dict(token):
    prob_dict = {} 
    emit_d, flag_predicted = predict_unkown_emit_prob(token)
    for tag in tag_set:
        t = predict_token()
        t.states = "start" +" " + tag
        if flag_predicted == 1:
            t.prob = states_dict["start"].transfer_dict[tag] * emit_d[tag]
        else:
            t.prob = states_dict["start"].transfer_dict[tag] * guess_unpredicatable_unknown_dict[tag]
        prob_dict[tag] = t
    return prob_dict
def get_prob_dict(token, prior_d):
    prob_dict = {}
    if unknown_words(token) == 0:
        emit_d, flag_predicted = predict_unkown_emit_prob(token)
        for tag in tag_set:
            s = states_dict[tag]
            if flag_predicted == 1:
                emit_prob = emit_d[tag]
            else:
                emit_prob = guess_unpredicatable_unknown_dict[tag]
            prob_dict_this_tag = {}
            for from_tag in tag_set:
                p_prob = prior_d[from_tag].prob
                t_prob = states_dict[from_tag].transfer_dict[tag]
                prob_dict_this_tag[prior_d[from_tag].states+" "+ tag] = p_prob * t_prob * emit_prob

            states = max(prob_dict_this_tag.items(), key=operator.itemgetter(1))[0]
            t = predict_token()
            t.states = states
            t.prob = prob_dict_this_tag[states]
            prob_dict[tag] = t
    else:
        if token != "__end__":
            for tag in tag_set:
                s = states_dict[tag]
                emit_prob = 0
                if token in s.emit_dict:
                    emit_prob = s.emit_dict[token]
                prob_dict_this_tag = {}
                for from_tag in tag_set:
                    p_prob = prior_d[from_tag].prob
                    t_prob = states_dict[from_tag].transfer_dict[tag]
                    prob_dict_this_tag[prior_d[from_tag].states+" "+ tag] = p_prob * t_prob * emit_prob

                states = max(prob_dict_this_tag.items(), key=operator.itemgetter(1))[0]
                t = predict_token()
                t.states = states
                t.prob = prob_dict_this_tag[states]
                prob_dict[tag] = t
        else:
            tag = "__end__"
            prob_dict_this_tag = {}
            for from_tag in tag_set:
                p_prob = prior_d[from_tag].prob
                t_prob = states_dict[from_tag].transfer_dict[tag]
                prob_dict_this_tag[prior_d[from_tag].states+" "+ tag] = p_prob * t_prob 

            states = max(prob_dict_this_tag.items(), key=operator.itemgetter(1))[0]
            t = predict_token()
            t.states = states
            t.prob = prob_dict_this_tag[states]
            prob_dict[tag] = t
    return prob_dict
        
def my_Vertibi_decoder(sent):
    origi_sent = sent.copy()
    sent.append("__end__")
    sent.reverse()
    
    prob_dict = {}
    prior_dict = {}
    
    token = sent.pop()
    if unknown_words(token) == 1:
        for tag in tag_set:
            s = states_dict[tag]
            emit_prob = 0
            if token in s.emit_dict:
                emit_prob = s.emit_dict[token]
            t = predict_token()
            t.states = "start" +" " + tag
            t.prob = emit_prob * states_dict["start"].transfer_dict[tag]
            prob_dict[tag] = t
    else:
        emit_d = predict_first_word_unkown_prob_dict(token)
        for tag in tag_set:
            s = states_dict[tag]
            emit_prob = 0
            emit_prob = emit_d[tag].prob
            t = predict_token()
            t.states = emit_d[tag].states
            t.prob = emit_prob * states_dict["start"].transfer_dict[tag]
            prob_dict[tag] = t
    
    prior_dict = prob_dict.copy()
    while len(sent)!=0:
        token = sent.pop()
        prob_dict = get_prob_dict(token, prior_dict)
        prior_dict= prob_dict.copy()
    
    max_prob = -1
    pos_tag_o = predict_token()
    for k, v in prior_dict.items():
        if v.prob > max_prob:
            pos_tag_o = v
    
    #if pos_tag_o.prob == 0:
    #    print(origi_sent, pos_tag_o.states)
    
    pos_tag_list = pos_tag_o.states.split(" ")
    pos_tag_list.pop()
    if pos_tag_list:
        pos_tag_list.pop(0)
    else:
        print(pos_tag_list)   
            
    return pos_tag_list

###sent = ["I", "earn", "1,239,418,982,349","."]

###pos_tag_list = my_Vertibi_decoder(sent)
###print(pos_tag_list)

print("start to test using "+test_file_name)
test_file = open(test_file_name)
test_sent = []
sent = []
for line in test_file:
    if(line is not "\n"):
        sent.append(line.rstrip("\n"))
    else:
        test_sent.append(sent)
        sent = []
test_file.close()



w_file = open(generate_pos_tag_file_name,"w")
for sent in test_sent:
    origi_sent = sent.copy()
    output_list = my_Vertibi_decoder(sent)
    for i in range (len(origi_sent)):
        w_file.write(origi_sent[i] + "\t"+output_list[i]+"\n")
    w_file.write("\n")


print("Finish testing, file write into "+ generate_pos_tag_file_name)
w_file.close()
