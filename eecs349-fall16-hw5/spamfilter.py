#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Prem Seetharaman (updated by Liangji Wang)

import sys
import numpy as np
import os
import shutil

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, dictionary_filename):
	#Making the dictionary. 
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
	words = {}

	#These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
	spamSet = set()
	hamSet = set()
	helpSet = set()
	spamCounter = 0
	hamCounter = 0
	for s in spam:
		spamCounter += 1
		for word in parse(open(spam_directory + s)):
			spamSet.add(word)
			helpSet.add(word)
			if word not in words:
				words[word] = {'spam': 0, 'ham': 0}
		for word in helpSet:
			temp = words[word]['spam']
			words[word]['spam'] = temp + 1
		helpSet.clear()
			

	for h in ham:
		hamCounter += 1
		for word in parse(open(ham_directory + h)):
			hamSet.add(word)
			helpSet.add(word)
			if word not in words:
				words[word] = {'spam': 0, 'ham': 0}
		for word in helpSet:
			temp = words[word]['ham']
			words[word]['ham'] = temp + 1
		helpSet.clear()

	spamCounter += 1
	hamCounter += 1
	
	for word in words:
		temp = words[word]['spam']
		if temp == 0:
			#print 'special case: word %s exist in ham but not spam, to avoid log(0), set its spam probability to 1/(spamCounter+1)' %(word)
			words[word]['spam'] = 1.0 / float(spamCounter)
		else:
			words[word]['spam'] = float(temp) / float(spamCounter)
		temp = words[word]['ham']
		if temp == 0:
			#print 'special case: word %s exist in spam but not ham, to avoid log(0), set it ham probability to 1/(hamCounter+1)' %(word)
			words[word]['ham'] = 1.0 / float(hamCounter)
		else:
			words[word]['ham'] = float(temp) / float(hamCounter)
	# check
	print 'spam file Counter: %s, ham file Counter: %s, dictLength: %s, spam word counter: %s, ham word counter: %s, spam_prior_probability: %s' %(spamCounter, hamCounter, len(words), len(spamSet), len(hamSet), spam_prior_probability)
	#Write it to a dictionary output file.
	writedictionary(words, dictionary_filename)
	
	return words, spam_prior_probability

def is_spam(content, dictionary, spam_prior_probability):
	#TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. You need to update it to make it use the dictionary and the content of the mail. Here is where your naive Bayes classifier goes.
	# Vnb = argMax(log(prior_probability) + sum(log(P(ai|vj))))
	# use the log to help replace multiple with addition
	spamV = []
	hamV = []
	spamV.append(spam_prior_probability)
	hamV.append(1-spam_prior_probability)
	#spamV = [spam_prior_probability]
	#hamV = [1-spam_prior_probability]
	for word in content:
		if word in dictionary:
			spamV.append(dictionary[word]['spam'])
			hamV.append(dictionary[word]['ham'])
	#print sum(np.log10(spamV)), sum(np.log10(hamV))
	if sum(np.log10(spamV)) > sum(np.log10(hamV)):
		return True
	return False

def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
	mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
	for m in mail:
		content = parse(open(mail_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		#print spam
		if spam:
			shutil.copy(mail_directory + m, spam_directory)
		else:
			shutil.copy(mail_directory + m, ham_directory)

if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
	training_spam_directory = sys.argv[1]
	training_ham_directory = sys.argv[2]
	
	test_mail_directory = sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'
	
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
	
	dictionary_filename = "dictionary.dict"
	
	#create the dictionary to be used
	dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	#sort the mail
	spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 
