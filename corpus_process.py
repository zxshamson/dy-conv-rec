# encoding=utf-8
import sys
import collections
import random
import numpy as np


class Corpus:

    def __init__(self, train_files, test_file, valid_file, using_voting=False):

        self.convNum = 0            # Number of conversations
        self.convIDs = {}           # Dictionary that maps conversations to integer IDs
        self.r_convIDs = {}         # Inverse of last dictionary
        self.userNum = 0            # Number of users
        self.userIDs = {}           # Dictionary that maps users to integer IDs
        self.r_userIDs = {}         # Inverse of last dictionary
        self.wordNum = 1            # Number of words, 0 padding not stores words
        self.wordIDs = {}           # Dictionary that maps words to integers
        self.r_wordIDs = {}         # Inverse of last dictionary

        # Each conv is a list of turns, each turn is [userID, time_interval, turnID, p_turnID, words]
        self.convs = collections.defaultdict(list)
        self.convs_in_train = set()
        self.convs_in_test = set()
        self.convs_in_valid = set()
        # Stores each user's replying history
        self.user_history = collections.defaultdict(list)
        # Stores the convs that the user replying in predict part
        self.test_pred = collections.defaultdict(set)
        self.valid_pred = collections.defaultdict(set)

        self.posttime = {}          # Record each turn's arriving time
        if using_voting:
            self.voting = {}        # Record voting for each turn, format: (up_votes, down_votes)
        wordCount = collections.Counter()  # The count every word appears
        turnNum = collections.Counter()    # The turn num of each conv
        pos_in_conv = {'null': -1}  # The position of each turn in that conv

        # Reading train files
        readNum = 0
        for tfile in train_files:
            with open(tfile, 'r') as f:
                for line in f:
                    readNum += 1
                    if readNum % 1000000 == 0:
                        print "Data reading.... Line ", readNum
                    # Split each message to several parts
                    msgs = line.strip().split('\t')
                    # Update and record conv and user ID
                    if msgs[0] not in self.convIDs:
                        self.convs_in_train.add(self.convNum)
                        self.convIDs[msgs[0]] = self.convNum
                        self.r_convIDs[self.convNum] = msgs[0]
                        self.convNum += 1
                    if msgs[5] not in self.userIDs:
                        self.userIDs[msgs[5]] = self.userNum
                        self.r_userIDs[self.userNum] = msgs[5]
                        self.userNum += 1
                    u = self.userIDs[msgs[5]]
                    c = self.convIDs[msgs[0]]
                    # Record side information
                    self.posttime[msgs[1]] = int(msgs[6])
                    if using_voting:
                        self.voting[msgs[1]] = (int(msgs[7]), int(msgs[8]))
                    # Record turn information
                    pos_in_conv[msgs[1]] = turnNum[c]
                    turnNum[c] += 1
                    # Update and record word information
                    current_words = []
                    for word in msgs[4].split(' '):
                        if wordCount[word] == 0:
                            self.wordIDs[word] = self.wordNum
                            self.r_wordIDs[self.wordNum] = word
                            self.wordNum += 1
                        current_words.append(self.wordIDs[word])
                        wordCount[word] += 1
                    # Record each turn's messages in conv
                    current_turn = [u, self.posttime[msgs[1]], pos_in_conv[msgs[1]], pos_in_conv[msgs[2]]]
                    current_turn.extend(current_words)
                    self.convs[c].append(current_turn)
                    # Record each message's information in user history
                    current_msg = [c, self.posttime[msgs[1]], pos_in_conv[msgs[1]]]
                    self.user_history[u].append(current_msg)
        train_userNum, train_convNum, train_msgNum = self.userNum, self.convNum, readNum

        # Reading test file
        with open(test_file, 'r') as f:
            for line in f:
                readNum += 1
                if readNum % 1000000 == 0:
                    print "Data reading.... Line ", readNum
                # Split each message to several parts
                msgs = line.strip().split('\t')
                # find out whether it is predicted part
                if len(msgs) == 2:
                    pred_users = msgs[1].split(' ')
                    for user in pred_users:
                        if user in self.userIDs:
                            self.test_pred[self.userIDs[user]].add(self.convIDs[msgs[0]])
                    continue
                # Update and record conv and user ID
                if msgs[0] not in self.convIDs:
                    self.convIDs[msgs[0]] = self.convNum
                    self.r_convIDs[self.convNum] = msgs[0]
                    self.convNum += 1
                if msgs[5] not in self.userIDs:
                    self.userIDs[msgs[5]] = self.userNum
                    self.r_userIDs[self.userNum] = msgs[5]
                    self.userNum += 1
                u = self.userIDs[msgs[5]]
                c = self.convIDs[msgs[0]]
                self.convs_in_test.add(c)
                # check whether current conv is in train convs
                if c in self.convs_in_train:
                    continue
                # Record side information
                self.posttime[msgs[1]] = int(msgs[6])
                if using_voting:
                    self.voting[msgs[1]] = (int(msgs[7]), int(msgs[8]))
                # Record turn information
                pos_in_conv[msgs[1]] = turnNum[c]
                turnNum[c] += 1
                # Update and record word information
                current_words = []
                for word in msgs[4].split(' '):
                    if wordCount[word] == 0:
                        self.wordIDs[word] = self.wordNum
                        self.r_wordIDs[self.wordNum] = word
                        self.wordNum += 1
                    current_words.append(self.wordIDs[word])
                    wordCount[word] += 1
                # Record each turn's messages in conv
                current_turn = [u, self.posttime[msgs[1]], pos_in_conv[msgs[1]], pos_in_conv[msgs[2]]]
                current_turn.extend(current_words)
                self.convs[c].append(current_turn)

        # Reading valid file
        with open(valid_file, 'r') as f:
            for line in f:
                readNum += 1
                if readNum % 1000000 == 0:
                    print "Data reading.... Line ", readNum
                # Split each message to several parts
                msgs = line.strip().split('\t')
                # find out whether it is predicted part
                if len(msgs) == 2:
                    pred_users = msgs[1].split(' ')
                    for user in pred_users:
                        if user in self.userIDs:
                            self.valid_pred[self.userIDs[user]].add(self.convIDs[msgs[0]])
                    continue
                # Update and record conv and user ID
                if msgs[0] not in self.convIDs:
                    self.convIDs[msgs[0]] = self.convNum
                    self.r_convIDs[self.convNum] = msgs[0]
                    self.convNum += 1
                if msgs[5] not in self.userIDs:
                    self.userIDs[msgs[5]] = self.userNum
                    self.r_userIDs[self.userNum] = msgs[5]
                    self.userNum += 1
                u = self.userIDs[msgs[5]]
                c = self.convIDs[msgs[0]]
                self.convs_in_valid.add(c)
                # check whether current conv is in train convs
                if c in self.convs_in_train:
                    continue
                # Record side information
                self.posttime[msgs[1]] = int(msgs[6])
                if using_voting:
                    self.voting[msgs[1]] = (int(msgs[7]), int(msgs[8]))
                # Record turn information
                pos_in_conv[msgs[1]] = turnNum[c]
                turnNum[c] += 1
                # Update and record word information
                current_words = []
                for word in msgs[4].split(' '):
                    if wordCount[word] == 0:
                        self.wordIDs[word] = self.wordNum
                        self.r_wordIDs[self.wordNum] = word
                        self.wordNum += 1
                    current_words.append(self.wordIDs[word])
                    wordCount[word] += 1
                # Record each turn's messages in conv
                current_turn = [u, self.posttime[msgs[1]], pos_in_conv[msgs[1]], pos_in_conv[msgs[2]]]
                current_turn.extend(current_words)
                self.convs[c].append(current_turn)

        print "Corpus process over!"
        print "In total -- UserNum: ", self.userNum, "ConvNum: ", self.convNum, "MsgNum: ", readNum
        print "For training -- UserNum: ", train_userNum, "ConvNum: ", train_convNum, "MsgNum: ", train_msgNum


if __name__ == '__main__':
    file_prefix = 'Datafiles/technology_20150'
    train_files = [file_prefix+'1.data', file_prefix+'2.data', file_prefix+'3.data', file_prefix+'4.data']
    test_file = file_prefix+'5_test.data'
    valid_file = file_prefix+'5_valid.data'
    corp = Corpus(train_files, test_file, valid_file)
