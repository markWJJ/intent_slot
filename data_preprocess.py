import numpy as np
import pickle
import os
global PATH
PATH=os.path.split(os.path.realpath(__file__))[0]
import logging
import random
import jieba
import re
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("r_net_data")




class Intent_Slot_Data(object):
    '''
    intent_slot 数据处理模块
    '''
    def __init__(self, train_path, dev_path, test_path, batch_size, flag,max_length,use_auto_bucket):
        self.train_path = train_path  # 训练文件路径
        self.dev_path = dev_path  # 验证文件路径
        self.test_path = test_path  # 测试文件路径
        self.batch_size = batch_size  # batch大小
        self.max_length = max_length
        self.use_auto_bucket=use_auto_bucket
        self.index=0
        if flag == "train_new":
            self.vocab, self.slot_vocab, self.intent_vocab = self.get_vocab()
            pickle.dump(self.vocab, open(PATH + "/vocab.p", 'wb'))  # 词典
            pickle.dump(self.slot_vocab, open(PATH + "/slot_vocab.p", 'wb'))  # 词典
            pickle.dump(self.intent_vocab, open(PATH + "/intent_vocab.p", 'wb'))  # 词典
        elif flag == "test" or flag == "train":
            self.vocab = pickle.load(open(PATH + "/vocab.p", 'rb'))  # 词典
            self.slot_vocab = pickle.load(open(PATH + "/slot_vocab.p", 'rb'))  # 词典
            self.intent_vocab = pickle.load(open(PATH + "/intent_vocab.p", 'rb'))  # 词典
        else:
            pass
        self.vocab_num=len(self.vocab)
        self.slot_num=len(self.slot_vocab)
        self.intent_num=len(self.intent_vocab)
        batch_list, self.num_batch = self.data_deal_train()
        self.batch_list = self.shuffle(batch_list)

    def shuffle(self, data_list):
        '''

        :param data_list:
        :return:
        '''
        index = [i for i in range(len(data_list))]
        random.shuffle(index)
        new_data_list = [data_list[e] for e in index]
        return new_data_list

    def get_vocab(self):
        '''
        构造字典 dict{NONE:0,word1:1,word2:2...wordn:n} NONE为未登录词
        :return:
        '''
        train_file = open(self.train_path, 'r')
        test_file = open(self.dev_path, 'r')
        dev_file = open(self.test_path, 'r')
        vocab = {"NONE": 0,'BOS':1,'EOS':2}
        intent_vocab = {'NONE':0}
        slot_vocab={}
        self.index = len(vocab)
        self.intent_label_index = len(intent_vocab)
        self.slot_label_index=len(slot_vocab)

        def _vocab(file):
            for ele in file:
                ele = ele.replace("\n", "")
                eles=ele.split('\t')
                words=eles[0].split(' ')
                slots=eles[1].split(' ')[:-1]
                slots=[e for e in slots if e]
                intent=eles[1].split(' ')[-1]

                for slots_label in slots:
                    slots_label=str(slots_label.lower().replace('[','')).replace("'",'').replace("'",'').replace(',','')
                    if slots_label not in slot_vocab :
                        slot_vocab[slots_label] = self.slot_label_index
                        self.slot_label_index += 1
                for word in words:
                    word=word.lower()
                    if word not in vocab:
                        vocab[word]=self.index
                        self.index+=1
                if intent not in intent_vocab:
                    intent=intent.lower()
                    intent_vocab[intent]=self.intent_label_index
                    self.intent_label_index+=1
        _vocab(train_file)
        _vocab(dev_file)
        _vocab(test_file)

        return vocab,slot_vocab,intent_vocab

    def seg_feature(self, seg_list):
        '''
        构建分词特征
        :param seg_list:
        :return:
        '''
        seg_fea = []
        for e in seg_list:
            if len(e) == 1:
                seg_fea.append(0)
            else:
                ss = [2] * len(e)
                ss[0] = 1
                ss[-1] = 3
                seg_fea.extend(ss)
        return seg_fea

    def _convert_sent(self, sent):
        '''
        将sent中的数字分开
        :param sent:
        :return:
        '''
        sents = str(sent).replace("\n", "")
        new_sent = [e for e in sents]
        return ' '.join(new_sent)

    def shuffle_sent(self, data_list):
        '''

        :param data_list:
        :return:
        '''
        index_list = [i for i in range(len(data_list))]
        random.shuffle(index_list)
        new_data_list = [data_list[i] for i in index_list]
        return new_data_list

    def padd_sentences(self, sent_list):
        '''
        find the max length from sent_list , and standardation
        :param sent_list:
        :return:
        '''
        words = [str(sent).replace('\n', '').split('\t')[0] for sent in sent_list]
        # slot_labels = [' '.join(str(sent).replace('\n', '').split('\t')[1].split(' ')[:-1]) for sent in sent_list]
        # intent_labels=[str(sent).replace('\n', '').split('\t')[1].split(' ')[-1] for sent in sent_list]
        slot_labels = [' '.join([sent.replace('\n','').split('\t')[1].split(' ')[:-1]])   for sent in sent_list]
        intent_labels = [' '.join([sent.replace('\n','').split('\t')[1].split(' ')[-1]]) for sent in sent_list]
        max_len = max([len(ele.split(' ')) for ele in words])
        word_arr = []
        slot_arr = []
        intent_arr = []
        real_len_arr = []
        for sent, slot,intent in zip(words, slot_labels,intent_labels):

            sent_list = []
            real_len = len(sent.split(' '))
            for word in sent.split(' '):
                word = word.lower()
                if word in self.vocab:
                    sent_list.append(self.vocab[word])
                else:
                    sent_list.append(0)

            slot_list = []
            slots = slot.split(' ')
            for ll in slots:
                ll=str(ll.lower().replace('[','')).replace("'",'').replace("'",'').replace(',','')
                if ll in self.slot_vocab:
                    slot_list.append(self.slot_vocab[ll])
                else:
                    slot_list.append(self.slot_vocab['O'.lower()])

            intent_list=[]
            intents=intent.split(' ')
            for ll in intents:
                ll=ll.lower()
                if ll in self.intent_vocab:
                    intent_list.append(self.intent_vocab[ll])
                else:
                    intent_list.append(0)


            if len(sent_list) >= max_len:
                new_sent_list = sent_list[0:max_len]
            else:
                new_sent_list = sent_list
                ss = [0] * (max_len - len(sent_list))
                new_sent_list.extend(ss)

            if len(slot_list) >= max_len:
                new_slot_list = slot_list[0:max_len]
            else:
                new_slot_list = slot_list
                ss_l = [0] * (max_len - len(slot_list))
                new_slot_list.extend(ss_l)

            if real_len >= max_len:
                real_len = max_len

            real_len_arr.append(real_len)
            word_arr.append(new_sent_list)
            slot_arr.append(new_slot_list)
            intent_arr.append(intent_list)

        real_len_arr = np.array(real_len_arr)
        word_arr = np.array(word_arr)
        slot_arr=np.array(slot_arr)
        intent_arr = np.array(intent_arr)
        intent_arr=np.reshape(intent_arr,(intent_arr.shape[0]))

        return word_arr, slot_arr, intent_arr, real_len_arr

    def padd_sentences_no_buckets(self, sent_list):
        '''
        find the max length from sent_list , and standardation
        :param sent_list:
        :return:
        '''
        # slot_labels = [' '.join(str(sent).replace('\n', '').split('\t')[1].split(' ')[:-1]) for sent in sent_list]
        # intent_labels=[str(sent).replace('\n', '').split('\t')[1].split(' ')[-1] for sent in sent_list]
        words,slot_labels,intent_labels=[],[],[]
        for sent in sent_list:
            sent=sent.replace('\n','')
            sents=sent.split('\t')
            words.append(sents[0].split(' '))
            slot_intent=sents[1].split(' ')
            slot_labels.append(slot_intent[:-1])
            intent_labels.append(slot_intent[-1])
        # slot_labels = [' '.join([sent.replace('\n', '').split('\t')[1].split(' ')[:-1]]) for sent in sent_list]
        # intent_labels = [' '.join([sent.replace('\n', '').split('\t')[1].split(' ')[-1]]) for sent in sent_list]
        # print(slot_labels)
        max_len=self.max_length
        word_arr = []
        slot_arr = []
        intent_arr = []
        real_len_arr = []
        for sent, slot,intent in zip(words, slot_labels,intent_labels):
            sent_list = []
            real_len = len(sent)
            for word in sent:
                word = word.lower()

                if word in self.vocab:
                    sent_list.append(self.vocab[word])
                else:
                    sent_list.append(0)

            slot_list = []
            slots = slot
            for ll in slots:
                ll=str(ll.lower().replace('[','')).replace("'",'').replace("'",'').replace(',','')
                if ll in self.slot_vocab:
                    slot_list.append(self.slot_vocab[ll])
                else:
                    slot_list.append(self.slot_vocab['O'.lower()])

            intent_list=[]
            intents=intent.split(' ')
            for ll in intents:
                ll=ll.lower()
                if ll in self.intent_vocab:
                    intent_list.append(self.intent_vocab[ll])
                else:
                    intent_list.append(0)


            if len(sent_list) >= max_len:
                new_sent_list = sent_list[0:max_len]
            else:
                new_sent_list = sent_list
                ss = [0] * (max_len - len(sent_list))
                new_sent_list.extend(ss)

            if len(slot_list) >= max_len:
                new_slot_list = slot_list[0:max_len]
            else:
                new_slot_list = slot_list
                ss_l = [0] * (max_len - len(slot_list))
                new_slot_list.extend(ss_l)

            if real_len >= max_len:
                real_len = max_len

            real_len_arr.append(real_len)
            word_arr.append(new_sent_list)
            slot_arr.append(new_slot_list)
            intent_arr.append(np.array(intent_list))

        real_len_arr = np.array(real_len_arr)
        word_arr = np.array(word_arr)
        slot_arr=np.array(slot_arr)
        intent_arr = np.array(intent_arr)
        intent_arr=np.reshape(intent_arr,(intent_arr.shape[0]))

        return word_arr, slot_arr, intent_arr, real_len_arr

    def data_deal_train(self):
        '''
        对训练样本按照长度进行排序 分箱
        :return:
        '''
        train_flie = open(self.train_path, 'r')
        data_list = [line for line in train_flie.readlines()]

        data_list.sort(key=lambda x: len(x))  # sort not shuffle

        num_batch = int(len(data_list) / int(self.batch_size))

        batch_list = []
        for i in range(num_batch):
            ele = data_list[i * self.batch_size:(i + 1) * self.batch_size]
            if self.use_auto_bucket:
                word_arr, slot_arr, intent_arr, real_len_arr = self.padd_sentences(ele)
            else:
                word_arr, slot_arr, intent_arr, real_len_arr = self.padd_sentences_no_buckets(ele)

            _logger.info('word:%s slot_shape:%s intent_shape:%s '%(word_arr.shape,slot_arr.shape,intent_arr.shape))
            batch_list.append((word_arr, slot_arr,intent_arr, real_len_arr))
        return batch_list, num_batch

    def next_batch(self):
        '''

        :return:
        '''
        num_iter = self.num_batch
        if self.index < num_iter:
            return_sent = self.batch_list[self.index][0]
            return_slot = self.batch_list[self.index][1]
            return_intent = self.batch_list[self.index][2]
            return_real_len=self.batch_list[self.index][3]
            current_length = self.batch_list[self.index][0].shape[1]
            current_length = np.array((current_length,), dtype=np.int32)
            self.index += 1
        else:
            self.index = 0
            return_sent = self.batch_list[self.index][0]
            return_slot = self.batch_list[self.index][1]
            return_intent = self.batch_list[self.index][2]
            return_real_len = self.batch_list[self.index][3]
            current_length = np.array((len(self.batch_list[self.index][0]),), dtype=np.int32)

        return return_sent, return_slot, return_intent,return_real_len,current_length

    def get_dev(self):
        train_flie = open(self.dev_path, 'r')
        data_list = [line for line in train_flie.readlines()]

        data_list.sort(key=lambda x: len(x))  # sort not shuffle

        num_batch = int(len(data_list) / int(self.batch_size))

        batch_list = []
        ele = data_list[:]
        if self.use_auto_bucket:
            word_arr, slot_arr, intent_arr, real_len_arr = self.padd_sentences(ele)
        else:
            word_arr, slot_arr, intent_arr, real_len_arr = self.padd_sentences_no_buckets(ele)

        _logger.info('word:%s slot_shape:%s intent_shape:%s ' % (word_arr.shape, slot_arr.shape, intent_arr.shape))
        return  word_arr, slot_arr, intent_arr, real_len_arr



if __name__ == '__main__':

    dd = Intent_Slot_Data(train_path="./dataset/atis-2.train.w-intent_1.iob", test_path="./dataset/atis.test.w-intent.iob",
                            dev_path="./dataset/atis-2.dev.w-intent.iob", batch_size=20 ,max_length=30, flag="train_new",use_auto_bucket=False)


    sent,slot,intent,real_len,cur_len=dd.next_batch()
    # print(sent.shape,slot.shape,intent.shape,real_len.shape,cur_len.shape)
    # print(dd.slot_vocab)
    print(cur_len)
