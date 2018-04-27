import numpy as np
import sys
import os
import random
import tensorflow as tf
import logging
from logging.config import fileConfig
import time
from data_preprocess import Intent_Slot_Data
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append("./")
sys.path.append("./data/")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='modellog.log',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")


class Config(object):
    '''
    默认配置
    '''
    learning_rate = 0.001
    batch_size = 16
    sent_len = 50  # 句子长度
    embedding_dim = 100  # 词向量维度
    hidden_dim = 100
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = './save_model_1/model_%s/r_net_model_%s.ckpt'
    if not os.path.exists('./save_model/model_%s_'):
        os.makedirs('./save_model/model_%s_')
    use_cpu_num = 16
    keep_dropout = 0.7
    summary_write_dir = "./tmp/r_net.log"
    epoch = 100
    use_auto_buckets=False
    lambda1 = 0.01
    model_mode = 'bilstm_attention_crf'  # 模型选择：bilstm bilstm_crf bilstm_attention bilstm_attention_crf,cnn_crf


config = Config()
tf.app.flags.DEFINE_float("lambda1", config.lambda1, "l2学习率")
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "学习率")
tf.app.flags.DEFINE_float("keep_dropout", config.keep_dropout, "dropout")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "批处理的样本数量")
tf.app.flags.DEFINE_integer("max_len", config.sent_len, "句子长度")
tf.app.flags.DEFINE_integer("embedding_dim", config.embedding_dim, "词嵌入维度.")
tf.app.flags.DEFINE_integer("hidden_dim", config.hidden_dim, "中间节点维度.")
tf.app.flags.DEFINE_integer("use_cpu_num", config.use_cpu_num, "限定使用cpu的个数")
tf.app.flags.DEFINE_integer("epoch", config.epoch, "epoch次数")
tf.app.flags.DEFINE_string("summary_write_dir", config.summary_write_dir, "训练数据过程可视化文件保存地址")
tf.app.flags.DEFINE_string("train_dir", config.train_dir, "训练数据的路径")
tf.app.flags.DEFINE_string("dev_dir", config.dev_dir, "验证数据文件路径")
tf.app.flags.DEFINE_string("test_dir", config.test_dir, "测试数据文件路径")
tf.app.flags.DEFINE_string("model_dir", config.model_dir, "模型保存路径")
tf.app.flags.DEFINE_boolean('use Encoder2Decoder',False,'')
tf.app.flags.DEFINE_string("mod", "train", "默认为训练")  # true for prediction
tf.app.flags.DEFINE_string('model_mode', config.model_mode, '模型类型')
tf.app.flags.DEFINE_boolean('use_auto_buckets',config.use_auto_buckets,'是否使用自动桶')
tf.app.flags.DEFINE_string('only_mode','intent_slot','执行哪种单一任务')
FLAGS = tf.app.flags.FLAGS


class Model(object):

    def __init__(self, slot_num_class,intent_num_class,vocab_num):


        self.hidden_dim = FLAGS.hidden_dim
        self.use_buckets=FLAGS.use_auto_buckets
        self.model_mode = FLAGS.model_mode
        self.batch_size = FLAGS.batch_size
        self.max_len=FLAGS.max_len
        self.embedding_dim = FLAGS.embedding_dim
        self.slot_num_class=slot_num_class
        self.intent_num_class=intent_num_class
        self.vocab_num=vocab_num
        self.init_graph()

        with tf.device('/gpu:2'):

            self.encoder_outs,self.encoder_final_states=self.encoder()

            if FLAGS.only_mode=='intent':

                self.intent_losses=self.intent_loss()
                self.loss_op=self.intent_losses
            elif FLAGS.only_mode=='slot':
                # self.slot_loss=self.Encoder_Decoder()
                self.slot_loss=self.decoder()
                self.loss_op=self.slot_loss
            elif FLAGS.only_mode=='intent_slot' or FLAGS.only_mode=='intent_slot_crf':
                self.intent_losses=self.intent_loss()
                self.slot_loss=self.decoder()
                self.loss_op=self.intent_losses+self.slot_loss

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss_op)

            # self.opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            # grads_vars = self.opt.compute_gradients(self.loss_op)
            #
            # capped_grads_vars = [[tf.clip_by_value(g, -5.0, 5.0), v] for g, v in grads_vars]
            #
            # self.optimizer = self.opt.apply_gradients(capped_grads_vars)

    def init_graph(self):
        '''

        :return:
        '''
        if self.use_buckets:
            self.sent=tf.placeholder(shape=(None,None),dtype=tf.int32)
            self.slot=tf.placeholder(shape=(None,None),dtype=tf.int32)
            self.intent=tf.placeholder(shape=(None,),dtype=tf.int32)
            self.seq_vec=tf.placeholder(shape=(None,),dtype=tf.int32)
            self.rel_num=tf.placeholder(shape=(1,),dtype=tf.int32)
        else:
            self.sent = tf.placeholder(shape=(None, self.max_len), dtype=tf.int32)
            self.slot = tf.placeholder(shape=(None, self.max_len), dtype=tf.int32)
            self.intent = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.seq_vec = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.rel_num = tf.placeholder(shape=(1,), dtype=tf.int32)

        # self.global_step = tf.Variable(0, trainable=True)

        self.sent_embedding=tf.Variable(tf.random_normal(shape=(self.vocab_num,self.embedding_dim),
                                                         dtype=tf.float32),trainable=False)
        self.slot_embedding=tf.Variable(tf.random_normal(shape=(self.slot_num_class,self.embedding_dim),
                                                         dtype=tf.float32),trainable=False)

        self.sent_emb=tf.nn.embedding_lookup(self.sent_embedding,self.sent)
        self.slot_emb=tf.nn.embedding_lookup(self.slot_embedding,self.slot)

        self.lstm_fw=tf.contrib.rnn.LSTMCell(self.hidden_dim)
        self.lstm_bw=tf.contrib.rnn.LSTMCell(self.hidden_dim)

    def encoder(self):
        '''
        编码层
        :return:
        '''
        #final_states=((fw_c_last,fw_h_last),(bw_c_last,bw_h_last))
        lstm_out, final_states = tf.nn.bidirectional_dynamic_rnn(
            self.lstm_fw,
            self.lstm_bw,
            self.sent_emb,
            dtype=tf.float32,
            sequence_length=self.seq_vec,)

        lstm_out=tf.concat(lstm_out,2)
        lstm_outs=tf.stack(lstm_out) # [batch_size,seq_len,dim] 作为attention的注意力矩阵

        state_c=tf.concat((final_states[0][0],final_states[1][0]),1) #作为decoder的inital states中state_c
        state_h=tf.concat((final_states[0][1],final_states[1][1]),1) #作为decoder的inital states中state_h

        encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
            c=state_c,
            h=state_h
        )
        return lstm_outs,encoder_final_state

    def intent_attention(self, lstm_outs):
        '''
        输入lstm的输出组，进行attention处理
        :param lstm_outs:
        :return:
        '''

        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.seq_len)))
        b_h=tf.Variable(tf.random_normal(shape=(self.seq_len,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        G=tf.nn.softmax(tf.nn.tanh(tf.add(logit,b_h)))#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)
        '''
        w_h = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, 2 * self.hidden_dim)))
        b_h = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim,)))
        logit = tf.einsum("ijk,kl->ijl", lstm_outs, w_h)
        logit = tf.nn.tanh(tf.add(logit, b_h))
        logit = tf.tanh(tf.einsum("ijk,ilk->ijl", logit, lstm_outs))
        G = tf.nn.softmax(logit)  # G.shape=[self.seq_len,self.seq_len]
        logit_ = tf.einsum("ijk,ikl->ijl", G, lstm_outs)

        # 注意力得到的logit与lstm_outs进行链接

        outs = tf.concat((logit_, lstm_outs), 2)  # outs.shape=[None,seq_len,4*hidden_dim]
        return outs

    def self_lstm_attention_ops(self,lstm_out_t,lstm_outs):
        '''

        :return:
        '''
        w=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_out_t 参数
        g=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_outs 参数
        lstm_out_t=tf.reshape(lstm_out_t,[-1,1,2*self.hidden_dim])

        v=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,1)))
        with tf.variable_scope('self_attention',reuse=True):
            lstm_out_t_=tf.einsum('ijk,kl->ijl',lstm_out_t,w)
            lstm_outs_=tf.einsum('ijk,kl->ijl',lstm_outs,g)
            gg=tf.tanh(lstm_out_t_+lstm_outs_)
            gg_=tf.einsum('ijk,kl->ijl',gg,v)
            gg_soft=tf.nn.softmax(gg_,1)
            a=tf.einsum('ijk,ijl->ikl',lstm_outs,gg_soft)
            a=tf.reshape(a,[-1,2*self.hidden_dim])
            return a

    def attention(self,lstm_outs):
        '''
        输入lstm的输出组，进行attention处理
        :param lstm_outs:
        :return:
        '''

        '''
                w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.seq_len)))
        b_h=tf.Variable(tf.random_normal(shape=(self.seq_len,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        G=tf.nn.softmax(tf.nn.tanh(tf.add(logit,b_h)))#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)
        '''
        w_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,2*self.hidden_dim)))
        b_h=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,)))
        logit=tf.einsum("ijk,kl->ijl",lstm_outs,w_h)
        logit=tf.nn.tanh(tf.add(logit,b_h))
        logit=tf.tanh(tf.einsum("ijk,ilk->ijl",logit,lstm_outs))
        G=tf.nn.softmax(logit)#G.shape=[self.seq_len,self.seq_len]
        logit_=tf.einsum("ijk,ikl->ijl",G,lstm_outs)


        # 注意力得到的logit与lstm_outs进行链接

        outs=tf.concat((logit_,lstm_outs),2)#outs.shape=[None,seq_len,4*hidden_dim]
        return outs


    def self_lstm_attention_ops_decoder(self,lstm_out_t,lstm_outs):
        '''

        :return:
        '''
        w=tf.Variable(tf.random_uniform(shape=(self.hidden_dim,2*self.hidden_dim))) #lstm_out_t 参数
        g=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,2*self.hidden_dim))) #lstm_outs 参数
        lstm_out_t=tf.reshape(lstm_out_t,[-1,1,self.hidden_dim])

        out_w_=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,self.hidden_dim)))
        out_b_=tf.Variable(tf.random_uniform(shape=(self.hidden_dim,)))

        v=tf.Variable(tf.random_uniform(shape=(2*self.hidden_dim,1)))
        with tf.variable_scope('self_attention',reuse=True):
            lstm_out_t_=tf.einsum('ijk,kl->ijl',lstm_out_t,w)
            lstm_outs_=tf.einsum('ijk,kl->ijl',lstm_outs,g)
            gg=tf.tanh(lstm_out_t_+lstm_outs_)
            gg_=tf.einsum('ijk,kl->ijl',gg,v)
            gg_soft=tf.nn.softmax(gg_,1)
            a=tf.einsum('ijk,ijl->ikl',lstm_outs,gg_soft)
            a=tf.reshape(a,[-1,2*self.hidden_dim])
            new_a=tf.add(tf.matmul(a,out_w_),out_b_)
            return new_a

    def self_lstm_attention(self,lstm_outs):
        '''
        对lstm输出再做一层 attention_lstm
        :param lstm_outs:
        :return:
        '''

        lstm_cell=tf.contrib.rnn.LSTMCell(2*self.hidden_dim,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                state_is_tuple=True)
        lstm_outs_list=tf.unstack(lstm_outs,self.max_len,1)
        init_state=tf.zeros_like(lstm_outs_list[0])
        states=[(init_state,init_state)]
        H=[]
        w=tf.Variable(tf.random_uniform(shape=(4*self.hidden_dim,4*self.hidden_dim)))
        with tf.variable_scope('lstm_attention'):
            for i in range(self.max_len):
                if i>0:
                    tf.get_variable_scope().reuse_variables()
                lstm_outs_t=lstm_outs_list[i]
                a=self.self_lstm_attention_ops(lstm_outs_t,lstm_outs) #attention的值

                new_input=tf.concat((lstm_outs_t,a),1)

                new_input_=tf.sigmoid(tf.matmul(new_input,w))*new_input

                h,state=lstm_cell(new_input_,states[-1])
                H.append(h)
                states.append(state)
        H=tf.stack(H)
        H=tf.transpose(H,[1,0,2])
        return H

    def intent_loss(self):
        '''

        :return:
        '''

        intent_mod='origin_attention'

        if intent_mod=='max_pool':
            lstm_w = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.intent_num_class), dtype=tf.float32))
            lstm_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,), dtype=tf.float32))

            encoder_out=tf.expand_dims(self.encoder_outs,3)
            lstm_out=tf.nn.max_pool(encoder_out, ksize = [1,self.rel_num, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'maxpool1')
            lstm_out=tf.reshape(lstm_out,[-1,2*self.hidden_dim])
            logit=tf.add(tf.matmul(lstm_out,lstm_w),lstm_b)
            intent_one_hot=tf.one_hot(self.intent,self.intent_num_class,1,0)
            intent_loss=tf.losses.softmax_cross_entropy(intent_one_hot,logit)
            return intent_loss

        elif intent_mod=='origin_attention':
            lstm_w = tf.Variable(tf.random_normal(shape=(4 * self.hidden_dim, self.intent_num_class), dtype=tf.float32))
            lstm_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,), dtype=tf.float32))
            lstm_out=self.intent_attention(self.encoder_outs)
            lstm_out=tf.transpose(lstm_out,[1,0,2])[-1]
            logit = tf.add(tf.matmul(lstm_out, lstm_w), lstm_b)
            self.intent_soft_logit=tf.nn.softmax(logit,1)
            # intent_one_hot = tf.one_hot(self.intent, self.intent_num_class, 1, 0)
            # intent_loss = tf.losses.softmax_cross_entropy(intent_one_hot, logit)
            intent_loss=tf.losses.sparse_softmax_cross_entropy(self.intent,logit,reduction=tf.losses.Reduction.NONE)
            # intent_loss=tf.reduce_mean(intent_loss)
            # mask=tf.sequence_mask(self.seq_vec,self.intent_num_class)
            # intent_loss=tf.boolean_mask(loss,mask)
            intent_loss=tf.reduce_mean(intent_loss)
            return intent_loss

        elif intent_mod=='origin_self_attenion':
            lstm_w = tf.Variable(tf.random_normal(shape=(2 * self.hidden_dim, self.intent_num_class), dtype=tf.float32))
            lstm_b = tf.Variable(tf.random_normal(shape=(self.intent_num_class,), dtype=tf.float32))
            lstm_out=self.self_lstm_attention(self.encoder_outs)
            lstm_out=tf.transpose(lstm_out,[1,0,2])[-1]
            logit = tf.add(tf.matmul(lstm_out, lstm_w), lstm_b)
            self.soft_logit=tf.nn.softmax(logit,1)
            intent_one_hot = tf.one_hot(self.intent, self.intent_num_class, 1, 0)
            intent_loss=tf.losses.sparse_softmax_cross_entropy(self.intent,logit,reduction=tf.losses.Reduction.NONE)
            # intent_loss=tf.reduce_mean(intent_loss)
            # mask=tf.sequence_mask(self.seq_vec,self.intent_num_class)
            # intent_loss=tf.boolean_mask(loss,mask)
            intent_loss=tf.reduce_mean(intent_loss)
            return intent_loss

    def decoder(self):
        '''
        slot decoder layer
        :return:
        '''
        decoder_mod='self_decoder'
        if decoder_mod=='decoder':
            lstm_cell=tf.contrib.rnn.LSTMCell(2*self.hidden_dim,state_is_tuple=True)
            decoder_list=tf.unstack(self.slot_emb,self.max_len,1)
            # decoder_list=tf.unstack(self.encoder_outs,self.max_len,1)
            encoder_state = self.encoder_final_states
            decoder_out, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs=decoder_list,
                initial_state=encoder_state,
                attention_states=self.encoder_outs,
                cell=lstm_cell,
                output_size=None,
            )
            decoder_out=tf.stack(decoder_out,1)
            softmax_w=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.slot_num_class),dtype=tf.float32))
            softmax_b=tf.Variable(tf.random_normal(shape=(self.slot_num_class,),dtype=tf.float32))

            soft_logit=tf.add(tf.einsum('ijk,kl->ijl',decoder_out,softmax_w),softmax_b)
            self.soft_logit=tf.nn.softmax(soft_logit,1)

            slot_one_hot=tf.one_hot(self.slot,self.slot_num_class,1,0,axis=2)
            slot_loss=tf.losses.softmax_cross_entropy(slot_one_hot,soft_logit)
            return slot_loss

        elif decoder_mod=='crf' or FLAGS.only_mode=='intent_slot_crf':

            out_w=tf.Variable(tf.random_normal(shape=(2*self.hidden_dim,self.slot_num_class)),dtype=tf.float32)
            out_b=tf.Variable(tf.random_normal(shape=(self.slot_num_class,)),dtype=tf.float32)
            # print(lstm_out)

            # 普通的lstm输出
            # decoder_in=tf.add(tf.einsum('ijk,kl->ijl',self.encoder_outs,out_w),out_b)
            # self.soft_logit = tf.nn.softmax(decoder_in,2)
            # slot_one_hot = tf.one_hot(self.slot, self.slot_num_class, 1, 0, axis=2)
            # slot_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(slot_one_hot, decoder_in))

            #lstm+attetion
            # lstm_out=self.self_lstm_attention(self.encoder_outs)
            # decoder_in=tf.add(tf.einsum('ijk,kl->ijl',lstm_out,out_w),out_b)
            # self.soft_logit = tf.nn.softmax(decoder_in, 2)
            # slot_one_hot = tf.one_hot(self.slot, self.slot_num_class, 1, 0, axis=2)
            # slot_loss = tf.losses.softmax_cross_entropy(slot_one_hot, decoder_in)

            # lstm+attetion_1
            lstm_out = self.attention(self.encoder_outs)
            out_w = tf.Variable(tf.random_normal(shape=(4 * self.hidden_dim, self.slot_num_class)), dtype=tf.float32)
            out_b = tf.Variable(tf.random_normal(shape=(self.slot_num_class,)), dtype=tf.float32)
            decoder_in = tf.add(tf.einsum('ijk,kl->ijl', lstm_out, out_w), out_b)
            self.slot_soft_logit = tf.nn.softmax(decoder_in,2)
            # slot_one_hot = tf.one_hot(self.slot, self.slot_num_class, 1, 0, axis=2)
            # slot_loss = tf.losses.softmax_cross_entropy(slot_one_hot,decoder_in)

            mask = tf.sequence_mask(self.seq_vec, self.max_len)
            mod = tf.losses.Reduction().NONE
            slot_loss = tf.losses.sparse_softmax_cross_entropy(self.slot, decoder_in, reduction=mod)
            losses = tf.boolean_mask(slot_loss, mask)
            slot_loss = tf.reduce_mean(losses)

            # bilstm+crf
            # decoder_in=tf.add(tf.einsum('ijk,kl->ijl',self.encoder_outs,out_w),out_b)
            #
            # log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            #     decoder_in, self.slot, self.seq_vec)
            # self.soft_logit_crf = decoder_in
            # self.trans_params = trans_params  # need to evaluate it for decoding
            # slot_loss = tf.reduce_mean(-log_likelihood)

            return slot_loss

        elif decoder_mod=='self_decoder':

            cell=tf.contrib.rnn.LSTMCell(2*self.hidden_dim)
            inputs=tf.unstack(self.sent_emb,self.max_len,1)

            # init_state = tf.zeros_like(inputs[0])
            # init_state=tf.contrib.rnn.LSTMStateTuple(
            #     c=init_state,
            #     h=init_state
            # )
            # states = [init_state]
            states=[self.encoder_final_states]
            init_h=self.encoder_final_states[0]
            H=[init_h]

            with tf.variable_scope('decoder'):
                for i in range(self.max_len):
                    if i>0:
                        tf.get_variable_scope().reuse_variables()
                    input_t=inputs[i]
                    a=self.self_lstm_attention_ops_decoder(input_t,self.encoder_outs)
                    new_i=tf.concat((a,input_t),1)
                    new_i=new_i+H[-1]
                    h,state=cell(new_i,states[-1])
                    H.append(h)
                    states.append(state)
            outs=tf.stack(H[1:],1)
            softmax_w = tf.Variable(
                tf.random_normal(shape=(2*self.hidden_dim, self.slot_num_class), dtype=tf.float32))
            softmax_b = tf.Variable(tf.random_normal(shape=(self.slot_num_class,), dtype=tf.float32))

            soft_logit = tf.add(tf.einsum('ijk,kl->ijl', outs, softmax_w), softmax_b)
            self.slot_soft_logit = tf.nn.softmax(soft_logit, 2)

            slot_one_hot = tf.one_hot(self.slot, self.slot_num_class, 1, 0, axis=2)
            # slot_loss = tf.losses.softmax_cross_entropy(slot_one_hot, soft_logit)

            mask = tf.sequence_mask(self.seq_vec, self.max_len)
            mod=tf.losses.Reduction().NONE
            slot_loss=tf.losses.sparse_softmax_cross_entropy(self.slot,soft_logit,reduction=mod)
            losses = tf.boolean_mask(slot_loss, mask)
            slot_loss = tf.reduce_mean(losses)

            return slot_loss

    def Encoder_Decoder(self):
        '''
        编码+解码
        :return:
        '''

        encoder_inputs=tf.unstack(self.sent,self.max_len,1)
        decoder_inputs=tf.unstack(self.sent,self.max_len,1)
        out_w = tf.Variable(tf.random_uniform(shape=(self.hidden_dim, self.slot_num_class), maxval=1.0, minval=-1.0),
                            dtype=tf.float32)
        out_b = tf.Variable(tf.random_uniform(shape=(self.slot_num_class,)), dtype=tf.float32)

        cell=tf.contrib.rnn.LSTMCell(self.hidden_dim)

        outs,state=tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs,
            cell,
            num_encoder_symbols=self.vocab_num,
            num_decoder_symbols=self.vocab_num,
            embedding_size=self.hidden_dim,
            # output_projection=(out_w,out_b),
            feed_previous=False,
            dtype=tf.float32)

        out_w_ = tf.Variable(tf.random_uniform(shape=(self.vocab_num, self.slot_num_class), maxval=1.0, minval=-1.0),
                            dtype=tf.float32)
        out_b_ = tf.Variable(tf.random_uniform(shape=(self.slot_num_class,)), dtype=tf.float32)
        outs=tf.stack(outs,1)
        outs=tf.add(tf.einsum('ijk,kl->ijl',outs,out_w_),out_b_)
        self.soft_logit = tf.nn.softmax(outs, 1)
        label_one_hot = tf.one_hot(self.slot, self.slot_num_class, 1, 0, 2)
        loss = tf.losses.softmax_cross_entropy(logits=outs, onehot_labels=label_one_hot)
        return loss

    def crf_acc(self, pre_label, real_label, rel_len):
        """

        :param best_path:
        :param path:
        :return:
        """
        real_labels_all = []
        for label, r_len in zip(real_label, rel_len):
            real_labels_all.extend(label[:r_len])

        verbit_seq_all = []
        for seq, r_len in zip(pre_label, rel_len):
            verbit_seq_all.extend(seq[:r_len])

        best_path = verbit_seq_all
        path = real_labels_all
        # ss = sum([1 for i, j in zip(best_path, path) if int(i) == int(j)])
        # length = sum([1 for i in path if int(i) != 0])
        if len(best_path) != len(path):
            print("error")
        else:

            ss = sum([1 for i, j in zip(best_path, path) if int(i) == int(j) and int(i) != 0])
            length = sum([1 for i, j in zip(best_path, path) if int(i) != 0 or int(j) != 0])
            acc = (float(ss) / float(length))
            return acc

    def Verbit(self, logits, batch_size, trans_params, sequence_lengths):

        viterbi_sequences = []
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit_ = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit_, trans_params)
            viterbi_sequences += [viterbi_seq]
        viterbi_sequences = viterbi_sequences
        return viterbi_sequences

    def train(self,dd):


        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=8,
                                intra_op_parallelism_threads=8,
                                log_device_placement=False,
                                allow_soft_placement=True,
                                )
        saver=tf.train.Saver()
        init_dev_loss=999.99
        num_batch=dd.num_batch
        with tf.Session(config=config) as sess:
            if os.path.exists('./save_model/model_dynamic.ckpt.meta'):
                saver.restore(sess,'./save_model/model_dynamic.ckpt')
            else:
                sess.run(tf.global_variables_initializer())
            dev_sent,dev_slot,dev_intent,dev_rel_len=dd.get_dev()
            for j in range(FLAGS.epoch):
                _logger.info('第%s次epoch'%j)
                for i in range(num_batch):
                    start_time=time.time()
                    sent,slot,intent,rel_len,cur_len=dd.next_batch()

                    if FLAGS.only_mode=='intent':
                        intent_loss,softmax_logit, _ = sess.run([self.loss_op,self.soft_logit, self.optimizer], feed_dict={self.sent: sent,
                                                                                             self.slot: slot,
                                                                                             self.intent: intent,
                                                                                             self.seq_vec: rel_len,
                                                                                             self.rel_num: cur_len
                                                                                             })
                        intent_train_acc=self.intent_acc(softmax_logit,intent)
                        _logger.info('index:%s'%i)
                        _logger.info('intent_train_loss:%s intent_train_acc:%s ' % (intent_loss,intent_train_acc))

                        dev_softmax_logit,dev_loss = sess.run([self.soft_logit,self.loss_op], feed_dict={self.sent: dev_sent,
                                                                         self.slot: dev_slot,
                                                                         self.intent: dev_intent,
                                                                         self.seq_vec: dev_rel_len,
                                                                         })
                        dev_intent_acc=self.intent_acc(dev_softmax_logit,dev_intent)
                        _logger.info('intent_dev_loss:%s intent_dev_acc:%s' %(dev_loss,dev_intent_acc))
                        if dev_loss < init_dev_loss:
                            init_dev_loss = dev_loss
                            saver.save(sess, './save_model/model_dynamic.ckpt')
                            _logger.info('save model')
                        _logger.info('\n')

                    elif FLAGS.only_mode=='slot':

                        losses,softmax_logit, _ = sess.run([self.loss_op, self.slot_soft_logit,self.optimizer], feed_dict={self.sent: sent,
                                               self.slot: slot,
                                               self.intent: intent,
                                               self.seq_vec: rel_len,
                                               self.rel_num: cur_len
                                               })
                        # train_acc=self.slot_acc(softmax_logit,slot,rel_len)
                        # _logger.info('slot_loss:%s slot_acc:%s'%(losses,train_acc))

                        # losses, _,logit,tran_param = sess.run([self.loss_op, self.optimizer,self.logit,self.trans_params], feed_dict={self.sent: sent,
                        #                                                                 self.slot: slot,
                        #                                                                 self.intent: intent,
                        #                                                                 self.seq_vec: rel_len,
                        #                                                                 self.rel_num: cur_len
                        #                                                                 })
                        #
                        # verbit_seq=self.Verbit(logit,None,tran_param,rel_len)
                        # crf_acc_train = self.crf_acc(verbit_seq, slot, rel_len)
                        # _logger.info('slot_loss:%s slot_acc:%s'%(losses,crf_acc_train))


                        # dev_softmax_logit,dev_loss = sess.run([self.soft_logit,self.loss_op], feed_dict={self.sent: dev_sent,
                        #                                                  self.slot: dev_slot,
                        #                                                  self.intent: dev_intent,
                        #                                                  self.seq_vec: dev_rel_len,
                        #                                                  })
                        # dev_acc=self.slot_acc(dev_softmax_logit,dev_slot)
                        # _logger.info('dev_slot_loss:%s dev_slot_acc:%s'%(dev_loss,dev_acc))
                        # if dev_loss < init_dev_loss:
                        #     init_dev_loss = dev_loss
                        #     saver.save(sess, './save_model/model_dynamic.ckpt')
                        #     _logger.info('save model')

                        _logger.info('\n')

                    elif FLAGS.only_mode=='intent_slot':

                        intent_loss,slot_loss, intent_logit,slot_logit, _ = sess.run([self.intent_losses,self.slot_loss, self.intent_soft_logit,self.slot_soft_logit, self.optimizer],
                                                                 feed_dict={self.sent: sent,
                                                                            self.slot: slot,
                                                                            self.intent: intent,
                                                                            self.seq_vec: rel_len,
                                                                            self.rel_num: cur_len
                                                                            })
                        intent_train_acc = self.intent_acc(intent_logit, intent)
                        train_slot_acc = self.slot_acc(slot_logit, slot,rel_len)
                        #
                        # _logger.info('epoch:%s index:%s' % (j,i))
                        # _logger.info('intent_train_loss:%s intent_train_acc:%s slot_loss:%s slot_acc:%s' % (intent_loss, intent_train_acc,slot_loss, train_slot_acc))

                        #
                        # dev_intent_loss, dev_slot_loss, dev_intent_logit, dev_slot_logit,  = sess.run(
                        #     [self.intent_losses, self.slot_loss, self.intent_soft_logit, self.slot_soft_logit],
                        #     feed_dict={self.sent: dev_sent,
                        #                self.slot: dev_slot,
                        #                self.intent: dev_intent,
                        #                self.seq_vec: dev_rel_len,
                        #                self.rel_num: cur_len
                        #                })
                        #
                        # dev_slot_acc = self.slot_acc(dev_slot_logit, dev_slot)
                        # intent_dev_acc = self.intent_acc(dev_intent_logit, dev_intent)
                        # _logger.info('intent_dev_loss:%s intent_dev_acc:%s slot_loss:%s slot_acc:%s' % (dev_intent_loss, intent_dev_acc,dev_slot_loss, dev_slot_acc))
                        #
                        # dev_loss=dev_slot_loss+dev_intent_loss
                        # if dev_loss < init_dev_loss:
                        #     init_dev_loss = dev_loss
                        #     saver.save(sess, './save_model/model_dynamic.ckpt')
                        #     _logger.info('save model')

                    elif FLAGS.only_mode=='intent_slot_crf':

                        intent_loss, slot_loss, intent_logit, slot_logit,tran_param, _ = sess.run(
                            [self.intent_losses, self.slot_loss, self.intent_soft_logit, self.soft_logit_crf,self.trans_params,
                             self.optimizer],
                            feed_dict={self.sent: sent,
                                       self.slot: slot,
                                       self.intent: intent,
                                       self.seq_vec: rel_len,
                                       self.rel_num: cur_len
                                       })

                        # intent_train_acc = self.intent_acc(intent_logit, intent)
                        # verbit_seq_train = self.Verbit(slot_logit, None, tran_param, rel_len)
                        # crf_acc_train = self.crf_acc(verbit_seq_train, slot, rel_len)
                        # # train_slot_acc = self.slot_acc(slot_logit, slot)
                        #
                        # _logger.info('epoch:%s index:%s' % (j, i))
                        # _logger.info('intent_train_loss:%s intent_train_acc:%s slot_loss:%s slot_acc:%s' % (
                        # intent_loss, intent_train_acc, slot_loss, crf_acc_train))

                dev_intent_loss, dev_slot_loss, dev_intent_logit, dev_slot_logit,  = sess.run(
                    [self.intent_losses, self.slot_loss, self.intent_soft_logit, self.slot_soft_logit],
                    feed_dict={self.sent: dev_sent,
                               self.slot: dev_slot,
                               self.intent: dev_intent,
                               self.seq_vec: dev_rel_len,
                               self.rel_num: cur_len
                               })

                dev_slot_acc = self.slot_acc(dev_slot_logit, dev_slot,dev_rel_len)
                intent_dev_acc = self.intent_acc(dev_intent_logit, dev_intent)
                _logger.info('intent_dev_loss:%s intent_dev_acc:%s slot_loss:%s slot_acc:%s' % (dev_intent_loss, intent_dev_acc,dev_slot_loss, dev_slot_acc))


                # dev_intent_loss, dev_slot_loss, dev_intent_logit, dev_slot_logit,tran_param = sess.run(
                #     [self.intent_losses, self.slot_loss, self.intent_soft_logit, self.soft_logit_crf,self.trans_params],
                #     feed_dict={self.sent: dev_sent,
                #                self.slot: dev_slot,
                #                self.intent: dev_intent,
                #                self.seq_vec: dev_rel_len,
                #                })
                #
                # # dev_slot_acc = self.slot_acc(dev_slot_logit, dev_slot)
                # intent_dev_acc = self.intent_acc(dev_intent_logit, dev_intent)
                # verbit_seq_dev = self.Verbit(dev_slot_logit, None, tran_param, dev_rel_len)
                # crf_acc_dev = self.crf_acc(verbit_seq_dev, dev_slot, dev_rel_len)
                # _logger.info('intent_dev_loss:%s intent_dev_acc:%s slot_loss:%s slot_acc:%s' % (
                # dev_intent_loss, intent_dev_acc, dev_slot_loss, crf_acc_dev))

                # dev_loss = dev_slot_loss + dev_intent_loss
                # if dev_loss < init_dev_loss:
                #     init_dev_loss = dev_loss
                #     saver.save(sess, './save_model/model_dynamic.ckpt')
                #     _logger.info('save model')


                endtime=time.time()
                print('time:%s'%(endtime-start_time))
                _logger.info('\n')


    def intent_acc(self,pre,label):
        '''
        获取intent准确率
        :param pre:
        :param label:
        :return:
        '''
        all_sum=len(pre)
        pre=np.argmax(pre,1)
        num=sum([1 for e,e1 in zip(pre,label) if e==e1])
        return float(num)/float(all_sum)

    def slot_acc(self,pre,label,rel_len):
        '''
        slot准确率
        :param pre:
        :param label:
        :return:
        '''
        pre=np.argmax(pre,2)
        ss=0.0
        for e,e1,rl in zip(pre,label,rel_len):
            num=0.0
            for i in range(rl):
                if e[i]==e1[i]:
                    num+=1.0
            acc=num/float(rl)
            ss+=acc
        return ss/float(len(pre))




if __name__ == '__main__':
    start_time = time.time()
    with tf.device("/cpu:0"):
        _logger.info("load data")
        dd = Intent_Slot_Data(train_path="./dataset/atis-2.train.w-intent.iob",
                              test_path="./dataset/atis.test.w-intent.iob",
                              dev_path="./dataset/atis.test.w-intent.iob", batch_size=FLAGS.batch_size, max_length=FLAGS.max_len, flag="train_new",
                              use_auto_bucket=FLAGS.use_auto_buckets)

        # sent, slot, intent, rel_len, cur_len = dd.next_batch()
        # _logger.info('input_param:{}'.format(sent.shape,slot.shape,intent.shape,rel_len.shape,cur_len.shape))
        nn_model = Model(slot_num_class=dd.slot_num,intent_num_class=dd.intent_num,vocab_num=dd.vocab_num)
        nn_model.train(dd)