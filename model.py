import numpy as np
import sys
import os
import random
import tensorflow as tf
import logging
from logging.config import fileConfig
import time
from data_preprocess import Intent_Slot_Data

sys.path.append("./")
sys.path.append("./data/")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger = logging.getLogger("intent_bagging")


class Config(object):
    '''
    默认配置
    '''
    learning_rate = 0.2
    batch_size = 200
    sent_len = 30  # 句子长度
    embedding_dim = 50  # 词向量维度
    hidden_dim = 100
    train_dir = './data/train_out_%s.txt'
    dev_dir = './data/dev_out.txt'
    test_dir = './data/test.txt'
    model_dir = './save_model_1/model_%s/r_net_model_%s.ckpt'
    if not os.path.exists('./save_model/model_%s_'):
        os.makedirs('./save_model/model_%s_')
    use_cpu_num = 8
    keep_dropout = 0.7
    summary_write_dir = "./tmp/r_net.log"
    epoch = 10
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
tf.app.flags.DEFINE_string('only_mode','intent','执行哪种单一任务')
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
        self.encoder_outs,self.encoder_final_states=self.encoder()

        if FLAGS.only_mode=='intent':

            self.intent_losses=self.intent_loss()
            self.loss_op=self.intent_losses
        elif FLAGS.only_mode=='slot':
            self.slot_loss=self.decoder()
            self.loss_op=self.slot_loss
        else:
            self.intent_losses=self.intent_loss()
            self.slot_loss=self.decoder()
            self.loss_op=self.intent_losses+self.slot_loss
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)

        grads_vars = self.opt.compute_gradients(self.loss_op)

        capped_grads_vars = [[tf.clip_by_value(g, -5.0, 5.0), v] for g, v in grads_vars]


        self.optimizer = self.opt.apply_gradients(capped_grads_vars)

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
            intent_one_hot = tf.one_hot(self.intent, self.intent_num_class, 1, 0)
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
        lstm_cell=tf.contrib.rnn.LSTMCell(2*self.hidden_dim,state_is_tuple=True)
        decoder_list=tf.unstack(self.slot_emb,self.max_len,1)
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
        self.soft_logit=tf.nn.softmax(soft_logit,2)

        slot_one_hot=tf.one_hot(self.slot,self.slot_num_class,1,0,axis=2)
        slot_loss=tf.losses.softmax_cross_entropy(slot_one_hot,self.soft_logit)
        return slot_loss


    def train(self,dd):

        saver=tf.train.Saver()
        init_dev_loss=99
        config = tf.ConfigProto(device_count={"CPU": FLAGS.use_cpu_num},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=2,
                                intra_op_parallelism_threads=2,
                                log_device_placement=False)
        with tf.Session(config=config) as sess:
            if os.path.exists('./model.ckpt.meta'):
                saver.restore(sess,'./model.ckpt')
                _logger.info('load paramter')
            else:
                sess.run(tf.global_variables_initializer())
            dev_sent,dev_slot,dev_intent,dev_rel_len=dd.get_dev()
            for _ in range(1000):
                sent,slot,intent,rel_len,cur_len=dd.next_batch()
                if FLAGS.only_mode=='intent':
                    intent_loss,_=sess.run([self.loss_op,self.optimizer],feed_dict={self.sent:sent,
                                                    self.slot:slot,
                                                    self.intent:intent,
                                                    self.seq_vec:rel_len,
                                                    self.rel_num:cur_len
                                                    })
                    print('intent_train:%s '%(intent_loss))
                    [dev_loss]= sess.run([self.loss_op], feed_dict={self.sent: dev_sent,
                                                                     self.slot: dev_slot,
                                                                     self.intent: dev_intent,
                                                                     self.seq_vec: dev_rel_len,
                                                                     })
                    print('intent_dev:%s'%dev_loss)
                    print('\n')
                    if dev_loss<init_dev_loss:
                        init_dev_loss=dev_loss
                        saver.save(sess,'./model.ckpt')
                        _logger.info('save model')


                elif FLAGS.only_mode=='slot':
                    losses, _ = sess.run([self.loss_op, self.optimizer], feed_dict={self.sent: sent,
                                           self.slot: slot,
                                           self.intent: intent,
                                           self.seq_vec: rel_len,
                                           self.rel_num: cur_len
                                           })

                    print('slot:%s ' % (losses))
                else:

                    intent_loss, slot_loss, losses, _ = sess.run(
                        [self.intent_losses, self.slot_loss, self.loss_op, self.optimizer], feed_dict={self.sent: sent,
                                                                                                       self.slot: slot,
                                                                                                       self.intent: intent,
                                                                                                       self.seq_vec: rel_len,
                                                                                                       self.rel_num: cur_len
                                                                                                       })
                    print('intent:%s slot:%s all:%s' % (intent_loss, slot_loss, losses))







if __name__ == '__main__':
    start_time = time.time()
    with tf.device("/cpu:0"):
        _logger.info("load data")
        dd = Intent_Slot_Data(train_path="./dataset/atis.train.w-intent.iob",
                              test_path="./dataset/atis.test.w-intent.iob",
                              dev_path="./dataset/atis.dev.w-intent.iob", batch_size=FLAGS.batch_size, max_length=FLAGS.max_len, flag="train",
                              use_auto_bucket=FLAGS.use_auto_buckets)



        nn_model = Model(slot_num_class=dd.slot_num,intent_num_class=dd.intent_num,vocab_num=dd.vocab_num)
        nn_model.train(dd)