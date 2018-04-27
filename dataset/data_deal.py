import jieba
import random
import re
fw_train=open('./origin_data_train_new.txt','w')
fw_dev=open('./origin_data_dev_new.txt','w')

sub_char=['?','？','，','、']
pattern="\\?|？|，|、|\\.|。|“|”|》"
res=[]
# for line in open('./origin_data.txt','r').readlines():
for line in open('./write.txt', 'r').readlines():

    line=line.replace('\n','')
    line=re.subn(pattern,'',line)[0]
    print(line)
    lines=line.split('\t\t')
    label=lines[0].replace('\t','')
    ele=lines[1].replace('\t','')
    ss=['BOS']
    token_ls=[e for e in jieba.cut(ele)]
    ss.extend(token_ls)
    ss.append('EOS')
    token_ls=ss
    slot_list=['O']*len(token_ls)
    token=" ".join(token_ls)
    slot=" ".join(slot_list)
    res.append([token,slot,label])

index = [i for i in range(len(res))]
random.shuffle(index)
new_res=[res[e] for e in index]

dev_res=new_res[:300]
train_res=new_res[300:]

for ele in dev_res:
    fw_dev.write(ele[0])
    fw_dev.write('\t\t')
    fw_dev.write(ele[1])
    fw_dev.write('\t\t')
    fw_dev.write(ele[2])
    fw_dev.write('\n')

for ele in train_res:
    fw_train.write(ele[0])
    fw_train.write('\t\t')
    fw_train.write(ele[1])
    fw_train.write('\t\t')
    fw_train.write(ele[2])
    fw_train.write('\n')



