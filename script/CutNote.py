import re
import string
import jieba
import jieba.analyse

#import zhon
#from zhon.hanzi import punctuation

file = open("poetry_data", "r", encoding="utf-8")
instr = file.read()
#deletenote = string.punctuation + zhon.hanzi.punctuation
#去除标点
out = re.sub("[，。（）！【】“”？/·《》‘’；：|]", "", instr)

#分词
output = jieba.cut(out)
#print(" ".join(output))

fileout = open("poetry_data_cut", "w",encoding="utf-8")
fileout.write(u" ".join(output))
fileout.close()