import re
import string
import jieba
import jieba.analyse

#import zhon
#from zhon.hanzi import punctuation

file = open("poetry_data3", "r", encoding="utf-8")
instr = file.read()
#deletenote = string.punctuation + zhon.hanzi.punctuation
#去除标点和数字
out = re.sub("[，。（）！【】、“”？,.*?/·X《》‘’；：|1234567890  　]", "", instr)

#分词
output = jieba.cut(out)
#print(" ".join(output))

fileout = open("poetry_data_cut3", "w",encoding="utf-8")
fileout.write(u" ".join(output))
fileout.close()