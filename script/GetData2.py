# coding=utf-8
import pymysql
conn = pymysql.connect(host='localhost', user='root', passwd='0123456', db='tang_poetry', charset='utf8')
cur = conn.cursor()
count = cur.execute("\
    SELECT poetries.title, poets.name, poetries.content \
    from poetries join poets \
    where poetries.poet_id = poets.id;")
result = cur.fetchall()
output = open('poetry_data2', 'a', encoding="utf-8")
for i in range(0, count):
    str=result[i][0]+u" "+result[i][1]+u" "+result[i][2]+u" "+ "\n"
    output.write(str)
    if i % 100 == 0:
        print(i,u"/",count)
output.close()
cur.close()
conn.close()

