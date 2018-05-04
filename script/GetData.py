import pymysql
conn = MySQLdb.connect(host='localhost', user='root', passwd='0123456', db='tang_poetry')
cur = conn.cursor()
count = cur.execute("SELECT poetries.content, poetries.title, poet., book.Book_Name, borrowTime, returnTime, Admin_ID \
                   FROM Book NATURAL JOIN Record NATURAL JOIN Card WHERE Card.Card_ID = '%s'" % cardID)
result = cur.fetchall()

cur.close()
conn.close()

