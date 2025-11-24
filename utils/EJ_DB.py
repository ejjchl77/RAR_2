import pymysql

def upload(sql, val):
    conn = pymysql.connect(host='192.168.0.2',
               user='pbcl',
               password='pbcl7896',
               db='testbed',
               charset='utf8')

    cursor = conn.cursor()

    cursor.execute(sql, val)

    conn.commit()
