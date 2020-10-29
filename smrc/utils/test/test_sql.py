import MySQLdb
import sqlite3
# connection = sqlite3.connect("company.db")
#
# sql_command = """
# CREATE TABLE employee (
# staff_number INTEGER PRIMARY KEY,
# fname VARCHAR(20),
# lname VARCHAR(30),
# gender CHAR(1),
# joining DATE,
# birth_date DATE);"""


connection = sqlite3.connect("company.db")
cursor = connection.cursor()

# delete
#cursor.execute("""DROP TABLE employee;""")

sql_command = """
CREATE TABLE employee ( 
staff_number INTEGER PRIMARY KEY, 
fname VARCHAR(20), 
lname VARCHAR(30), 
gender CHAR(1), 
joining DATE,
birth_date DATE);"""

cursor.execute(sql_command)

sql_command = """INSERT INTO employee (staff_number, fname, lname, gender, birth_date)
    VALUES (NULL, "William", "Shakespeare", "m", "1961-10-25");"""
cursor.execute(sql_command)


sql_command = """INSERT INTO employee (staff_number, fname, lname, gender, birth_date)
    VALUES (NULL, "Frank", "Schiller", "m", "1955-08-17");"""
cursor.execute(sql_command)

# never forget this, if you want the changes to be saved:
connection.commit()

connection.close()

#
# connection = MySQLdb.connect (host = "localhost",
#                               user = "testuser",
#                               passwd = "testpass",
#                               db = "company")
#
# cursor = connection.cursor()
# cursor.execute ("SELECT VERSION()")
# row = cursor.fetchone()
# print("server version:", row[0])
# cursor.close()
# connection.close()