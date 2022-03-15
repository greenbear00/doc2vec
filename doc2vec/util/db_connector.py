import pymssql
import pandas as pd


def get_news_db(server: str, user: str, password: str, database: str, port: int = 1433, is_autocommit=False):
	conn = pymssql.connect(server=server, port=port, user=user, password=password, database=database,
						   autocommit=is_autocommit)
	return conn


def do_exe_query(db_conn, query: str):
	cursor = db_conn.cursor(as_dict=True)
	try:

		cursor.execute(query)
		db_conn.commit()
	except Exception as es:
		print(f"error= {es}")
		db_conn.rollback()
	finally:
		cursor.close()
		db_conn.close()


def search_query(db_conn, query: str):
	df = pd.DataFrame()
	cursor = db_conn.cursor(as_dict=True)
	try:
		cursor.execute(query)
		items = cursor.fetchall()
		df = pd.DataFrame(items)
	except Exception as es:
		print(f"Error = {es}")
	finally:
		cursor.close()
		db_conn.close()
	return df
