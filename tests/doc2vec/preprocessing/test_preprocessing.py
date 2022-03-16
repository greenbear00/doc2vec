import unittest
from datetime import datetime, timedelta
from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
from doc2vec.util.common import *
import rootpath
import pandas as pd
import os
from doc2vec.preprocessing.preprocessing import *
from doc2vec.util.common import load_pickle


class TestPreProcessing(unittest.TestCase):
	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger

		self.p = Parser()

	def test_do_generate_tagged_documment(self):

		data = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]))

		d2v_result = generate_tagged_document(data)
		print(d2v_result)

	def test_load_user_history(self):

		user_df = user_history_df("user_history.csv")
		print(user_df.tail())

	def test_load_pickle(self):

		data = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]))

		for k, v in data.items():
			print(k, v[:10])




	def test_load_history(self):
		"step 1. 중복제거하여 2021.11~2022.02까지 사용자 history 생성: 211523"
		data_path = os.path.join(rootpath.detect(), *['data', 'jtbcnews_user_history_20211101_20220201.csv'])
		old_df = load_pdf(file_path=data_path)

		data_path = os.path.join(rootpath.detect(), *['data', 'jtbcnews_user_history.csv'])
		new_df = load_pdf(file_path=data_path)
		new_df['reqtime'] = new_df['reqtime'].apply(lambda x: datetime.strptime(x, "%d/%b/%Y:%H:%M:%S +0900"))

		print("old_df: ", old_df.shape)
		print("new_df: ", new_df.shape)
		df= pd.concat([old_df, new_df], axis=0)
		print("df: ", df.shape)


		new_df = df.drop_duplicates(['remote', 'reqtime', 'method', 'path', 'parms', 'code', 'size', 'agent',
									 'referer', 'xforward', 'reqbody', 'timestamp', 'hostname',
									 'es_index_name', 'action_name', 'parm_referer', 'parm_current_path',
									 'parm_member_id', 'parm_user_id', 'parm_member_type', 'parm_origin',
									 'ua_browser_family', 'ua_browser_version', 'ua_os_family',
									 'ua_os_version', 'ua_os_major_version', 'ua_device', 'news_nm', 'tag',
									 'news_id', 'ua_browser_major_version', 'action_type'], keep='last')
		print("remove duplicate df: ", new_df.shape)
		new_df['reqtime'] = pd.to_datetime(new_df['reqtime'])
		new_df = new_df.sort_values(by='reqtime')
		new_df.reset_index(drop=True, inplace=True)

		user_df = new_df[['reqtime', 'referer', 'action_name', 'parm_member_id', 'parm_member_type', 'news_id', 'news_nm',
						'parm_user_id', 'parm_origin',
						 'ua_browser_family', 'ua_browser_version', 'ua_os_family',
						 'ua_os_version', 'ua_os_major_version', 'ua_device', 'ua_browser_major_version'
		]].copy()
		print(user_df.shape)
		user_df = user_df[(~user_df.parm_member_id.isnull()) & (~user_df.news_id.isnull())]

		# 결측치 제거 : 616061 ->  211523 (viewpage가 페이지 로딩부터 시작이므로, 데이터가 없어서 사라짐)
		print("remove..", user_df.shape)
		write_pdf(user_df, "jtbcnews-raw-20211101-20220228.csv")

	def test_check_user_history_csv(self):
		"step2. 해당 기간동안 본 news_id 수: 15098"
		df = load_pdf('jtbcnews-raw-20211101-20220228.csv')
		print(df.shape)
		print(df.action_name.unique())

		unique_news_ids = df.news_id.unique().tolist()
		print("해당 기간동안 본 news_id 수: ", len(unique_news_ids))

		save_pickle(os.path.join(rootpath.detect(), *['data', 'rec', 'unique_news_ids.pickle']), unique_news_ids)




if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestPreProcessing)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())