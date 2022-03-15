import unittest
from datetime import datetime, timedelta
from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
import rootpath
import pandas as pd
import os

from doc2vec.util.db_connector import *
from doc2vec.preprocessing.preprocessing import load_pdf, reform_news_df


class TestDBConnector(unittest.TestCase):
	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger

		self.p = Parser()

	def test_save_news_df_to_csv(self):
		"""db에서 사용자 history기반 데이터 가져오기"""
		history_df = load_pdf('user_history.csv')
		news_ids = history_df.news_id.to_list()

		news_df = pd.DataFrame(columns=['news_id', 'title', 'section_name', 'service_dt','ARTICLE_CONTENTS_BULK'])
		step = 100
		for idx in range(0, len(news_ids), step):
			db_conn = get_news_db(**self.p.db_conf)
			print(f"{idx}/{len(news_ids)}")
			if idx + 3 > len(news_ids):
				news_11_query = f"('{news_ids[0]}')" if len(news_ids) == 1 else tuple(
					map(str, news_ids[idx:len(news_ids)]))
			else:
				news_11_query = f"('{news_ids[0]}')" if len(news_ids) == 1 else tuple(
					map(str, news_ids[idx:idx + step]))

			query = f"""
					select a.news_id, a.title, a.section_name, a.service_dt, b.ARTICLE_CONTENTS_BULK
					from VI_ELK_NEWS_BASIC a
						join TB_NEWS_CONTENTS_BULK b
						on a.news_id = b.news_id
					where a.news_id in {news_11_query}
				"""
			df = search_query(db_conn, query)
			if not df.empty:
				news_df = pd.concat([news_df, df], axis=0)

		# 사용자 history 기반 news 수 : 65144
		news_df.to_csv(os.path.join(rootpath.detect(), *['data', "rec", "ori_news.csv"]), sep="^")

	def test_reform_csv(self):
		"분석할 수 있는 데이터 형식으로 변환 "
		df = load_pdf("ori_news.csv", sep="^")
		reform_news_df(df)
		# 사용자 history 기반 news 수 : 65144
		#          news_id  ...                                       news_content
		# 65139  NB12045093  ...  <div id='div_NV10478125' ...
		# 65140  NB12045094  ...  <div id='div_NV10478127' ...




if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestDBConnector)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())