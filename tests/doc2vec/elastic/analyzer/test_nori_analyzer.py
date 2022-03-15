import unittest
from datetime import datetime, timedelta

from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
from doc2vec.elastic.analyzer.nori_analyzer import NoriAnalyzer
import rootpath
import pandas as pd
import os
from doc2vec.util.common import load_pdf, write_pdf, load_pickle, save_pickle


class TestNoriAnalyzer(unittest.TestCase):
	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger

		self.p = Parser()

		self.nori = NoriAnalyzer(**self.p.elastic_conf)

	def _get_naver_airs_section_news(self):
		ranking_news_li = [
			{'news_nm': '[전문] 이재명 "여의도 정치 확 바꿀 것... 네거티브 안 한다"', 'url': 'https://news.naver.com/main/read.naver',
			 'service_date': '2022-01-26T09:23:00+09:00', 'company': '오마이뉴스', 'section': '정치', 'platform': 'naver',
			 'category': 'AiRS 추천뉴스'},
			{'news_nm': '김 총리 "확진자 1만 3천 명 넘어…29일부터 신속항원검사 전국 확대"', 'url': 'https://news.naver.com/main/read.naver',
			 'service_date': '2022-01-26T09:32:00+09:00', 'company': '노컷뉴스', 'section': '정치', 'platform': 'naver',
			 'category': 'AiRS 추천뉴스'},
			{'news_nm': '윤석열 44.7%·이재명 35.6%·안철수 9.8%', 'url': 'https://news.naver.com/main/read.naver',
			 'service_date': '2022-01-26T10:01:00+09:00', 'company': '한국일보', 'section': '정치', 'platform': 'naver',
			 'category': 'AiRS 추천뉴스'},
			{'news_nm': "이스라엘은 방역패스 폐지, 한국은 '강화'…어떤 선택이 맞나?", 'url': 'https://news.naver.com/main/read.naver',
			 'service_date': '2022-01-26T07:03:00+09:00', 'company': '뉴스1', 'section': '정치', 'platform': 'naver',
			 'category': 'AiRS 추천뉴스'}]

		return ranking_news_li

	def _get_ranking_news_li_news(self):
		ranking_news_li = [
			{'news_nm': '김건희 "바보같은 보수" 폄훼에…원팀·지지층 실망 우려', 'url': 'https://n.news.naver.com/article/003/0010948194',
			 'service_date': '2022-01-17T10:48:00+09:00', 'company': 'newsis', 'ranking_num': 1, 'platform': 'naver',
			 'reg_date': '2022-01-17T00:00:00+09:00'}]

		return ranking_news_li

	def test_nori_analyzer_for_naver_ranking_news(self):
		the_date = datetime.now()
		data = self._get_ranking_news_li_news()
		result = self.nori.get_parsing_news_nm_by_nori_analyzer(the_date, data)
		self.logger.info(result)

		self.assertEqual(list(result[0].keys()), ["news_nm", "url", "service_date", "company", "ranking_num",
												  "platform",
											"reg_date", "news_nm_analyzer"], "nori_analyzer not passed")

	def _get_new_contents_by_date_range(self, start_date, end_date):
		"""
		특정 기간 내에 test-doc2vec index에서 news_id, news_content를 추출하여 html 태그 제거 등을 통해서
		뉴스 기사에 대해서 형태소 분석한 결과를 dict(key는 news_id, value는 형태소 분석한 list)으로 리턴
		:param start_date:
		:param end_date:
		:return:
			flag(bool) : 데이터를 가지고 오는 날 별로 처리가 잘 되었는지에 대한 flag
			parsing_data(dict) : key는 news_id, value는 형태소 분석한 list
		"""
		parsing_data = {}
		news_data = []

		flag = True
		while start_date <= end_date:
			flag, result, tmp_news_data = self.nori.get_parsing_news_contents_from_elastic(start_date)
			if not flag:
				break
			parsing_data.update(result)
			news_data.extend(tmp_news_data)
			start_date = start_date + timedelta(days=1)

		news_data_df = pd.DataFrame(news_data)

		return flag, parsing_data, news_data_df

	def test_verify_calculate_doc2vec(self):
		df = load_pdf("news_all.csv", sep="^")
		new_df = df[df['news_id'].isin(['NB12035622','NB12035501',  "NB12035460", "NB12035342", "NB12035304",
							   "NB12035183", "NB12035244", "NB12035069", "NB12035079", "NB12034593"])]
		print(new_df)
		tmp_parsing = self.nori.get_parsing_news_contents_from_csv(new_df)


	def test_calculate_doc2vec(self):
		save_flag = True
		# 현재 11월달 꺼
		df = load_pdf("news_1_reform.csv", sep="^")
		print(df.tail())
		print(df.shape)
		step = 10

		news_contents_parsing = {}
		for idx in range(0, df.shape[0], step):
			if idx + step > df.shape[0]:
				self.logger.info(f"{idx} ~ {df.shape[0]} / {df.shape[0]}")
				parsing_df = df.iloc[idx: df.shape[0]][['news_id','news_content']]
			else:
				self.logger.info(f"{idx} ~ {idx+step} / {df.shape[0]}")
				parsing_df = df.iloc[idx: idx+step][['news_id','news_content']]

			tmp_parsing = self.nori.get_parsing_news_contents_from_csv(parsing_df)
			news_contents_parsing.update(tmp_parsing)

		if save_flag:
			save_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data_1.pickle"]), news_contents_parsing)

	def test_get_parsing_news_contents_by_nori_analyzer(self):
		"""
		elastic test-doc2vec에 있는 데이터를 가져와서 형태소 분석을 수행한다.
		:return:
		"""
		save_flag = True
		start_date = datetime.now().replace(year=2021, month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
		end_date = datetime.now().replace(year=2021, month=12, day=1, hour=23)

		flag, parsing_data, news_data_df = self._get_new_contents_by_date_range(start_date=start_date,
																				end_date=end_date)
		if save_flag:
			save_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]), parsing_data)
			write_pdf(news_data_df, "news.csv", sep="^")

		self.assertEqual(flag, True, f"{start_date} upsert error")


	def test_put_data_to_doc2vec(self):
		"""
		지정해 놓은 test-doc2vec index에 샘플 데이터(data/news_train_tmp.csv) 파일을 write 한다.
		:return:
		"""
		index = 'test-doc2vec'
		self.nori.update_index_setting()
		file_path = os.path.join(rootpath.detect(), *['data', 'news_train_tmp.csv'])

		df = pd.read_csv(file_path, delimiter=',', header=0)
		self.logger.info(df.tail())
		self.nori.put_data(df.to_dict(orient='records'))











if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestNoriAnalyzer)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())