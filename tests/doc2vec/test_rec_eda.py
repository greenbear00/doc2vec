import unittest
from datetime import datetime, timedelta
from pathlib import Path
from doc2vec.elastic.analyzer.nori_analyzer import NoriAnalyzer
from doc2vec.model.doc2vec import build_model
from doc2vec.preprocessing.preprocessing import generate_tagged_document, generate_tagged_document_by_pandas, \
	load_pdf, is_contain_str
from doc2vec.util.Logger import Logger
import rootpath
import pandas as pd
import os
from doc2vec.util.common import load_pickle, save_pickle
from doc2vec.util.conf_parser import Parser
import gensim
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
from doc2vec.util.common import set_pandas_format, set_matplotlib_sns_font
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class TestEDA(unittest.TestCase):

	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger
		self.p = Parser()
		set_pandas_format()
		set_matplotlib_sns_font()

		self.nori = NoriAnalyzer(**self.p.elastic_conf)

	def test_calculate_d2v_tagdocument(self):
		""" step 1) 뉴스 데이터(2021.11-2022.01)를 읽어서 elastic기반 형태소 분석한 데이터를 pickle로 떨구기"""
		save_flag = True

		step = 10
		ori_df = load_pdf("news.csv", sep="^")
		print(ori_df.shape)
		ori_df.drop_duplicates(['news_id'], keep='last', inplace=True)

		# 제거할 거 (news_nm기준으로 제거, 13,405 -> 12,553로 의미 없는 기사 제거 )
		# - 클로징, BGM, 썰전 라이브 다시보기(-> 이건 text 내용이 없음)
		# 사용자 history 기반 news 수 :
		#          news_id  ...                                       news_content
		# 65139  NB12045093  ...  <div id='div_NV10478125' ...
		# 65140  NB12045094  ...  <div id='div_NV10478127' ...
		ori_df['is_in_str'] = ori_df['news_nm'].apply(is_contain_str)
		df = ori_df[~ori_df['is_in_str']==True].copy()
		df.reset_index(drop=True, inplace=True)
		print(df.shape)

		analyzed_df = pd.DataFrame(columns=['news_id', 'tagged_doc'])

		news_contents_parsing = {}
		self.nori.update_index_setting()
		for idx in range(0, df.shape[0], step):
			if idx + step > df.shape[0]:
				self.logger.info(f"{idx} ~ {df.shape[0]} / {df.shape[0]}")
				parsing_df = df.iloc[idx: df.shape[0]][['news_id', 'news_content', 'news_nm']]
			else:
				self.logger.info(f"{idx} ~ {idx + step} / {df.shape[0]}")
				parsing_df = df.iloc[idx: idx + step][['news_id', 'news_content', 'news_nm']]

			tmp_parsing = self.nori.get_parsing_news_contents_from_csv(parsing_df)
			news_contents_parsing.update(tmp_parsing)
			result_df = pd.DataFrame({'news_id': list(tmp_parsing.keys()), 'tagged_doc': list(tmp_parsing.values())})
			analyzed_df = pd.concat([analyzed_df, result_df], axis=0)

		if save_flag:
			save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "parsing_data.pickle"]), news_contents_parsing)
			save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "parsing_data_df.pickle"]),
						analyzed_df)

	def test_check_word_dist(self):
		from nltk.probability import FreqDist
		parsing_df = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "parsing_data_df.pickle"]))

		word_split = []
		for i in range(len(parsing_df)):
			for j in parsing_df.iloc[i]['tagged_doc']:
				word_split.append(j)

		FreqDist(word_split).plot(50)

		wv = WordCloud(font_path='/System/Library/Fonts/Supplemental/AppleGothic.ttf',
					   background_color='white', max_words=100, max_font_size=300,
					   width=1000, height=1000)
		wv.generate(str(parsing_df['tagged_doc']))
		plt.imshow(wv)
		plt.show()



	def test_generate_d2v_tagducument(self):
		""" step 2) elastic기반 corpus를 tagged_doc으로 변환"""
		# parsing_data_path = os.path.join(rootpath.detect(), *["data", "rec", "parsing_data.pickle"])
		# parsing_data = load_pickle(parsing_data_path)
		from sklearn.model_selection import train_test_split

		parsing_data_df_path = os.path.join(rootpath.detect(), *["data", "rec", "parsing_data_df.pickle"])
		parsing_data_df = load_pickle(parsing_data_df_path)

		x_train, x_test, y_train, y_test = train_test_split(parsing_data_df[['tagged_doc']],
															parsing_data_df['news_id'],
															test_size=0.3,
															shuffle=True,
															random_state=42)
		x_train = x_train.reset_index(drop=True)
		x_test = x_test.reset_index(drop=True)
		y_train = y_train.reset_index(drop=True)
		y_test = y_test.reset_index(drop=True)
		print("x_train.shape= ", x_train.shape, ", y_train.shape= ", y_train.shape)
		print("x_test.shape= ", x_test.shape, ", y_test.shape= ", y_test.shape)

		train_tagged_doc = generate_tagged_document_by_pandas(x_train, y_train)
		test_tagged_doc = generate_tagged_document_by_pandas(x_test, y_test)

		# train_tagged_doc.pickle
		# - [TaggedDocument(words=[...], tags=['NB12043728']), ....]
		save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_tagged_doc.pickle"]), train_tagged_doc)
		save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "test_tagged_doc.pickle"]), test_tagged_doc)

	def test_save_model(self):
		""" step 3) tagged_doc 기반으로 doc2vec 모델 생성 (70%의 document로 훈련) """
		tagged_doc = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_tagged_doc.pickle"]))

		d2v_model = build_model()
		d2v_model.build_vocab(tagged_doc)
		d2v_model.train(tagged_doc, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

		d2v_model.save(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))

	def test_model_test(self):
		test_tagged_doc = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "test_tagged_doc.pickle"]))

		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))

		print(test_tagged_doc[0])
		vector = d2v_model.infer_vector(test_tagged_doc[0].words)
		print("vector size = ", len(vector))
		print("Top 10 values in Doc2Vec inferred vecotr: ")
		print(vector[:10])

	def test_d2vmodel_tags(self):
		"""학습시킨 해당 모델의 tags들이 뭐가 있는지 확인하는 방법"""
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))
		print(d2v_model.docvecs.index_to_key)

	def test_d2vmodel_infer_vector(self):
		""" 학습 시킨 데이터 전체적으로 infer시켜서 df로 만듬"""
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))
		train_tagged_doc = load_pickle(os.path.join(rootpath.detect(), *['data', 'rec', 'train_tagged_doc.pickle']))

		vector_df = pd.DataFrame(columns = [str(ele) for ele in range(1000)])
		for idx in tqdm(len(train_tagged_doc), desc="infer_vector...", mininterval=30):
		# for idx, tagged_doc in enumerate(train_tagged_doc):
			vector = d2v_model.infer_vector(train_tagged_doc[idx].words)
			# if idx == 0:
			# 	header = ",".join(str(ele) for ele in range(1000))
			# line1 = ",".join([str(vector_ele) for vector_ele in vector])
			# print(len(vector), " -> ", line1[:100])
			vector_df.loc[len(vector_df)] = vector

		save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "vector_df.pickle"]), vector_df)



	def _check_sim_news(self, data, news_df, d2v_model):

		for a_data in data:
			selected_news_id = a_data.tags[0]
			selected_news_df = news_df[news_df['news_id'] == selected_news_id][['news_id', 'news_nm', 'news_content']]
			print(
				"\n=============================================================================================================")
			print(f"[{selected_news_id}] {selected_news_df.news_nm.values[0]}")
			inferred = d2v_model.infer_vector(a_data.words)
			sims = d2v_model.dv.most_similar(inferred, topn=5)

			# y_train, features
			# targets, feature_vectors = zip(*[(selected_news_id, d2v_model.infer_vector(a_data.words))])

			for sim in sims:
				news_id = sim[0]
				sim_value = sim[1]
				tmp_sim_news_df = news_df[news_df['news_id'] == news_id][
					['news_id', 'news_nm', 'news_content']]
				if not tmp_sim_news_df.empty:
					print(f" 연관추천 [{news_id} ({sim_value})] {tmp_sim_news_df.news_nm.values[0]}")
			print("\n\n")

	def test_recommand_for_news(self):
		# user_history: data/rec/
		# 형태소 분석한 데이터: data/rec/parsing_data.pickle (key-value => news_id : [....]
		# 뉴스 데이터 : data/rec/news.csv
		# 모델(default) : model/model.doc2vec

		news_df = load_pdf('news.csv', sep="^")
		news_df.drop_duplicates(['news_id'], keep='last', inplace=True)

		# 제거할 거 (news_nm기준으로 제거, 13,405 -> 12,553로 의미 없는 기사 제거 )
		# - 클로징, BGM, 썰전 라이브 다시보기(-> 이건 text 내용이 없음)
		# 사용자 history 기반 news 수 :
		#          news_id  ...                                       news_content
		# 65139  NB12045093  ...  <div id='div_NV10478125' ...
		# 65140  NB12045094  ...  <div id='div_NV10478127' ...
		news_df['is_in_str'] = news_df['news_nm'].apply(is_contain_str)
		news_df = news_df[~news_df['is_in_str'] == True].copy()
		news_df.reset_index(drop=True, inplace=True)

		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))

		# test 데이터를 가지고, 얼마나 모델에서 학습된 데이터와 유사도 있는 데이터를 찾는지 확인하기
		data = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "test_tagged_doc.pickle"]))

		self._check_sim_news(data[:10], news_df, d2v_model)

	def _vector_for_learning(self, model, tagged_docs):

		# li = [[1,[1,1]], [2, [2,2]]]
		# zip_a, zip_b = zip(*(li))
		# zip_a = (1, 2)
		# zip_b = ([1,1], [2,2])
		targets, feature_vectors = zip(*[(tag_doc.tags[0], model.infer_vector(tag_doc.words)) for tag_doc in tagged_docs])

	def test_to_verify(self):
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))

		# train_tagged_doc.pickle
		# - [TaggedDocument(words=[...], tags=['NB12043728']), ....]
		train_data = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_tagged_doc.pickle"]))
		test_data = load_pickle(os.path.join(rootpath.detect(), *["data", "rec",
																			   "test_tagged_doc.pickle"]))
		from sklearn.linear_model import LogisticRegression
		# lr =

		y_train, x_train = self._vector_for_learning(d2v_model, train_data)
		y_test, x_test = self._vector_for_learning(d2v_model, test_data)
		pass

	def test_cosine_similarity(self):
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))
		train_data = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_tagged_doc.pickle"]))

		print(train_data[0].tags[0], self.cos_similarity(d2v_model.infer_vector(train_data[0].words),
								  d2v_model.infer_vector(train_data[0].words)))
		print(d2v_model.dv.most_similar(d2v_model.infer_vector(train_data[0].words), topn=20))


	def cos_similarity(self, v1, v2):
		dot_product = np.dot(v1, v2)
		l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
		similarity = dot_product / l2_norm
		return similarity

	# def make_user_embedding










if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestEDA)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())