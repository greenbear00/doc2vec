import unittest
from datetime import datetime, timedelta
from pathlib import Path
from doc2vec.elastic.analyzer.nori_analyzer import NoriAnalyzer
from doc2vec.model.doc2vec import build_model
from doc2vec.preprocessing.preprocessing import generate_tagged_document, generate_tagged_document_by_pandas, \
	load_pdf, is_contain_str, normalize_d2v_docuemnt, write_pdf
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
import time


class TestEDA(unittest.TestCase):

	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger
		set_pandas_format()
		set_matplotlib_sns_font()

	def _calculate_d2v_taggdocument(self, csv_file_name, step, tagdoc_file_name):
		self.p = Parser()
		self.nori = NoriAnalyzer(**self.p.elastic_conf)

		ori_df = load_pdf(csv_file_name, sep="^")
		print(ori_df.shape)
		ori_df.drop_duplicates(['news_id'], keep='last', inplace=True)
		print(ori_df.shape)

		# 제거할 거 (news_nm기준으로 제거, 13634 -> 13432로 의미 없는 기사 제거(BGM, 썰전 라이브 다시보기와 같은 뉴스 제목은 text 내용이 없음)
		# - 클로징, BGM, 썰전 라이브 다시보기(-> 이건 text 내용이 없음)
		# 사용자 history 기반 news 수 :
		#          news_id  ...                                       news_content
		# 65139  NB12045093  ...  <div id='div_NV10478125' ...
		# 65140  NB12045094  ...  <div id='div_NV10478127' ...
		ori_df['is_in_str'] = ori_df['news_nm'].apply(is_contain_str)
		df = ori_df[~ori_df['is_in_str'] == True].copy()
		df.reset_index(drop=True, inplace=True)
		print(df.shape)

		analyzed_df = pd.DataFrame(columns=['news_id', 'tagged_doc'])

		news_contents_parsing = {}
		self.nori.update_index_setting()
		for idx in range(0, df.shape[0], step):
			if idx + step > df.shape[0]:
				self.logger.info(f"{idx} ~ {df.shape[0]} / {df.shape[0]}")
				parsing_df = df.iloc[idx: df.shape[0]][['news_id', 'article_contents', 'news_nm']]
			else:
				self.logger.info(f"{idx} ~ {idx + step} / {df.shape[0]}")
				parsing_df = df.iloc[idx: idx + step][['news_id', 'article_contents', 'news_nm']]

			tmp_parsing = self.nori.get_parsing_news_contents_from_csv(parsing_df)
			news_contents_parsing.update(tmp_parsing)
			result_df = pd.DataFrame({'news_id': list(tmp_parsing.keys()), 'tagged_doc': list(tmp_parsing.values())})
			analyzed_df = pd.concat([analyzed_df, result_df], axis=0)

		if True:
			# save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "parsing_data.pickle"]), news_contents_parsing)
			save_pickle(os.path.join(rootpath.detect(), *["data", "rec", tagdoc_file_name]),
						analyzed_df)

	def test_step1_calculate_d2v_tagdocument(self):
		""" step 1) 뉴스 데이터(2021.11-2022.01)를 읽어서 elastic기반 형태소 분석한 데이터를 pickle로 떨구기"""
		# 이때, news_title+article_contents 를 포함하여 tagged_document로 만듬
		# news_nm기준으로 제거, 13634 -> 13432로 의미 없는 기사 제거(BGM, 썰전 라이브 다시보기와 같은 뉴스 제목은 text 내용이 없음)
		save_flag = True

		step = 10
		self._calculate_d2v_taggdocument(csv_file_name='news_train.csv', step=step,
										 tagdoc_file_name='parsing_train_data_df.pickle')
		# test는 article summary 데이터임
		self._calculate_d2v_taggdocument(csv_file_name='news_test.csv', step=step,
										 tagdoc_file_name='parsing_test_data_df.pickle')

	def test_word_in_news_csv(self):
		ori_df = load_pdf("news_train.csv", sep="^")
		print(ori_df.shape)
		ori_df.drop_duplicates(['news_id'], keep='last', inplace=True)
		print(ori_df.shape)

		# 제거할 거 (news_nm기준으로 제거, 13634 -> 13432로 의미 없는 기사 제거(BGM, 썰전 라이브 다시보기와 같은 뉴스 제목은 text 내용이 없음)
		# - 클로징, BGM, 썰전 라이브 다시보기(-> 이건 text 내용이 없음)
		# 사용자 history 기반 news 수 :
		#          news_id  ...                                       news_content
		# 65139  NB12045093  ...  <div id='div_NV10478125' ...
		# 65140  NB12045094  ...  <div id='div_NV10478127' ...
		ori_df['is_in_str'] = ori_df['news_nm'].apply(lambda x: is_contain_str(news_nm=x, stop_words=['구자철']))
		df = ori_df[ori_df['is_in_str'] == True].copy()
		df.reset_index(drop=True, inplace=True)
		print(df[['news_id', 'news_nm']])

	def test_check_word_dist(self):
		from nltk.probability import FreqDist
		parsing_df = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "parsing_train_data_df.pickle"]))

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
		plt.axis("off")
		plt.show()

	def _generate_d2v_tagged_document(self, parsing_data_df_path, tagged_doc_file_name):
		# from sklearn.model_selection import train_test_split
		parsing_train_df = load_pickle(parsing_data_df_path)

		# x_train, x_test, y_train, y_test = train_test_split(parsing_data_df[['tagged_doc']],
		# 													parsing_data_df['news_id'],
		# 													test_size=0.3,
		# 													shuffle=True,
		# 													random_state=42)
		# x_train = x_train.reset_index(drop=True)
		# x_test = x_test.reset_index(drop=True)
		# y_train = y_train.reset_index(drop=True)
		# y_test = y_test.reset_index(drop=True)
		# print("x_train.shape= ", x_train.shape, ", y_train.shape= ", y_train.shape)
		# print("x_test.shape= ", x_test.shape, ", y_test.shape= ", y_test.shape)
		#
		# train_tagged_doc = generate_tagged_document_by_pandas(x_train, y_train)
		# test_tagged_doc = generate_tagged_document_by_pandas(x_test, y_test)
		#
		# # train_tagged_doc.pickle
		# # - [TaggedDocument(words=[...], tags=['NB12043728']), ....]
		# save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_tagged_doc.pickle"]), train_tagged_doc)
		# save_pickle(os.path.join(rootpath.detect(), *["data", "rec", "test_tagged_doc.pickle"]), test_tagged_doc)

		train_tagged_doc = generate_tagged_document_by_pandas(parsing_train_df)
		save_pickle(os.path.join(rootpath.detect(), *["data", "rec", tagged_doc_file_name]),
					train_tagged_doc)

	def test_step2_generate_d2v_tagducument(self):
		""" step 2) elastic기반 corpus를 tagged_doc으로 변환"""

		parsing_train_data_df_path = os.path.join(rootpath.detect(), *["data", "rec", "parsing_train_data_df.pickle"])
		self._generate_d2v_tagged_document(parsing_train_data_df_path, "train_taggeddoc.pickle")

		parsing_test_data_df_path = os.path.join(rootpath.detect(), *["data", "rec", "parsing_test_data_df.pickle"])
		self._generate_d2v_tagged_document(parsing_test_data_df_path, "test_taggeddoc.pickle")

	def test_check_tagged_doc(self):
		""" step2-1) tagged_doc 확인 """
		train_tagged_doc = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_taggeddoc.pickle"]))
		print("train tagged_doc size: ", len(train_tagged_doc))
		train_tagged_news_ids = [t.tags[0] for t in train_tagged_doc]

		test_tagged_doc = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "test_taggeddoc.pickle"]))
		print("test tagged_doc size: ", len(test_tagged_doc))
		test_tagged_news_ids = [t.tags[0] for t in test_tagged_doc]
		self.assertEqual(set(train_tagged_news_ids), set(test_tagged_news_ids))

		print('NB10142926' in train_tagged_news_ids)
		print('NB10142926' in test_tagged_news_ids)

	def test_step3_save_model(self):
		""" step 3) doc2vec 모델 생성 (현재 13,432 doc에 대해서 00:07:10.015364이 소요됨 )"""
		# 참고: 모델 생성 및 평가 (참고로, test데이터는 train데이터에서 뉴스 본문의 summary임)
		# 참고(target이 감성(1,0)에 대한 평가): https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html
		# 참고(target이 장르에 대한 평가): https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
		start_time = time.time()
		tagged_doc = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_taggeddoc.pickle"]))

		d2v_model = build_model()
		d2v_model.build_vocab(tagged_doc)
		d2v_model.train(tagged_doc, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

		d2v_model.save(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))

		print("total_time: ", timedelta(seconds=(time.time() - start_time)))

	def test_model_test(self):
		""" step 3-1) model test"""
		test_tagged_doc = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "test_taggeddoc.pickle"]))

		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))

		print(test_tagged_doc[0])
		tag_id = test_tagged_doc[0].tags[0]
		vector = d2v_model.infer_vector(test_tagged_doc[0].words)
		print("vector size = ", len(vector))
		print("Top 10 values in Doc2Vec inferred vecotr: ")
		print(vector[:10])

		docvec2 = d2v_model.dv[0]
		docvecsyn2 = d2v_model.docvecs.doctag_syn0[0]

		sims = d2v_model.dv.most_similar(vector, topn=20)

		for sim in sims:
			news_id = sim[0]
			sim_value = sim[1]

			print('\t- ', sim_value, " : ", news_id, " <- find " if tag_id == news_id else '')

	def test_d2vmodel_info(self):
		"""학습시킨 해당 모델의 tags들이 뭐가 있는지 확인하는 방법"""
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))
		print('d2v_model의 문서 수: ', d2v_model.corpus_count)
		print('d2v_model의 단어 수: ', d2v_model.corpus_total_words)
		print('d2v_model에 NB10142926 tag가 포함되어 있나? ', d2v_model.dv.has_index_for('NB10142926'))
		print('CCTV 단어가 모델에 훈련된 빈도 수 : ', d2v_model.wv.get_vecattr('CCTV', 'count'))
		all_docvecs = d2v_model.dv.vectors
		print('d2v_model의 tags 수: ', all_docvecs.shape[0])
		print('d2v_model의 vector_size: ', all_docvecs.shape[1])

	def _check_sim_news(self, data, news_df, d2v_model, train_data=None):

		for a_data in data:
			selected_news_id = a_data.tags[0]
			selected_news_df = news_df[news_df['news_id'] == selected_news_id][['news_id', 'news_nm',
																				'article_contents']]
			# Compare and print the most/median/least similar documents from the train corpus
			print(
				"\n=============================================================================================================")
			print(f"(Test) [{selected_news_id}] {selected_news_df.news_nm.values[0]}")
			# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)

			inferred = d2v_model.infer_vector(a_data.words)
			sims = d2v_model.dv.most_similar(inferred, topn=10)

			# y_train, features
			# targets, feature_vectors = zip(*[(selected_news_id, d2v_model.infer_vector(a_data.words))])

			for sim in sims:
				news_id = sim[0]
				sim_value = sim[1]
				tmp_sim_news_df = news_df[news_df['news_id'] == news_id][
					['news_id', 'news_nm', 'article_contents']]
				if not tmp_sim_news_df.empty:
					# filtered_train_corpus = list(filter(lambda x: x.tags[0] == news_id, train_data))
					# a_train_corpus = filtered_train_corpus[0].words if filtered_train_corpus else []
					print(f"연관추천 [{news_id} ({sim_value})] {tmp_sim_news_df.news_nm.values[0]} "
						  f"{(' ' * 20 + '<-find ') if selected_news_id == news_id else ''} ")
				# print(f"«{' '.join(a_train_corpus)}»")
			print("\n\n")

	def test_step4_recommand_for_news(self):
		# user_history: data/rec/
		# 형태소 분석한 데이터: data/rec/parsing_data.pickle (key-value => news_id : [....]
		# 뉴스 데이터 : data/rec/news.csv
		# 모델(default) : model/model.doc2vec

		news_df = load_pdf('news_train.csv', sep="^")
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
		test_data = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "test_taggeddoc.pickle"]))
		train_data = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_taggeddoc.pickle"]))

		self._check_sim_news(train_data[:10], news_df, d2v_model, train_data)

	def _vector_for_learning(self, model, tagged_docs):

		# li = [[1,[1,1]], [2, [2,2]]]
		# zip_a, zip_b = zip(*(li))
		# zip_a = (1, 2)
		# zip_b = ([1,1], [2,2])
		targets, feature_vectors = zip(
			*[(tag_doc.tags[0], model.infer_vector(tag_doc.words)) for tag_doc in tagged_docs])


	def test_cosine_similarity(self):
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))
		train_data = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "train_taggeddoc.pickle"]))

		print(train_data[0].tags[0], self.cos_similarity(d2v_model.infer_vector(train_data[0].words),
														 d2v_model.infer_vector(train_data[0].words)))
		print(d2v_model.dv.most_similar(d2v_model.infer_vector(train_data[0].words), topn=20))

	def cos_similarity(self, v1, v2):
		dot_product = np.dot(v1, v2)
		l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
		similarity = dot_product / l2_norm
		return similarity

	def test_step5_d2vmodel_infer_vector(self):
		""" step5) 학습 시킨 데이터를 전체적으로 infer시켜서 norm2로 정규화한 다음에 file로 write ( 13432 doc infered time 16:34 sec ) """
		# 우선 13432 doc에 대해서 infer시킨 후, norm2로 정규화해서 파일로 저장하는 시간이 총 0:17:21.905376
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))
		train_tagged_doc = load_pickle(os.path.join(rootpath.detect(), *['data', 'rec', 'train_taggeddoc.pickle']))

		normalize_d2v_docuemnt(d2v_model, train_tagged_doc)


	def test_load_user_history(self):

		############################################
		# user_history_df
		############################################
		user_df = load_pdf("user_history.csv")
		user_df.groupby('parm_member_id')
		user_df['reqtime'] = pd.to_datetime(user_df['reqtime'])
		user_df['news_id'] = user_df['news_id'].apply(lambda x: x.upper())
		user_df.sort_values(by='reqtime', inplace=True)

		############################################
		# news_df
		############################################
		news_df = load_pdf("news_train.csv", sep="^")

		############################################
		# user_news_df (user_history_df + news_df)
		# - parm_member_id, action_name, news_id 가 없는거 제거
		############################################
		user_news_df = pd.merge(user_df[['reqtime', 'action_name', 'parm_member_id', 'news_id']],
								news_df[['news_id', 'section', 'news_nm', 'service_date', 'reg_dt']], on='news_id',
								how='left')
		user_news_df.drop_duplicates(['parm_member_id', 'action_name', 'news_id'], keep='last', inplace=True)
		user_news_df.sort_values(by='reqtime', inplace=True)


		del(user_df)
		del(news_df)

		# " 검증용 사용자 "
		# user_visit_count_df = pd.DataFrame(user_news_df['parm_member_id'].value_counts())
		# min_visit_users = user_visit_count_df[
		# 	(user_visit_count_df.parm_member_id > 2) & (user_visit_count_df.parm_member_id<4) ].index.tolist()

		############################################
		# user_d2v_weight_df2 계산 (by user_history_ord_df, news_d2_weight_df)
		############################################
		# user_history_ord_df = user_news_df[user_news_df.parm_member_id.isin(['disch62', 'jangdaneng', 'gspmilk'])][
		# 	['parm_member_id', 'action_name', 'news_id', 'section', 'news_nm', 'reqtime']]
		# user_history_ord_df = user_news_df[user_news_df.parm_member_id.isin(['bagchihu1', 'conahn12', '1760203540843478'])].copy()
		user_history_ord_df = user_news_df.copy()
		user_history_ord_df['ord'] = user_history_ord_df.groupby('parm_member_id').cumcount() + 1

		news_d2v_weight_df = load_pdf('news_d2v_w.csv')

		print("calcuate user_d2v_w")
		user_d2v_weight_df = pd.merge(user_history_ord_df, news_d2v_weight_df, on='news_id', how='left')
		del(user_history_ord_df)

		# step 1. parm_member_id, topic 기준으로 weight 값 모두 sum
		user_d2v_weight_df2 = user_d2v_weight_df.copy()
		user_d2v_weight_df2 = user_d2v_weight_df.groupby(['parm_member_id', 'topic']).agg({'w': 'sum'})
		user_d2v_weight_df2.reset_index(inplace=True)
		del(user_d2v_weight_df)

		# step 2. norm2로 표준화
		user_d2v_weight_df2['w_pow'] = user_d2v_weight_df2['w'].pow(2)
		grouped_user_d2v_weight = user_d2v_weight_df2.groupby('parm_member_id').agg({'w_pow': 'sum'}).rename(columns={
			'w_pow': 'w_sum_pow'})
		grouped_user_d2v_weight.reset_index(inplace=True)
		grouped_user_d2v_weight['w_sum_pow'] = grouped_user_d2v_weight['w_sum_pow'].apply(np.sqrt)
		user_d2v_w = pd.merge(user_d2v_weight_df2, grouped_user_d2v_weight, on='parm_member_id', how='left')
		user_d2v_w['final_w'] = user_d2v_w['w'] / user_d2v_w['w_sum_pow']
		del(user_d2v_weight_df2)
		del (grouped_user_d2v_weight)

		print("write user_d2v_w")
		user_d2v_w = user_d2v_w.astype({'topic':int})
		write_pdf(user_d2v_w[['parm_member_id', 'topic', 'final_w']], 'user_d2v_w.csv')

		# rec
		# - news_d2v_weight_df 와
		# - 사용자 history를 기반하여 news_d2v_weight_df와 join시켜 weight값 구한 user_d2v_weight_df3로
		# - 유사도를 구함
		# 나온 결과에 대해서 사용자별로 관심있는 category 비중과 time weight(최신기사)로 추천 해줘야 함.
		# 또한 평가는?

		#
		# 	select b.news_id, sum(a.w*b.w) sim
		#     from mind_train_user_d2v_w a
		#         join mind_train_news_lda_w b on b.topic_id = a.topic_id
		#     where a.user_id = '{user_id}'
		#     group by b.news_id
		#     )


		# rec_df = pd.merge(user_d2v_w[['parm_member_id', 'topic', 'final_w']], news_d2v_weight_df, on='topic', how='left')









if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestEDA)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())
