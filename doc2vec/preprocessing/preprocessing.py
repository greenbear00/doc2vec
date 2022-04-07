import gensim
import numpy as np
import pandas as pd
from tabulate import tabulate
import rootpath
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import re
from konlpy.tag import Mecab

URL_PATTERN = re.compile(
	r'^(?:http|ftp)s?://'  # http:// or https://
	r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
	r'localhost|'  # localhost...
	r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
	r'(?::\d+)?'  # optional port
	r'(?:/?|[/?]\S+)$', re.IGNORECASE)


# html 태그 제거
HTML_PATTERN = re.compile('<.*?>')

# &quot;와 같은 태그 제거
HTML_CHAR_PATTERN = re.compile("&.*?;")
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)
WIKI_REMOVE_CHARS = re.compile("'+|(=+.{2,30}=+)|__TOC__|(ファイル:).+|:(en|de|it|fr|es|kr|zh|no|fi):|\n", re.UNICODE)
WIKI_SPACE_CHARS = re.compile("(\\s|゙|゚|　)+", re.UNICODE)
EMAIL_PATTERN = re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
WIKI_REMOVE_TOKEN_CHARS = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)
# [출처=이데일리] 또는 [출처=http...] 제거
REF_PATTERN = re.compile("\[출처(.*?)\]", re.UNICODE)

tokenizer = Mecab()

def tokenize(content):
	# install : https://konlpy-ko.readthedocs.io/ko/v0.4.3/install/
	# 품사 정보 확인
	# tokenizer.pos(content)
	# Token으로 쪼개기
	return tokenizer.morphs(content)

def clean_content(content):
	content = re.sub(EMAIL_PATTERN, ' ', content)
	content = re.sub(URL_PATTERN, ' ', content)
	content = re.sub(HTML_PATTERN, ' ', content)
	content = re.sub(HTML_CHAR_PATTERN, ' ', content)
	content = re.sub(WIKI_REMOVE_CHARS, ' ', content)
	content = re.sub(WIKI_SPACE_CHARS, ' ', content)
	content = re.sub(MULTIPLE_SPACES, ' ', content)

	content = re.sub(REF_PATTERN, ' ', content)


	return content


def generate_tagged_document_by_pandas(x_train_df):
	# def generate_tagged_document_by_pandas(x_train_df, y_train_series):

	docs = []
	for idx, row in x_train_df.iterrows():
		docs.append(gensim.models.doc2vec.TaggedDocument(row['tagged_doc'], [row['news_id']]))

	# for idx, row in x_train_df.iterrows():
	# 	# y_train_series는 series 데이터 임
	# 	# print(idx, row['tagged_doc'], " -> ", y_train_series.iloc[idx])
	# 	docs.append(gensim.models.doc2vec.TaggedDocument(row['tagged_doc'], [y_train_series.iloc[idx]]))

	return docs


def reform_news_df(df: pd.DataFrame):
	# service_date = datetime.strptime("202111041817", '%Y%m%d%H%M')
	# print("service_date", service_date.strftime('%Y-%m-%dT%H:%M:%S+09:00'))
	#
	# df = load_pdf("news_1.csv", sep="^")
	# print(df.columns)
	new_df = df.copy()
	new_df = new_df.astype({'service_dt': str})
	new_df['service_dt'] = new_df['service_dt'].apply(lambda x: datetime.strptime(x, '%Y%m%d%H%M').strftime(
		'%Y-%m-%dT%H:%M:%S+09:00'))
	new_df = new_df.rename(
		columns={'title': 'news_nm', 'article_contents_bulk': 'news_content', 'section_name': 'section',
				 'service_dt': 'service_date'})

	if 'unnamed: 0' in new_df.columns.tolist():
		new_df.drop(columns=['unnamed: 0'], inplace=True)

	print(new_df.tail())
	return new_df


def load_pdf(file_name: str = "view_data.csv", sep: str = ",", file_path: str = None) -> pd:
	# root_path = Path(__file__).parent.parent.parent
	if not file_path:
		file_path = os.path.join(rootpath.detect(), *["data", "rec", file_name])
	df = pd.read_csv(file_path, sep=sep)
	df.columns = df.columns.str.lower()

	df = df.replace({np.nan: None})

	if 'service_date' in df.columns:
		df['service_date'] = pd.to_datetime(df['service_date'])

	print(f"file_path : {file_path}")
	print(f"file_name : {file_name}")
	print("DATA INFO")
	print(tabulate(df.info(), tablefmt="psql", headers="keys"))
	print("-" * 100)
	print("DATA IS_NULL")

	print(tabulate(pd.DataFrame(df.isnull().sum()), tablefmt="psql", headers="keys"))

	return df


def is_contain_str(news_nm, stop_words=['BGM', '클로징', '썰전 라이브 다시보기']):
	pop_list = list(filter(lambda x: x in news_nm, stop_words))
	if pop_list:
		return True
	else:
		return False


def generate_tagged_document(data):
	"""

	:param data: {news_id : words, ...}
	:return:
		list(<class 'gensim.models.doc2vec.TaggedDocument'>)
	"""
	docs = []
	for news_id, words in data.items():
		docs.append(gensim.models.doc2vec.TaggedDocument(words, [news_id]))

	return docs


def user_history_df(file_path: str):
	load_df = load_pdf(file_path)
	return load_df


def normalize_d2v_docuemnt(model: gensim.models.doc2vec.Doc2Vec, d2v_document: list):
	import time
	from tqdm import tqdm

	start_time = time.time()
	news_topic_arr = []

	d2v_size = len(d2v_document)
	for index in tqdm(range(d2v_size), desc='document process...', mininterval=10):
		td = d2v_document[index]
		# td : TaggedDocument(['홍남기', '부총리', '기획', '재정부', '장관', ..., '통보'], ['2022030610442141299'])
		# td.tags : ['2022030610442141299']
		# td.words : ['홍남기', '부총리', '기획', '재정부', '장관', ..., '통보']
		news_id = td.tags[0]  # 실제 news_id

		# 위 모델에서 vector_size = 20으로 해두었기 때문에, 문장에 대해서 infer_vector를 출력하면 총 20개의 vector가 나옴
		# topics: [-0.3248782  -0.76411337 -0.30905467  1.3714583  -0.95404065 -0.13403396
		#  -1.4188823  -0.05682391  0.86159384 -0.772219    0.22174104  0.13577682
		#   0.11970853 -1.5014095   1.0538101   1.0860668   0.58602583 -0.19206288
		#  -1.2536151  -0.34537676]
		topics = model.infer_vector(td.words)  # 문장의 단어들을 집어 넣음

		# 정규화 시킴 (vector_size 만큼 각각 norm2로 정규화 시킴)
		# norm = 1.7675153
		norm = np.sqrt(np.sum([t * t for t in topics]))

		for idx, t in enumerate(topics):
			news_topic_arr.append([news_id, idx, t / norm])
	# print(news_topic_arr1)

	# topic은 vector로 현재 모델에서 1000으로 사용되고 있음
	topic_df = pd.DataFrame(
		columns=['news_id', 'topic', 'w'],
		data=news_topic_arr
	)

	# topic_df.to_csv('temp/news_d2v_w.csv', index=False)
	write_pdf(topic_df, "news_d2v_w.csv")

	print("total_time: ", timedelta(seconds=(time.time() - start_time)))

	return topic_df


def write_pdf(load_pdf: pd, file_name: str, sep: str = ',') -> bool:
	flag = True
	path = Path(__file__).parent.parent.parent

	try:

		base_path = os.path.join(path, *["data", "rec"])
		os.makedirs(base_path, exist_ok=True)

		if os.path.isfile(os.path.join(path, *["data", "rec", file_name])):
			os.remove(os.path.join(path, *["data", "rec", file_name]))

		load_pdf.to_csv(os.path.join(path, *["data", "rec", file_name]), sep=sep, na_rep='NaN', header=True,
						index=False)
		print("write csv : ", os.path.join(path, *["data", "rec", file_name]))
	except Exception as es:
		print(es)
		flag = False

	return flag
