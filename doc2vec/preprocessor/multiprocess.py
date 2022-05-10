import time
import multiprocessing

import pandas as pd
import psutil
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager
import numpy as np
import gensim
from doc2vec.preprocessor.preprocessing import clean_content
# from doc2vec.util.Logger import Logger

# log = Logger().logger


@contextmanager
def poolcontext(*args, **kwargs):
	pool = multiprocessing.Pool(*args, **kwargs)
	yield pool
	pool.terminate()


def _check_usage_of_cpu_and_memory():
	memory_usage = psutil.virtual_memory().percent
	available_memory = psutil.virtual_memory().available * 100 / \
					   psutil.virtual_memory().total
	cpu_usage = psutil.cpu_percent()

	print(f"process name: {multiprocessing.current_process()}")
	print(f"\t- cpu usage: {cpu_usage}")
	print(f"\t- memory usage: {memory_usage}")
	print(f"\t- available_memory: {available_memory}")


def f(x):
	# print("name: ", name1)
	for _ in range(100):
		x += 1
	_check_usage_of_cpu_and_memory()
	return x


def doc2vec_inference(model_path, tagged_doc):
	d2v_model = gensim.models.doc2vec.Doc2Vec.load(model_path)
	news_topic_arr = []
	for index in tqdm(range(len(tagged_doc)), desc="progress...", mininterval=1):
		time.sleep(0.1)
		_check_usage_of_cpu_and_memory()
		a_tagged_doc = tagged_doc[index]
		news_id = a_tagged_doc.tags[0]
		inferred = d2v_model.infer_vector(a_tagged_doc.words)
		sims = d2v_model.dv.most_similar(inferred, topn=len(d2v_model.dv))
		for a_sim in sims:
			# news_id, 연관 news_id, 연관 news_id의 유사도
			news_topic_arr.append({"news_id": news_id, "rec_news_id": a_sim[0], "score": a_sim[1]})
	return news_topic_arr


def printname(arg1, arg2):
	print(f"{arg1},{arg2}")


def split_list(li, size):
	return np.split(li, np.arange(size, len(li), size))


def multi_clean_contents(news_ids, contents):
	# print(len(news_ids), len(contents))
	li = []
	for index in tqdm(range(len(news_ids)), desc=f"{multiprocessing.current_process()} progress...", mininterval=1):
		time.sleep(0.1)
		# _check_usage_of_cpu_and_memory()
		print("process name: ", multiprocessing.current_process())
		id = news_ids[index]
		content = contents[index]
		li.append({'news_id': id, 'content': clean_content(content)})
	return li


def multi_clean_contents2(news_id, contents):
	# news_id, contents = data.items()
	new_contents = clean_content(contents)
	print(f"process name: {multiprocessing.current_process()}" )
	return {'news_id': news_id, 'content': new_contents}


# jupyter 내에서 multiprocessing을 하려면, 함수를 밖으로 내보내야 함.
# with multiprocessing.Pool(processes=cpu_count) as p:
#     print(p.map(f, range(200), "sss"))
# dir(pool)
if __name__ == "__main__":
	import os
	import rootpath
	from doc2vec.util.common import load_pickle, save_pickle
	from doc2vec.preprocessor.preprocessing import *


	cpu_count = multiprocessing.cpu_count()

	start_time = time.time()
	# print("start_time: ", datetime.now())
	# args1 = [i for i in range(1000)]
	# args2 = [f": {i}...." for i in range(1000)]
	# with poolcontext(processes=cpu_count) as pool:
	# 	clean_content_li = pool.starmap(printname, zip(args1, args2))
	# print(f"total_time: ", timedelta(seconds=(time.time() - start_time)))

	news_2022_01_df = load_pdf("news_df_202106_202201.csv", sep="^")

	news_ids = news_2022_01_df.news_id.values.tolist()
	news_contents = news_2022_01_df.article_contents.values.tolist()
	# split_news_ids = split_list(news_ids, int(len(news_ids) / cpu_count))
	# split_news_contents = split_list(news_contents, int(len(news_contents) / cpu_count))
	params = []
	for i in range(10):
		params.append((news_ids[i], news_contents[i]))

	with poolcontext(processes=cpu_count) as pool:
		result = pool.starmap(multi_clean_contents2, params)

	# with poolcontext(processes=cpu_count) as pool:
	# 	clean_content_li = pool.starmap(multi_clean_contents, zip(split_news_ids, split_news_contents))

	# new_clean_contents = []
	# for a_li in clean_content_li:
	# 	new_clean_contents.extend([content for content in a_li])
	# new_clean_content_df = pd.DataFrame(new_clean_contents)

	tagged_doc = load_pickle(os.path.join(rootpath.detect(), *["data", "rec", "tagged_doc_w_mecab.pickle"]))
	split_tagged_doc = split_list(tagged_doc[:100], int(len(tagged_doc[:100]) / cpu_count))

	new_tagged_docs = []
	for tag in split_tagged_doc:
		new_tagged_docs.append([gensim.models.doc2vec.TaggedDocument(t[0], t[1]) for t in tag])

	print(f"cpu count : {cpu_count}")
	model_path = os.path.join(rootpath.detect(), *['model', 'model_w_mecab.doc2vec'])
	arg1 = [model_path for _ in range(len(new_tagged_docs))]
	print(dict(zip(arg1, new_tagged_docs)))

	with poolcontext(processes=cpu_count) as pool:
		infer_li = pool.starmap(doc2vec_inference, zip(arg1, new_tagged_docs))
	new_infer_li = []
	for a_li in infer_li:
		new_infer_li.extend([infer_data for infer_data in a_li])
	df = pd.DataFrame(new_infer_li)
	len(df.news_id.unique().tolist())
