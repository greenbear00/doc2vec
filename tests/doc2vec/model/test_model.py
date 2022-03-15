import unittest
from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
import rootpath
import pandas as pd
import os
from doc2vec.preprocessing.preprocessing import *
from doc2vec.model.doc2vec import *
from doc2vec.util.common import *
from doc2vec.util.db_connector import *


class TestModel(unittest.TestCase):
	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger

		self.p = Parser()



	def test_generate_d2v_topic_df(self):
		data = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data_all.pickle"]))
		self.logger.info(data.keys())

		d2v_document = generate_tagged_document(data)
		d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))

		d2v_topics_df = normalize_d2v_docuemnt(d2v_model, d2v_document)
		self.logger.info("d2v_topics_df")
		self.logger.info(d2v_topics_df.tail())
		write_pdf(d2v_topics_df, "d2v_topic_all_df.csv")

	def test_concat_data(self):
		news_11_df = load_pdf('user_11_history.csv', sep="^")
		news_12_df = load_pdf('user_12_history.csv', sep="^")
		news_1_df = load_pdf('user_1_history.csv', sep="^")
		news_df = pd.concat([news_11_df, news_12_df, news_1_df], ignore_index=True)

		print(news_df.tail())
		write_pdf(news_df, 'user_all.csv', sep="^")

	def test_save_model(self):
		# 3달치 말뭉치 corpus ({news_id: [....]})
		data1 = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data_11.pickle"]))
		data2 = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data_12.pickle"]))
		data3 = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data_1.pickle"]))

		data1.update(data2)
		data1.update(data3)

		save_pickle(os.path.join(rootpath.detect(), *["data","parsing_data_all.pickle"]), data1)

		# self.logger.info(len(data1.keys()))
		#
		# d2v_document = generate_tagged_document(data1)
		# d2v_model = build_model()
		# d2v_model.build_vocab(d2v_document)
		# d2v_model.train(d2v_document, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
		#
		# d2v_model.save(os.path.join(rootpath.detect(), *['model', 'model.doc2vec']))












if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())