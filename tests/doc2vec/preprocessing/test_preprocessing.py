import unittest
from datetime import datetime, timedelta
from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
import rootpath
import pandas as pd
import os
from doc2vec.preprocessing.preprocessing import *
from doc2vec.util.common import load_pickle


class TestPreProcessing(unittest.TestCase):
	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger

		self.p = Parser()

	def _load_pickle(self, path):
		return load_pickle(path)


	def test_do_generate_tagged_documment(self):

		data = self._load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]))

		d2v_result = generate_tagged_document(data)
		print(d2v_result)

	def test_load_user_history(self):

		user_df = user_history_df("user_history.csv")
		print(user_df.tail())

	def test_load_pickle(self):

		data = self._load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]))

		for k, v in data.items():
			print(k, v[:10])


if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestPreProcessing)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())