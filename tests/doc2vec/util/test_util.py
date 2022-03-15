import unittest
from datetime import datetime, timedelta

from doc2vec.preprocessing.preprocessing import load_pdf
from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
import rootpath
import pandas as pd
import os


class TestUtil(unittest.TestCase):
	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger

		self.p = Parser()

	def test_load_db_conf(self):
		print(self.p.db_conf)

	def test_load_csv(self):
		df = load_pdf("news.csv", sep="^")
		print(df.head())

	def test_load_piclke(self):
		from doc2vec.util.common import load_pickle
		import rootpath
		data = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]))

		print(data.keys())

		self.assertEqual(type(data), dict, "type doesn't match")

	def test_path_test(self):

		path = os.path.join(rootpath.detect(), *["data2", 'tmp.csv'])
		os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestUtil)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())