import pandas as pd

from doc2vec.schema.SchemaGenerator import SchemaGenerator
from doc2vec.elastic.ElasticGenerator import ElasticGenerator
from datetime import datetime
import re

from doc2vec.util.common import clean_html_tag_in_text


class NoriAnalyzer(ElasticGenerator):
	def __init__(self, hosts:str, username:str, password:str):
		super(NoriAnalyzer, self).__init__(hosts, username, password)
		# self.nori_analyzer_update_flag = nori_analyzer_update_flag
		self.sg = SchemaGenerator(obj=self)

	def put_data(self, data:list):
		template_name = "nori-analyzer-template"
		template, index_name = self.sg.get_small_template(template_name)

		# create index template and index
		self.create_index_with_setting(index_name=index_name, template_name=template_name, template=template)

		self.do_elastic_write(index_name=index_name, elk_doc=data)


	def update_index_setting(self):
		template_name = "nori-analyzer-template"
		# prefix index_name = "test-doc2vec"
		template, index_name = self.sg.get_small_template(template_name)
		# setting_template, index_name = self.sg.get_template(template_name)

		# 기존 my_analyzer의 filter에 "lowercase"를 지움 -> LG엔솔 -> lg 엔솔로 분석되어져 나옴
		self.create_index_with_setting(index_name=index_name, template_name=template_name, template=template)
		return index_name

	def get_parsing_news_contents_from_csv(self, df: pd.DataFrame):
		news_contents_parsing = {}
		prefix_index = 'test-doc2vec'
		news_contents = {}
		text_position = [0]
		before_position = 0

		# print(df.news_id.values)
		for idx, row in df.iterrows():
			# print(idx, row['news_id'], row['news_content'][:10])
			news_content = row['news_nm'] + " " + row['news_content']
			news_id = row['news_id']
			# clean_news_content = clean_html_tag_in_text(row['news_content'])
			clean_news_content = clean_html_tag_in_text(news_content)
			clean_news_content = clean_news_content.replace('\n', '').replace("\r", '').replace("|", '')
			clean_news_content = clean_news_content.strip()

			news_content_size = len(clean_news_content)
			# text_position.append(before_position)
			text_position.append(text_position[-1] + news_content_size + 1)
			# before_position += (news_content_size + 1)
			news_contents[row['news_id']] = clean_news_content

			# self.logger.info(news_id)

		body = {
			"analyzer": "my_analyzer",
			"text": list(news_contents.values())
		}

		res2 = self.es_client.indices.analyze(index=prefix_index, body=body)

		tokens = []
		for index in range(1, len(text_position)):  # start 31
			# for a_token in res2.get('tokens'):
			# 	print(f"{text_position[index - 1]} <= {a_token.get('start_offset')} and"
			# 		  f" {text_position[index]} > {a_token.get('end_offset')} =====> {text_position[index - 1] <= a_token.get('start_offset') and text_position[index] > a_token.get('end_offset')}")
			filtered_tokens = list(
				filter(lambda x: text_position[index - 1] <= x.get('start_offset') and text_position[
					index] > x.get('end_offset'), res2.get('tokens')))
			tokens.append([a_token.get('token') for a_token in filtered_tokens])

		index = 0
		for k, v in news_contents.items():
			news_contents_parsing[k] = tokens[index]
			index += 1

		return news_contents_parsing


	def get_parsing_news_contents_from_elastic(self, the_date)-> (bool, dict):
		"""
			news_id, news_content 데이터를 엘라스틱에서 가져와서
			html 태그 및 \n을 제거한 뒤
			nori_analyzer를 적용하여 파싱된 데이터를 news_id와 맵핑하여 반환함
			:param the_date: datetime
			:returns
				flag(bool)
				news_contents_parsing(dict) : {'NB12..': ['홍남기', '경제', ...], ...}
		"""

		flag = True
		start_date = the_date.strftime("%Y-%m-%dT00:00:00+09:00")
		end_date = the_date.strftime("%Y-%m-%dT23:59:59+09:00")
		prefix_index = "test-doc2vec"
		self.logger.info(f"{start_date} ~ {end_date} -> {prefix_index}")

		news_contents_parsing = {}

		news_data = []

		try:
			query = {
				"size": 10,
				"query": {
					"bool": {
						"must": [],
						"filter": [
							{
								"match_all": {}
							},
							{
								"range": {
									"service_date": {
										"gte": start_date, #"2022-01-01T00:00:00+09:00",
										"lte": end_date #"2022-10-01T00:00:00+09:00"
									}
								}
							}
						],
						"should": [],
						"must_not": []
					}
				},
				"fields": [
					"news_id", "news_content",  'news_nm', 'section', 'service_date'
				],
				"sort": [
					{
						"service_date": {
							"order": "asc"
						}
					}
				],
				"_source": False
			}
			# # access search context for 5 seconds before moving on to the next step, .
			scroll = '30s'
			response = self.es_client.search(index=prefix_index, body=query, scroll=scroll)

			scroll_id = response['_scroll_id']
			scroll_size = response['hits']['total']['value']

			while scroll_size > 0:
				news_contents = {}
				text_position = [0]
				before_position = 0
				self.logger.info(f"<<< THE_DATE = {start_date} >>>")
				self.logger.info(f"scroll_size: {scroll_size}")
				# 해당 시간동안 수집한 news_id(list)
				# self.logger.info(f"{response['hits']['hits']}")
				for data in response['hits']['hits']:
					fields = data.get('fields')
					if fields:
						if not fields.get('news_id')[0] in news_contents:
							news_content = fields.get('news_content')[0]
							clean_news_content = clean_html_tag_in_text(news_content)
							clean_news_content = clean_news_content.replace('\n', '')
							clean_news_content = clean_news_content.strip()

							news_content_size = len(clean_news_content)
							# text_position.append(before_position)
							text_position.append(text_position[-1]+news_content_size+1)
							# before_position += (news_content_size + 1)
							news_contents[fields.get('news_id')[0]] = clean_news_content

							news_data.append({
								"news_id": fields.get('news_id')[0],
								"news_content": fields.get('news_content')[0],
								'news_nm': fields.get('news_nm')[0],
								'section': fields.get('section')[0],
								'service_date': fields.get('service_date')[0]
							})
							self.logger.info(f"{fields.get('news_id')[0]}")

						# news_nm_size = len(news_nm)
						# text_position.append(before_position)
						# text.append(news_nm)
						# before_position += (news_nm_size + 1)
				# text_position.append(before_position)

				body = {
					"analyzer": "my_analyzer",
					"text": list(news_contents.values())
				}

				res2 = self.es_client.indices.analyze(index=prefix_index, body=body)

				tokens = []
				for index in range(1, len(text_position)):  # start 31
					# for a_token in res2.get('tokens'):
					# 	print(f"{text_position[index - 1]} <= {a_token.get('start_offset')} and"
					# 		  f" {text_position[index]} > {a_token.get('end_offset')} =====> {text_position[index - 1] <= a_token.get('start_offset') and text_position[index] > a_token.get('end_offset')}")
					filtered_tokens = list(
						filter(lambda x: text_position[index - 1] <= x.get('start_offset') and text_position[
							index] > x.get('end_offset'), res2.get('tokens')))
					tokens.append([a_token.get('token') for a_token in filtered_tokens])

				index = 0
				for k,v in news_contents.items():
					news_contents_parsing[k] = tokens[index]
					index += 1

				response = self.es_client.scroll(scroll_id=scroll_id, scroll=scroll)

				scroll_id = response['_scroll_id']
				scroll_size = len(response['hits']['hits'])
				self.logger.info('\n\n\n')
		except Exception as es:
			self.logger.error(f"es = {es}")
			flag = False

		return flag, news_contents_parsing, news_data


	def get_parsing_news_nm_by_nori_analyzer(self, the_date:datetime, data:list)->list:
		"""
		index_name = test-doc2vec
		위 index에서 news_nm을 던져서 nori_analyzer에 의해서 분석된 데이터를 news_nm_analyzer로 로겨옴
		"""
		new_data = []
		try:
			index_name = self.update_index_setting()

			text = []
			text_position = []
			before_position = 0
			for a_data in data:
				news_nm = a_data.get('news_nm')
				news_nm_size = len(news_nm)
				text_position.append(before_position)
				text.append(news_nm)
				before_position += (news_nm_size + 1)
			text_position.append(before_position)
			body = {
				"analyzer": "my_analyzer",
				"text": text
			}
			if text:

				res2 = self.es_client.indices.analyze(index=index_name, body=body)

				tokens = []
				for index in range(1, len(text_position)):  # start 31
					# for a_token in res2.get('tokens'):
					# 	print(f"{text_position[index - 1]} <= {a_token.get('start_offset')} and"
					# 		  f" {text_position[index]} > {a_token.get('end_offset')} =====> {text_position[index - 1] <= a_token.get('start_offset') and text_position[index] > a_token.get('end_offset')}")
					filtered_tokens = list(
						filter(lambda x: text_position[index - 1] <= x.get('start_offset') and text_position[
							index] > x.get('end_offset'), res2.get('tokens')))
					tokens.append([a_token.get('token') for a_token in filtered_tokens])

				new_data = data.copy()
				for index in range(len(new_data)):
					new_data[index].update({"news_nm_analyzer": tokens[index]})
		except Exception as es:
			self.logger.error(f"error = {es}")

		return new_data







