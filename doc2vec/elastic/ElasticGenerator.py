import re
from abc import *
from elasticsearch import Elasticsearch
from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
from elasticsearch.helpers import bulk


class ElasticGenerator(metaclass=ABCMeta):
	def __init__(self, hosts: str, username: str, password: str):
		self.logger = Logger(file_name=type(self).__name__).logger
		self.hosts = hosts
		self.username = username
		self.password = password

		self.p = Parser()

		self._es_client = Elasticsearch(
			hosts=self.hosts,
			http_auth=(self.username, self.password), request_timeout=300, max_retries=1, retry_on_timeout=False)

	def do_elastic_write(self, index_name:str, elk_doc:list):
		try:

			bulk(self.es_client, elk_doc, index=index_name, refresh="wait_for")

		except Exception as es:
			self.logger.error(f"Error={es}")

	def do_elasitc_write(self, index_name: str, template_name: str, template: dict,
						 alias_name: str, aliases: dict, elk_doc: list):
		try:

			self.make_index_and_template(
				client=self.es_client,
				index_name=index_name,
				template_name=template_name,
				template=template,
				alias_name=alias_name,
				aliases=aliases)

			bulk(self.es_client, elk_doc, index=alias_name, refresh="wait_for")

		except Exception as es:
			self.logger.error(f"error = {es}")

	@property
	def es_client(self, request_timeout=300):
		if self._es_client is not None:
			return self._es_client
		else:
			self._es_client = Elasticsearch(
				self.hosts,
				http_auth=(self.username, self.password), request_timeout=300,
				timeout=7800, max_retries=1, retry_on_timeout=False)
		return self._es_client

	def make_index_and_template(self, client, index_name: str, template_name: str, template: dict,
								alias_name: str, aliases: dict):
		"""
        index_template, index, alias 등 schema 자동 생성 부분
        (기존) index template json 파일 로드 → index template 생성 → index 생성 → alias 생성
            template_name: 'naver-ranking-news-template'
            index_name: 'origin-naver-ranking-news-2022'
            alias_name: 'naver-ranking-news-2022'
            데이터 write to index_name

        (수정) index template json 파일 로드 → index template 생성 → index 생성 → alias 생성
            template_name: 'naver-ranking-news-template'
            index_name: 'naver-ranking-news-000001' (최초 1회 생성, 나머지는 rollover 적용)
            alias_name: 'naver-ranking-news'
            데이터 write to alias_name

        자동 생성 순서
        1. 최초 index_name(~~-000001)이 생성 안되었을 경우
            1.1 최초 index_name 생성 (~~~000001)
            1.2 index template 생성 & 수정
            1.3 alias 적용
        2. 최초 index_name이 있을 경우
            2.1 index template만 수정

        :param index_name:  실제 생성시킬 index 명
        :param template_name: 실제 index에 매핑될 index_pattern의 이름
        :param template: 생성시킬 index_pattern에 대한 json
        :param alias_name:
        :param aliases:
        :return:
        """
		try:
			if not client.indices.exists(index=index_name):
				# index_template와 index 생성

				self._create_index_pattern(client=client, template_name=template_name, template=template)
				self.logger.info(f"create index: {index_name}")
				client.indices.create(index=index_name)

				# alias 생성
				# (index 최초 생성시킬때 index_pattern과 index, alias를 최초로 한번 생성시킨다.)
				# 이유는 년(YYYY)이 넘어가면서 index를 새로 생길때, 같이 alias도 생긴다.
				self.create_alias(client=client, index_name=index_name, alias_name=alias_name)

			else:
				# index_template 생성
				self.logger.info(f"Is in index_pattern({template_name})? ")
				self._create_index_pattern(client=client, template_name=template_name, template=template)

		# # alias 생성
		# self.create_alias(index_name=index_name, alias_name=alias_name)
		except Exception as es:
			self.logger.error(f"Error={es}")

	def delete_index_template(self, index_template_name: str):
		try:
			if self.es_client.indices.exists_index_template(index_template_name):
				self.es_client.indices.delete_index_template(index_template_name)
				self.logger.info(f"deleted index template : {index_template_name}")
		except Exception as es:
			self.logger.error(f"ERROR={es}")

	def delete_index(self, index_name):
		try:
			if self.es_client.indices.exists(index=index_name):
				self.es_client.indices.delete(index=index_name)
				self.logger.info(f"deleted index : {index_name}")
		except Exception as es:
			self.logger.error(f"ERROR={es}")

	def delete_alias(self, index_name, alias_name):
		try:
			if self.es_client.indices.exists_alias(alias_name):
				actions = {
					"actions": [
						{
							"remove": {
								"index": index_name,
								"alias": alias_name,
							}
						}
					]
				}
				self.es_client.indices.put_alias(index=index_name, name=alias_name,
												 body=actions)
				self.logger.info(f"remove alias {alias_name} from {index_name} index")
				self.logger.info(f"{alias_name} alias actions {actions}")
				return True
		except Exception as es:
			self.logger.error(f"ERROR={es}")
		return False

	def create_alias(self, client, index_name: str, alias_name: str) -> bool:
		# appended new alias
		# self.logger.info(f"AUTO_APPLY_ALIAS= {self.p.auto_apply_alias}")
		try:

			# if self.p.auto_apply_alias:

			# alias_name에 해당하는 alias가 있는지 체크 또한 alias_name으로 index가 있는지 체크
			# and not self.es_client.indices.exists(alias_name)
			if not client.indices.exists_alias(alias_name) and not client.indices.exists(alias_name):
				# self.es_client.indices.put_alias(index=index_name, name=new_alias, body={"is_write_index": True})
				actions = {
					"actions": [
						{
							"add": {
								"index": index_name,
								"alias": alias_name,
								"is_write_index": True
							}
						}
					]
				}
				client.indices.put_alias(index=index_name, name=alias_name,
										 body=actions)
				self.logger.info(f"append new alias({alias_name}) to index({index_name})")
				self.logger.info(f"{alias_name} alias actions {actions}")
				return True
			elif client.indices.exists(alias_name):

				# actions = {
				#     "actions": [
				#         {
				#             "add": {
				#                 "index": index_name,
				#                 "alias": alias_name,
				#                 "is_write_index": True
				#             }
				#         }
				#     ]
				# }
				# self.es_client.indices.update_aliases(
				#                                  body=actions)
				self.logger.warning(f"an index exists with the same name as the alias({alias_name})")
				return True
			elif client.indices.exists_alias(alias_name):
				self.logger.warning(f"the alias({alias_name}) exists.")

		except Exception as es:
			self.logger.error(f"ERROR={es}")

		return False

	def get_alias_name(self, index_name):
		"""
        # TODO 나중에 re(regex) 추가
        :param index_name: origin-hourly-summary-2021
        :return:
            hourly-summary-2021
        """

		alias_name = index_name.replace('origin-', '') if 'origin-' in index_name else index_name

		return alias_name

	def get_indexnames_from_alias(self, alias_name: str):
		"""
        get origin-hourly-summary-2021/_alias
        {
          "origin-hourly-summary-2021" : {
            "aliases" : {
              "hourly-summary-2021" : { }
            }
          }
        }

        :param alias_name:  hourly-summary-2021
        :return:
            [origin-hourly-summary-2021]
        """
		try:
			if self.es_client.indices.exists_alias(name=alias_name):
				return [index_name for index_name in list(self.es_client.indices.get_alias(name=alias_name).keys())]

		# {
		#   "origin-hourly-summary-2021" : {
		#     "aliases" : {
		#       "hourly-summary-2021" : { }
		#     }
		#   }
		# }
		except Exception as es:
			self.logger.error(f"ERROR={es}")
		return []

	def create_index_with_setting(self, index_name: str, template_name:str, template: dict):
		"""
        index를 생성할 때, setting과 함께 생성하며,
        만약 index가 있다면, 해당 index의 setting을 체크하여 동일하면 패스,
                                                     다르면 colse->modify->open을 수행
        """
		try:
			if template:
				if not self.es_client.indices.exists(index=index_name):
					# index가 없을 경우, index를 setting과 더불어 생성
					self._create_index_pattern(template_name=template_name, template=template)
					self.logger.info(f"create index: {index_name}")
					self.es_client.indices.create(index=index_name)
				else:
					before_setting = self.es_client.indices.get_settings(index=index_name)

					before_analysis_setting = before_setting.get(index_name).get('settings').get('index').get(
						'analysis')
					if before_analysis_setting != template.get('template').get('settings').get('analysis'):
						# nori analyzer가 변동으로
						# 1. 아예 삭제(또는 close) 하고
						# 2. setting 변경 후
						# 3. open 수행
						self.logger.info(f"modify settings in index: {index_name}")
						settings = template.get('template').get('settings')
						settings.pop('lifecycle')
						settings.pop('number_of_shards')
						settings.pop('number_of_replicas')

						self.es_client.indices.close(index=index_name)
						self.es_client.indices.put_settings(index=index_name,
															body=settings)
						self.es_client.indices.open(index=index_name)
			else:
				self.logger.warning(f"{index_name}'s settings is None ({index_name}.json was not created.)")

		except Exception as es:
			self.logger.error(f"error = {es}")

	def _create_index_pattern(self, template_name, template):
		try:
			if template:
				if not self.es_client.indices.exists_index_template(template_name):
					# index_template 생성
					self.logger.info(f"create index_template: {template_name}")

					# put_index_template에서 create옵션이 default로 False가 되어 있어야 자동으로 index_pattern update
					self.es_client.indices.put_index_template(name=template_name, body=template)
				else:
					self.logger.warning(f"index_pattern= {template_name} already exists.")
					self.es_client.indices.put_index_template(name=template_name, body=template)
			else:
				self.logger.warning(f"{template_name}'s template is None (index_template was not created.)")
		# else:
		#     # appended new alias
		#     new_alias = index_name.replace('origin-', '')
		#     if not self.es_client.indices.exists_alias(new_alias):
		#
		#         self.es_client.indices.put_alias(index=index_name, name=new_alias)
		#         self.logger.info(f"append new alias {new_alias} to {index_name}")
		#         # ori_index_templates = self.es_client.indices.get_index_template(template_name)
		#         # # ori_index_template = ori_index_templates.get('index_templates')[0]
		#         # #
		#         # #
		#         # # # ori_template_name_def = ori_index_template.get('name')
		#         # #
		#         # # # index_template:
		#         # # #   - index_patterns
		#         # # #   - template
		#         # # #       - settings, mappings, aliases
		#         # # #   - composed_of
		#         # # ori_index_template_def = ori_index_template.get('index_template')
		#         # # # ori_def_index_patterns = ori_index_template_def.get('index_patterns')
		#         # # ori_def_index_template = ori_index_template_def.get('template')
		#         # # # ori_def_index_composed_of = ori_index_template_def.get('composed_of')
		#         # #
		#         # # # update aliases
		#         # # ori_aliases = ori_def_index_template.get('aliases')
		#         # # ori_aliases.update(aliases)
		#         # # ori_def_index_template.update({'aliases': ori_aliases})
		#         # # ori_index_template_def.update({'template': ori_def_index_template})
		#         # # ori_index_template_def.pop('composed_of')
		#         # # self.logger.info(f"update '{index_name.replace('origin-', '')}' aliases to {template_name}")
		#         # # self.es_client.indices.put_index_template(name=template_name, body=ori_index_template_def)

		except Exception as es:
			self.logger.error(f"ERROR={es}")