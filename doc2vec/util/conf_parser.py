from configparser import ConfigParser
from pathlib import Path
import os

from doc2vec.util.Logger import Logger
from doc2vec.util.Singleton import Singleton
import json

class Parser(metaclass=Singleton):

    def __init__(self):
        self.root_path = Path(__file__).parent.parent.parent

        self._conf_parser = ConfigParser()
        self.logger = Logger(file_name=type(self).__name__).logger

        self._news_ranking_url = None
        self._config = None
        self._load_conf(conf_path=self.root_path)
        self._load_build_conf(conf_path=self.root_path)

    def _load_elastic_conf(self):
        # self._elastic_shard = self._conf_parser.getint("elastic", "ELASTIC_DEV_SHARD") if self._build_level in [
        #     'dev','local'] \
        #     else self._conf_parser.getint("elastic", "ELASTIC_PROD_SHARD")
        # self._elastic_replica = self._conf_parser.getint("elastic", "ELASTIC_DEV_REPLICA") if self._build_level in [
        #     'dev', 'local']            \
        #     else self._conf_parser.getint("elastic", "ELASTIC_PROD_REPLICA")
        self._elastic_shard = self._conf_parser.getint("elastic", "ELASTIC_DEV_SHARD") if self._build_level in [
            'dev','local'] \
            else None
        self._elastic_replica = self._conf_parser.getint("elastic", "ELASTIC_DEV_REPLICA") if self._build_level in [
            'dev', 'local']            \
            else None

    def _load_conf(self, conf_path:Path):
        build_path = os.path.join(conf_path, *["conf", "build.ini"])
        if os.path.isfile(build_path):
            self.logger.info(f"load build.ini: {build_path} ")

            self._conf_parser.read(build_path)
            self._build_level = self._conf_parser.get("build", "BUILD_LEVEL")
            self._load_elastic_conf()

    def _load_build_conf(self, conf_path:Path):
        conf_file_name = "dev-config.json" if self.build_level in ["dev","local"] else "prod-config.json" if \
            self.build_level == "prod" else "qa-config.json"
        config_file = os.path.join(conf_path, *["conf", conf_file_name])
        if os.path.isfile(config_file):
            self.logger.info(f"load BUILD_LEVEL({self.build_level}) config: {config_file} ")

            with open(config_file, 'r') as f:
                self._config = json.load(f)

    @property
    def elastic_shard(self):
        return self._elastic_shard

    @property
    def elastic_replica(self):
        return self._elastic_replica

    @property
    def db_conf(self):
        result = {}
        if self._config:
            result = self._config.get('news-db')

        return result

    @property
    def elastic_conf(self):
        if self._config:
            result = self._config.get('elastic')
            if result:
                return result

        return None, None, None

    @property
    def elastic_cloud_conf(self):
        if self._config:
            result = self._config.get('jnd-elastic-cloud')
            if result:
                return result

        return None, None, None


    @property
    def build_level(self):
        return self._build_level

