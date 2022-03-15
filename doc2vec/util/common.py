import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm
import re
import pickle


def save_pickle(data_path:str, data):
	with open(data_path, "wb") as fw:
		pickle.dump(data, fw)
		fw.close()


def load_pickle(data_path:str):
	data = None
	if os.path.exists(data_path):
		with open(data_path, "rb") as fr:
			# 로드해서 데이터 확인
			data = pickle.load(fr)
			fr.close()
	return data


def clean_html_tag_in_text(data:str)->str:
	# html 태그 제거
	cleaner = re.compile('<.*?>')
	cleantext = re.sub(cleaner, '', data)

	# &quot;와 같은 태그 제거
	spetial_tag_cleanr = re.compile("&.*?;")
	cleantext2 = re.sub(spetial_tag_cleanr, '', cleantext)
	return cleantext2


def set_pandas_format():
	pd.set_option('display.expand_frame_repr', False)  # 컬럼 다 보기
	pd.set_option('display.width', -1)


def set_matplotlib_sns_font():
	import platform

	if platform.system() == 'Darwin':  # 맥
		apple_gothic = fm.FontEntry(fname='/System/Library/Fonts/Supplemental/AppleGothic.ttf', name='AppleGothic')
		fm.fontManager.ttflist.insert(0, apple_gothic)
		mpl.rcParams['font.family'] = apple_gothic.name
		sns.set(font=apple_gothic.name, rc={"axes.unicode_minus": False, 'figure.figsize':(10,7)}, style='darkgrid')
		# sns.set(font=apple_gothic.name, rc={"axes.unicode_minus": False}, style='darkgrid')
	elif platform.system() == 'Windows':  # 윈도우
		mpl.rc('font', family='Malgun Gothic')
	elif platform.system() == 'Linux':  # 리눅스 (구글 콜랩)
		# !wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
		# !mv malgun.ttf /usr/share/fonts/truetype/
		# import matplotlib.font_manager as fm
		# fm._rebuild()
		plt.rc('font', family='Malgun Gothic')
	mpl.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결

