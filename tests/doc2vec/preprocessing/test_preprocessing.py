import unittest
from datetime import datetime, timedelta
from doc2vec.util.Logger import Logger
from doc2vec.util.conf_parser import Parser
from doc2vec.util.common import *
import rootpath
import pandas as pd
import os
from doc2vec.preprocessing.preprocessing import *
from doc2vec.util.common import load_pickle


class TestPreProcessing(unittest.TestCase):
	def setUp(self) -> None:
		self.logger = Logger(file_name=self.__class__.__name__).logger

		self.p = Parser()

	def test_preprocessing(self):
		text ="""
		<!-- @ckeditor_contents_s@ --><div id='div_NV10442989' class='jtbc_vod'></div>초선 의원과 각 대선주자 캠프를 중심으로 커지고 있는 민주당의 &#39;경선 연기론&#39;. 민주당 초선 모임 &#39;더민초&#39;, 오늘(15일) 아침에 회의를 열었는데요. 대선 경선 연기가 주요 화두였습니다.<!-- @ckeditor_contents_e@ -->
		"""

		text2 = """
		<img src="http://photo.jtbc.joins.com/news/2011/12/02/20111202001000545.jpg" alt="" width="130" />이재술딜로이트 안진 대표이사 1997년 외환위기 당시 33개나 됐던 우리나라 은행은 이제 10여 개의 금융지주그룹과 개별은행으로 단출해졌다. 그동안 20여 개 은행이 퇴출 또는 합병으로 사라졌고 그중 제일과 외환·한미은행은 외국회사에 매각됐다. 제일은행은 미국계 사모펀드인 뉴브리지캐피털에 팔렸다가 스탠더드차터드가 인수해 운영 중이다. 한미은행도 미국계 사모펀드인 칼라일에 넘어갔다가 한국씨티은행이 인수 통합했다.

　요즘 논란의 대상이 되고 있는 외환은행은 외환위기 이후 독일계 은행인 코메르츠에 팔렸다. 2003년 카드대란으로 금융시장이 불안정해지자 대주주와 2대 주주인 수출입은행은 증자를 통한 경영 정상화를 시도했다. 그러나 당시 코메르츠은행은 증자여력이 없었고, 공적자금 투입에 대한 여론도 부정적이었다.

　차선책으로 떠오른 것이 제3자 매각이었고, 인수주체로 물망에 오른 대상은 미국계 사모펀드인 론스타였다. 당시에도 헐값 매각과 대주주 적격성 시비가 있었지만 선택의 여지가 크지 않았던 정부 당국은 금융시장 안정을 위해 신속하게 의사결정을 내렸다.

　알려진 대로 현재 론스타는 외환은행 매각과 관련해 '먹튀' 논란에 휩싸여 있다. 재미있는 것은, 과거 국내기업들이 외국기업에 넘어갈 때는 별 얘기가 없다가 유독 이번에만 말들이 많다는 점이다. 뉴브리지캐피털이 제일은행을 팔 때나, 칼라일이 한미은행을 매각할 때는 먹튀 시비가 없었다. 

　물론 과거와 지금은 상황이 다르다고 할 수 있다. 우리가 아쉬운 게 많아 앞뒤 돌볼 틈 없이 매각을 서둘렀던 때에 비해 지금은 여력이 생겼으니 이것저것 챙겨봐야 한다는 주장이다. 무엇보다 외환은행을 인수할 당사자가 우리 기업(하나금융)이어서 더 억울한 심정이 들 수도 있을 법하다.

　그러나 이 대목에서 분명히 알아둬야 할 것은 이른바 외국계 사모펀드의 속성과 사업방식이다. 론스타나 뉴브리지캐피털 등 사모펀드들은 기본적으로 위험을 감수하고 고수익을 창출하기 위해 투자자로부터 자금을 끌어 모은 펀드다. 실제로 이러한 사모펀드들은 통상 연간 30% 정도의 수익률을 목표로 한다. 이들은 대개 곤경에 처한 기업을 인수해 정상화시키고 나서 3~5년 뒤 회사를 처분하고 빠져 나온다. 그래야 투자자들에게 약속한 만큼의 이익을 실현시켜 줄 수 있고 다음에 다시 투자자를 모을 수 있다.

　론스타가 외환은행 매각으로 거둬들일 이익이 5조원에 달한다고 한다. 짧은 시간에 큰돈을 벌어 간다니 배가 아프긴 하다. 하지만 2조원 이상을 '무려' 8년간 투자했고, 배당 등을 감안할 때 연간 수익률이 약 18%에 그친다고 하니 그들 입장에서는 과하지 않다고 할 수도 있다. 외환위기 때 국내 부실채권과 부동산을 사들인 골드먼삭스 등 외국 투자은행은 연간 30% 이상의 수익률을 올렸다. 2003년 가을 종합주가지수가 780 선에 머물고 있을 때 우량상장 기업에 투자했더라도 그 정도의 수익률은 거뜬하게 올렸을 것이다.

　론스타를 두둔하고 싶은 생각은 없다. 외환은행 인수를 위해 금융위원회에 인가신청 할 당시의 주주 구성을 나중에 실제 투자할 때는 다른 회사로 바꿔 대주주 적격성 시비를 피해가려 한 것은 꼼수를 부린 느낌마저 든다. 그러나 8년이 지난 일을 지금 들춰본들 무슨 실익이 있겠는가. 같은 논리대로라면 제일은행과 한미은행을 인수한 뉴브리지캐피털과 칼라일도 대주주 적격성 심사를 해야 하나.

　걸림돌은 더 있다. 수익률을 지상과제로 삼는 속성상 사모펀드들은 돈이 된다 싶으면 대상이 일반기업이건 금융회사건 가리지 않고 투자한다. 따라서 산업자본인지 금융자본인지 구분하는 것 자체가 애매하다.  징벌적 매각명령을 통해 현재의 시장가격에 매도하도록 해야 한다는 주장도 다분히 감정적이다. 실제로 유럽 재정위기 등으로 현재 주가가 과도하게 떨어진 상태이고, 하나금융의 예정 인수가격은 외환은행 장부가 수준에 불과하다. 

　하루 빨리 비생산적인 먹튀 논쟁에서 헤어나와 전열을 가다듬고 미래로 나아갈 준비를 해야 한다. 우리도 골드먼삭스와 같은 대형 투자은행을 키워야 한다. 론스타에 버금가는 사모펀드를 조성해 유럽시장에 싸게 나온 알짜 기업을 사러 가야 한다. 또 향후 중국 경제가 휘청거릴 때를 대비해 부실채권이나 부실기업을 싸게 사 돈을 벌 기회를 준비해야 한다.

　그런데도 대형 종합 투자사업자 육성법안인 자본시장법 개정안이 론스타의 먹튀 논란에 파묻혀 국회에서 논의조차 되지 못하고 있으니 안타깝다. 우리 금융산업에서 관치나 '국민정서법'이 아니라 법치가 정착되려면 좀 더 차가운 머리가 필요해 보인다.

이재술 딜로이트 안진 대표이사
		"""

		text3="""<object style='height: 390px; width: 640px'><param name='movie' value='http://www.youtube.com/v/O7LyFtOj1k0?version=3&feature=player_detailpage'><param name='allowFullScreen' value='true'><param name='allowScriptAccess' value='always'><embed src='http://www.youtube.com/v/O7LyFtOj1k0?version=3&feature=player_detailpage' type='application/x-shockwave-flash' allowfullscreen='true' allowScriptAccess='always' width='640' height='390'></object>

영국의 데일리메일은 등에 제트 엔진을 멘 스위스의 전직 조종사인 이브 로시가 제트기와 함께  알프스 산맥을 나는데 성공했다고  26일(현지시간) 인터넷판을 통해 보도했다.

올 해  52세인 '제트맨' 로시는 지상에서 헬리콥터를 타고 하늘로 올라간 다음 곧바로 자유낙하를 시도했다. 급전직하로 떨어지는 공중에서 로시는 4개의 제트엔진을 작동한 뒤 방향을 잡고 안전고도를 유지하는 데 성공했다. 로시는 인근에 있던 제트기 2대와 합류한 다음 220km의 속도로 10분 동안 날았다. 영화 '아이언맨'에 필적하는 속도로 하늘은 난 그는 활공으로 하강을 시도한 다음 낙하산을 펴고 안전하게 착륙했다. 

로시가 메고 있는  날개는 폭 2.4m에 두께 3.5cm로 제작됐으며 이곳에 제트 엔진 4개가 달려있다. 제어장치가 들어있는 날개에 연료를 가득채우면 무게만 55kg에 이른다.

로시의 비행을 지켜본 알프스 관광객들은 '로시가 새인지 전투기인지 모르겠다','불안전한 이착륙만 보완한다면 사람이 하늘을 날아 다니는 날도 멀지않은 것 같다'고 말한다.  [출처=http://www.dailymail.co.uk]"""

		text4="""<img src="http://photo.jtbc.joins.com/news/2011/11/29/20111129154500693.jpg" alt="" />

영국의 데일리메일은 48년 전 캐나다 풋볼의 두 전설적인 인물이 최근 공식적인 행사에서 난투극을 벌인 영상을 28일(현지시간) 소개했다.

캐나다 풋볼의 전설이라 불리는 조 캡과 안젤로 모스카는 48년 동안 참았던 감정을 오랜만에 마련된 공식 무대에서 지팡이와 주먹을 주고 받는 난동을 벌였다. 

1963년 캐나다 슈퍼볼 그래이 컵에서 그들이 맞대결을 할 당시 조 캡은 브리티시 라아온스의 쿼터백이었고, 안젤로 모스카는 해밀턴 타이거 켓츠의 전방 수비수였다. 두 선수는 경기를 할 때마다 감정적인 대립을 가져왔고 이런 감정은 세월이 흐른 현재까지도 남아있었다.

최근 열린 오찬 모임에서 캡은 모스카에게 올리브 나무 가지를 선물하려 했다. 그러나 모스카는 장난스럽게 건네는 캡의 올리브 가지를 받지 않겠다는 몸짓을 했다. 그러나 이런 행동을 불쾌하게 여긴 캡은 장난스럽게 모스카의 코에 올리브 가지를 집어넣으려 했다.

이런 행동에 화가난 모스카는 들고 있던 지팡이로 캡의 머리에 일격을 가했고 사태는 걷잡을 수 없이 벌어졌다. 

영상을 본 테티즌들은 "경기중 쌓인 감정이 세월이 흘러도 변화지 않은 것 같다","같이 늙어가는 마당에 난투극은 좀 심한 것 같다"는 반응이다.[출처=데일리메일]"""

		tokenizer = get_tokenizer("mecab")
		print(clean_content(text))
		print("="*10)
		print(clean_content(text2))
		print("=" * 10)
		print(clean_content(text3))
		print("=" * 10)
		print(clean_content(text4))

		print(tokenize(tokenizer, clean_content(text4)))
		print("="*10)
		print(tokenizer.pos(clean_content(text)))



	def test_do_generate_tagged_documment(self):

		data = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]))

		d2v_result = generate_tagged_document(data)
		print(d2v_result)

	def test_load_user_history(self):

		user_df = user_history_df("user_history.csv")
		print(user_df.tail())

	def test_load_pickle(self):

		data = load_pickle(os.path.join(rootpath.detect(), *["data", "parsing_data.pickle"]))

		for k, v in data.items():
			print(k, v[:10])




	def test_load_history(self):
		"step 1. 중복제거하여 2021.11~2022.02까지 사용자 history 생성: 211523"
		data_path = os.path.join(rootpath.detect(), *['data', 'rec','jtbcnews-raw-20211101-20220228.csv'])
		old_df = load_pdf(file_path=data_path)

		data_path = os.path.join(rootpath.detect(), *['data', 'jtbcnews_user_history_211201_220402.csv'])
		new_df = load_pdf(file_path=data_path)
		new_df['reqtime'] = new_df['reqtime'].apply(lambda x: datetime.strptime(x, "%d/%b/%Y:%H:%M:%S +0900"))

		print("old_df: ", old_df.shape)
		print("new_df: ", new_df.shape)
		df= pd.concat([old_df, new_df], axis=0)
		print("df: ", df.shape)


		new_df = df.drop_duplicates(['remote', 'reqtime', 'method', 'path', 'parms', 'code', 'size', 'agent',
									 'referer', 'xforward', 'reqbody', 'timestamp', 'hostname',
									 'es_index_name', 'action_name', 'parm_referer', 'parm_current_path',
									 'parm_member_id', 'parm_user_id', 'parm_member_type', 'parm_origin',
									 'ua_browser_family', 'ua_browser_version', 'ua_os_family',
									 'ua_os_version', 'ua_os_major_version', 'ua_device', 'news_nm', 'tag',
									 'news_id', 'ua_browser_major_version', 'action_type'], keep='last')
		print("remove duplicate df: ", new_df.shape)
		new_df['reqtime'] = pd.to_datetime(new_df['reqtime'])
		new_df = new_df.sort_values(by='reqtime')
		new_df.reset_index(drop=True, inplace=True)

		user_df = new_df[['reqtime', 'referer', 'action_name', 'parm_member_id', 'parm_member_type', 'news_id', 'news_nm',
						'parm_user_id', 'parm_origin',
						 'ua_browser_family', 'ua_browser_version', 'ua_os_family',
						 'ua_os_version', 'ua_os_major_version', 'ua_device', 'ua_browser_major_version'
		]].copy()
		print(user_df.shape)
		user_df = user_df[(~user_df.parm_member_id.isnull()) & (~user_df.news_id.isnull())]

		# 결측치 제거 : 616061 ->  211523 (viewpage가 페이지 로딩부터 시작이므로, 데이터가 없어서 사라짐)
		print("remove..", user_df.shape)
		write_pdf(user_df, "jtbcnews-raw-20211101-20220228.csv")

	def test_check_user_history_csv(self):
		"step2. 해당 기간동안 본 news_id 수: 15098"
		df = load_pdf('jtbcnews-raw-20211101-20220228.csv')
		print(df.shape)
		print(df.action_name.unique())

		unique_news_ids = df.news_id.unique().tolist()
		print("해당 기간동안 본 news_id 수: ", len(unique_news_ids))

		save_pickle(os.path.join(rootpath.detect(), *['data', 'rec', 'unique_news_ids.pickle']), unique_news_ids)




if __name__ == "__main__":
	suite = unittest.TestLoader().loadTestsFromTestCase(TestPreProcessing)
	result = unittest.TextTestRunner(verbosity=2).run(suite)

	import sys

	print(f"unittest result: {result}")
	print(f"result.wasSuccessful()={result.wasSuccessful()}")
	# 정상종료는 $1 에서 0을 리턴함
	sys.exit(not result.wasSuccessful())