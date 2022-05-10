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
from dateutil import relativedelta

URL_PATTERN = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

URL_PATTERN2 = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                 re.IGNORECASE) # http로 시작되는 url
URL_PATTERN3 = re.compile(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
                          re.IGNORECASE) # http로 시작되지 않는 url

# html 태그 제거
HTML_PATTERN = re.compile('<.*?>')

# &quot;와 같은 태그 제거
HTML_CHAR_PATTERN = re.compile("&.*?;")
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)
WIKI_REMOVE_CHARS = re.compile("'+|(=+.{2,30}=+)|__TOC__|(ファイル:).+|:(en|de|it|fr|es|kr|zh|no|fi):|\n", re.UNICODE)
WIKI_SPACE_CHARS = re.compile("(\\s|゙|゚|　)+", re.UNICODE)
EMAIL_PATTERN = re.compile('([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+([\.|\,][A-Z|a-z]{2,})+', re.UNICODE)
WIKI_REMOVE_TOKEN_CHARS = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)

# <!-- --> 로 된 주석 제거
EXTRACT_CONTENT_PATTERN = re.compile("\<!-(.*?)-\>", re.UNICODE)

# [출처=이데일리] 또는 [출처=http...] 제거
REF_PATTERN = re.compile("\[출처(.*?)\]", re.UNICODE)
# 〈사진-연합뉴스〉 제거
REF_PATTERN2 = re.compile("〈사진(.*?)〉", re.UNICODE)

REPORTER_PATTERN = re.compile("\. *([\w+]{2,5} 기자|영상디자인 +[\w+]{2,5}|[\w+]{2,5} 디지털뉴스팀|사진MBC제공)", re.UNICODE)

# 일부 구두점 제거
# PUNCTATION_PATTER = re.compile('[-=+,#/\?:^$.@*\"※~&%ㆍ·!』\\‘|\(\)\[\]\<\>`\'…》;]', re.UNICODE)
PUNCTATION_PATTER = re.compile('[-=+,#/\?:^$@*\"※&%ㆍ·!』\\‘|\(\)\[\]\<\>`\'…》;]', re.UNICODE)

EXTRACT_TITLE_PATTERN = re.compile("\[(.*?)\]", re.UNICODE)
EXTRACT_TITLE_PATTERN2 = re.compile("\((.*?)\)", re.UNICODE)
DATE_PATTERN = re.compile('\d{4}\.(0?[1-9]|1[012])\.(0?[1-9]|[12][0-9]|3[01])$', re.UNICODE)

STOP_WORDS_PATTERNS = re.compile(r'(특별취재반|기자|앵커)')


def clean_content(content):
    content = re.sub(EXTRACT_TITLE_PATTERN, ' ', content)
    content = re.sub(EXTRACT_TITLE_PATTERN2, ' ', content)
    content = re.sub(EXTRACT_CONTENT_PATTERN, ' ', content)

    content = re.sub(HTML_PATTERN, ' ', content)
    content = re.sub(HTML_CHAR_PATTERN, ' ', content)
    content = re.sub(WIKI_REMOVE_CHARS, ' ', content)
    content = re.sub(WIKI_SPACE_CHARS, ' ', content)
    content = re.sub(EXTRACT_CONTENT_PATTERN, ' ', content)
    content = re.sub(EMAIL_PATTERN, ' ', content)
    content = re.sub(REF_PATTERN, ' ', content)
    content = re.sub(REF_PATTERN2, ' ', content)
    content = re.sub(URL_PATTERN2, ' ', content)
    content = re.sub(URL_PATTERN3, ' ', content)

    content = re.sub(PUNCTATION_PATTER, ' ', content)
    # #     print(content)
    # #     print("="*100)
    content = re.sub(REPORTER_PATTERN, ' ', content)
    content = re.sub(STOP_WORDS_PATTERNS, ' ', content)

    content = re.sub(MULTIPLE_SPACES, ' ', content)

    content = content.lower()
    content = content.strip()
    content = re.sub(DATE_PATTERN, '', content)
    return content


ARTICLE_CONTENTS_STOP_WORDS = [
    "JTBC뉴스운영팀", "디지털뉴스팀", "기자", "앵커", "특별취재반", "joongang.co", "joongang.co.kr", "영상디자인", "mbc제공",
    "joongang.co,kr", "joongang.co kr", 'a', 'b', '모바일운영팀', '속보', '연합뉴스', '헤드라인', 'vcr', '때',
    "곳", "아침&",'뉴스운영팀', 'pick', '복마크', '뒤', '모', '씨', '조', '달', '산', '여', '속', '해', '밑줄', '정치부회의', '저녁',
    '땐', '화면제공', 'jtbc', '이후', '전', '후', '집', '밤', '날', '앞', '뒤', '점','안', '간', '위', '선', '주',
    '수', '발','관','반','시','p','방','극','cm', 'm', '심','급','x','k','기','폭','호',
    'yonhap', 'photo', '쌍', '션', '한데', '텐', '베', '렛', '데어', '니지', '시스', '헴', '커',
    '지고', '힐', '람', '납', '룻', '서서', '아', 'z', '면'
]

def article_contents_stopwords(tokens:list, process_type:str='morphs'):
    """

    :param tokens:
    :param process_type: morphs, nouns, pos로 정의됨
    :return:
    """
    result = []
    if process_type in ['morphs', 'nouns']:
        for token in tokens:
            if token not in ARTICLE_CONTENTS_STOP_WORDS:
                result.append(token)
    else:
        result = _extract_tag_with_stopwords(tokens, stopwords=ARTICLE_CONTENTS_STOP_WORDS)
    return result


def remove_title_stopwords(title):
    """
    뉴스 제목에서 '[날씨]'만 제외하고 의미 없는 제목 삭제
    - 특히 [], ()로 되어 있는 안의 글들은 삭제 처리 함
    :param title:
    :return:
    """

    # if title.find('[속보]')>=0 :
    #     return title
    # elif title.find('[포토]')>=0:
    #     return title
    # elif title.find('[단독]') >= 0:
    #     return title
    if title.find('[날씨]') >= 0:
        title = re.sub(EXTRACT_TITLE_PATTERN, '', title)
        title = re.sub(EXTRACT_TITLE_PATTERN2, '', title)
        return '[날씨] '+title
    elif title.find('[인사]') >= 0:
        title = re.sub(EXTRACT_TITLE_PATTERN, '', title)
        title = re.sub(EXTRACT_TITLE_PATTERN2, '', title)
        return '[인사] '+title
    else:
        title = re.sub(EXTRACT_TITLE_PATTERN, '', title)
        title = re.sub(EXTRACT_TITLE_PATTERN2, '', title)
        return title


def article_title_stopwords(tokens, process_type:str='morphs'):
    """

    :param tokens:
    :param process_type: morphs, nouns, pos로 정의됨
    :return:
    """
    result = []
    if process_type in ['morphs', 'nouns']:
        for token in tokens:
            if token not in []:
                result.append(token)
    else:
        result = _extract_tag_with_stopwords(tokens, stopwords=ARTICLE_CONTENTS_STOP_WORDS)
    return result


target_tags = ['NNG', 'NNP', "SL"]

def _extract_tag_with_stopwords(tokens:list, stopwords:list):
    """
    https://docs.google.com/spreadsheets/d/1-9blXKjtjeKZqsf4NzHeYJCrr49-nXeRF6D80udfcwY/edit#gid=589544265
    tokens에 품사 태킹 정보에서 아래를 포함 시킨다.
        NNG	일반 명사
        NNP	고유 명사
        VV	동사
        VA	형용사
        VX	보조 용언
        VCP	긍정 지정사
        VCN	부정 지정사
        SL 외국어
        MAG	일반 부사 (예: 오늘)

    :param tokens:
    :param stopwords:
    :return:
    """
    result = []
    # target_tags = ['Noun', 'Verb', 'Adjective']
    # target_tags = ['NNG', 'NNP', 'VA', 'VV', 'VCP',  'VCN', "SL", "MAG"]
    # target_tags = ['NNG', 'NNP', "SL"]
    for token, tag in tokens:
        # 명사, 동사, 형용사만 사용
        if tag.find('+')>=0:
            for a_tag in tag.split('+'):
                if (a_tag in target_tags) and (token not in stopwords):
                    result.append('/'.join([token, tag]))
                    break
        else:
            if (tag in target_tags) and (token not in stopwords):
                result.append('/'.join([token, tag]))
    return result


# def article_title_stopwords(tokens:list, process_type:str='morphs'):
# 	"""
#
# 	:param tokens:
# 	:param process_type: morphs, nouns, pos로 정의됨
# 	:return:
# 	"""
# 	# def stopwords(content):
# 	# 	return re.sub(STOP_WORDS_PATTERNS, ' ', content)
# 	result = []
# 	if process_type in ['morphs', 'nouns']:
# 		for token in tokens:
# 			if token not in TITLE_STOP_WORDS:
# 				result.append(token)
# 	else:
# 		result = _extract_tag_with_stopwords(tokens, stopwords=TITLE_STOP_WORDS)
# 	return result




def get_tokenizer(tokenizer_name:str = "mecab"):
    tokenizer = None
    if tokenizer_name.lower() == "mecab":
        tokenizer = Mecab(dicpath="/usr/local/lib/mecab/dic/mecab-ko-dic")

    return tokenizer


def getMonthRage(year, month):
    this_month = datetime(year=year, month=month, day=1).date()
    next_month = this_month + relativedelta.relativedelta(months=1)
    # print(f"this month: {this_month}")
    # print(f"next month: {next_month}")

    first_day = this_month
    last_day = next_month - timedelta(days=1)
    # print(f"first day: {first_day}")
    # print(f"last day: {last_day}")

    return (first_day, last_day)

def concat_news_title_and_contents(clean_contents_len, news_title, news_contents, condition_len=50):
    """
     뉴스 본문에서 clean_contents를 거치고 나온 본문의 길이가 30보다 작으면 뉴스 타이틀만, 길면 title+content를 바탕으로
     tagged_doc 생성
    :param clean_contents_len: (int)
    :param news_title: (list) 형태소 분석된 데이터
    :param news_contents: (list) 형태소 분석된 데이터
    :return:
    """
    result = None
    if clean_contents_len<=condition_len:
        result = news_title
    else:
        result = news_title + news_contents
    return result

def concat_news_section(news_section, tagged_doc):
    return news_section+tagged_doc


def tokenize(tokenizer, content, type:str="morphs"):
    """

    :param tokenizer: mecab 등 tokenizer
    :param content: cleaning된 content
    :param type: morphs, nouns, pos로 정의됨
    :return:
    """
    # install : https://konlpy-ko.readthedocs.io/ko/v0.4.3/install/

    # 예제문장 : 안녕하세요. 테스트입니다.
    if type == "morphs":
        # 품사 태깅(Part-of-speech tagging)
        # 결과: [('안녕', 'NNG'), ('하', 'XSV'), ('세요', 'EP+EF'), ('.', 'SF'), ('테스트', 'NNG'), ('입니다', 'VCP+EF'), ('.', 'SF')]
        result = tokenizer.morphs(content)
    elif type == "pos":
        # 형태소 분석
        # 결과: ['안녕', '하', '세요', '.', '테스트', '입니다', '.']
        result = tokenizer.pos(content)
    else:
        # 명사 추출
        # 결과: ['안녕', '테스트']
        result = tokenizer.nouns(content)
    return result






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
