# 뉴스 추천 with elastic(nori analyzer)

## 사전 준비사항 (elastic 환경 및 데이터 Write)
### 1. 사용자 설정 파일
아래와 같이 사용자 설정파일이 존재 해야함. (elastic 구축 필요)

다만, 해당 소스 내에서는 proxy 사용을 안함

conf/dev-config.json
```
{
  "elastic": {
    "hosts": "xxxx.xxxx.xxxx.xxxx:xxxx",
    "username": "id",
    "password": "password"
  }
}
```
### 2. elastic에 nori 플러그인 설치 필요
```
# elastic 버전 확인 후, 설치(클러스터일 경우, 모든 클러스터에 설치)
$ bin/elasticsearch-plugin install analysis-nori

# 참고 porxy 기반 설치(클러스터일 경우, 모든 클러스터에 설치) 
$ sudo ES_JAVA_OPTS="-Dhttp.proxyHost=xxxx -Dhttp.proxyPort=xxx -Dhttps.proxyHost=xxxx -Dhttps.
proxyPort=xxx" bin/elasticsearch-plugin install analysis-nori

# 설치 확인
$ bin/elasticsearch-plugin list
analysis-nori

# 삭제
$ bin/elasticsearch-plugin remove analysis-nori

 
# 재실행
$ cd /box/elasticsearch
$ kill -SIGTERM $(cat /box/elasticsearch/PID)
$ elasticsearch/bin/elasticsearch -d -p elasticsearch/PID
 

# 아래는 plugin 없이 기본으로 사용하는 것
get _analyze
{
  "tokenizer": "standard",
  "text": [
    "동해물과 백두산이"
  ]
}
```

### 2. 임시 데이터를 elastic에 write 필요
#### 2.1 csv 파일을 elastic에 write 소스

현재 tests/doc2vec/elastic/analyzer/test_nori_analyzer.py 안에 test_put_data_to_doc2vec 함수를 통해서 수행

#### 2.2 csv 파일 예시
data 폴더내에 news_train_tmp.csv 형식의 파일이 존재하면, 바로 정의된 schema(index는 test-doc2vec)에 맞춰서 데이터가 write 됨

```
news_id,news_content,service_date,news_nm,section
2022030610442141299,홍남기 부총리 겸 기획재정부 장관이 코로나19(COVID-19)에 확진됐다. 앞서 김부겸 국무총리가 코로나19에 확진돼 격리에 들어간 지 3일 만이다. 정부 조직 수장들의 잇따른확진에 국정 운영에도 비상이 걸렸다.기획재정부는 6일 홍남기부총리는 출근시 매일 자가진단을 실시해 음성을 확인했으나 어제(5일) 오후 비서실 유증상자 발생 및 차후 여러 일정 수행등을 감안해 세종에서 코로나19 간이진단 및 PCR검사를 실시한 결과 오늘 오전 양성으로 통보 받았다고 밝혔다.",2022-03-06T13:00:00+09:00,"방역 총리 이어 경제 수장까지…홍남기 부총리 코로나 확진",정치
NBTestTest2,"          이건 테스트야, 제대로 잘 나올지 잘 봐야해. 테스트가 정말 테스트로 나오느냐?를 보는거다.",2022-03-07T13:59:00+09:00,elastic 테스트,테스트
NBTestTest3,"엘라스틱에서 termvector로 조회하면, freq때문에 중복 데이터가 한 개의 결과로 나오기 때문에 text에 대해서 analyzer filed를 쓰지 않는다.",2022-03-07T13:59:00+09:00,elastic termvector 테스트,테스트
```

#### 2.3 elastic test-doc2vec index 구조
아래는 test-doc2vec의 index의 데이터 구조를 의미하며, 

실제 test-doc2vec에 해당하는 index template는 doc2vec/schema_json/NoriAnalyzer/nori-analyzer-template.json에 정의 되어 있으며

자동으로 테스트 코드에서 index template 생성 + index 생성 및 데이터 PUT을 진행함

```
"properties": {
    "news_nm": {
        "type": "keyword"
    },
    "news_content": {
        "type": "text",
        "analyzer": "my_analyzer"
    },
    "service_date": {
        "type": "date"
    },
    "section": {
        "type": "keyword"
    }
}
```

## 실행 
각 단계별로 결과만 살펴보기

소스코드 : tests/doc2vec/test_rec_eda.py 를 참고

**노트북: doc2vec_rec_eda.ipynb 를 참고** 


### 1. 뉴스 데이터(csv)를 elastic기반으로 형태소 분석 수행

함수: test_calculate_d2v_tagdocument
- 해당 내용은 elastic analyzer가 문서를 처리 하는데 있어서 용량 제한이 있어서 한번에 10개 문서씩 파싱을 수행
- 해당 결과를 parsing_data_df.pickle로 저장(path: /data/rec/parsing_data_df.pickle)

### 2. 형태소 분석된 DataFrame 데이터를 TaggedDocuments로 변환 후, 데이터 셋 나눔

함수: test_generate_d2v_tagducument
- elastic으로 분석된 데이터를 train, test로 70:30 비율로 나눠서 TaggedDocument를 생성

### 3. 모델 생성 및 저장

함수: test_save_model
- 모델을 생성하고, 이를 저장함.

### 4. 연관 뉴스 추천

함수: test_recommand_for_news
- test 데이터를 기준으로 유사한 연관 뉴스를 추천


# 뉴스 추천 with Mecab(은전한잎)

## 사전 준비 사항

### osx 사전 준비사항
pip install konlpy 로 바로 설치 후 Mecab 호출시, 아래와 같은 오류가 난다면
```
from konlpy.tag import Mecab
tokenizer = Mecab()


Traceback (most recent call last):

  File "/home/<...>/testvenv/lib/python3.6/site-packages/konlpy/tag/_mecab.py", line 107, in __init__

    self.tagger = Tagger('-d %s' % dicpath)

NameError: name 'Tagger' is not defined
```
터미널에서 아래 명령어 수행 후(아래 부분 설치해도 위의 에러가 아직도 발생함)
```
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
해당 프로젝트가 실행되는 python 가상 환경에서 Mecab을 직접 설치 또는 <b>pip install mecab-python3</b>로 수행
(Finished processing dependencies for mecab-python===0.996-ko-0.9.2)
```
(venv) git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
(venv) cd mecab-python-0.996/
(venv) python setup.py build
(venv) python setup.py install
```

### Mecab(은전한잎) 사전 설치
1. Mecab-ko 설치
meacb-ko url: https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/
(osx 사용자라면, 마지막 su & make install을 sudo make install로 변경)
```
$ cd ~/project
$ tar zxfv mecab-0.996-ko-0.9.2.tar.gz
$ cd mecab-0.996-ko-0.9.2.tar.gz
$ ./configure 
$ make
$ make check
$ su
# make install
```

2. Mecab-ko 사전 설치
사전 url: https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/
만약 중간에 configure 등이 수행이 안된다면 brew install autoconf automake libtool
```
$ cd ~/project
$ tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz
$ cd mecab-ko-dic-2.1.1-20180720.tar.gz
$ ./autogen.sh
$ ./configure
$ make
$ sudo make install
```
3. Mecab-ko 사전 실행 및 테스트
meacb이 설치된 디렉토리에서 mecab -d . 로 수행하면 됩
```
# mecab가 설치된 디렉토리로 이동해서
$ cd ~/project/mecab-ko-dic-2.1.1-20180720
total 219160
-rw-r--r--  1 jmac  admin    262560  4 11 09:46 char.bin
-rw-r--r--  1 jmac  admin      1419  4 11 09:46 dicrc
-rw-r--r--  1 jmac  admin     76393  4 11 09:46 left-id.def
-rw-r--r--  1 jmac  admin  20585296  4 11 09:46 matrix.bin
-rw-r--r--  1 jmac  admin  10583428  4 11 09:46 model.bin
-rw-r--r--  1 jmac  admin      1550  4 11 09:46 pos-id.def
-rw-r--r--  1 jmac  admin      2479  4 11 09:46 rewrite.def
-rw-r--r--  1 jmac  admin    114511  4 11 09:46 right-id.def
-rw-r--r--  1 jmac  admin  80558854  4 11 09:46 sys.dic
-rw-r--r--  1 jmac  admin      4170  4 11 09:46 unk.dic

# mecab -d . 로 실행시킨 다음에, 커서가 깜빡일때 문장 하나 입력 후 엔터를 하면, 문장 분석 결과가 나옴.
$ mecab -d .
안녕하세요. 테스트입니다.
안녕	NNG,행위,T,안녕,*,*,*,*
하	XSV,*,F,하,*,*,*,*
세요	EP+EF,*,F,세요,Inflect,EP,EF,시/EP/*+어요/EF/*
.	SF,*,*,*,*,*,*,*
테스트	NNG,행위,F,테스트,*,*,*,*
입니다	VCP+EF,*,F,입니다,Inflect,VCP,EF,이/VCP/*+ᄇ니다/EF/*
.	SF,*,*,*,*,*,*,*

```

4. python 연동
위 osx 사전 준비사항을 통해서 (mecab-python===0.996-ko-0.9.2)가 이미 설치
아래 명령어로 설치할 경우, mecab-python3-1.0.5 버전이 최신으로 설치됨 (아무래도 mecab-python3-1.0.5로 진행하는게 나을 듯)
```
$ pip install mecab-python3
```

### Mecab(은전한잎) 사용
위에서 사전을 ~/project/mecab-ko-dic-2.1.1-20180720/에 설치하고 이를 빌드하면

자동으로 /usr/local/lib/mecab/dic/mecab-ko-dic 폴더에 rewrite를 수행함
```
from konlpy.tag import Mecab
# m = Mecab("/usr/local/lib/mecab/dic/mecab-ko-dic")
m = Mecab() 

# 품사 태깅(Part-of-speech tagging)
m.pos("안녕하세요. 테스트입니다.")
#[('안녕', 'NNG'), ('하', 'XSV'), ('세요', 'EP+EF'), ('.', 'SF'), ('테스트', 'NNG'), ('입니다', 'VCP+EF'), ('.', 'SF')]

# 형태소 추출
m.morphs("안녕하세요. 테스트입니다.")
# ['안녕', '하', '세요', '.', '테스트', '입니다', '.']

# 명사 추출
m.nouns("안녕하세요. 테스트입니다.")
# ['안녕', '테스트']
```

### Mecab(은전한잎) 사전 추가
위에서 분석된 데이터 중 일부 내용을 사전으로 추가하려면....
참조: https://docs.google.com/spreadsheets/d/1-9blXKjtjeKZqsf4NzHeYJCrr49-nXeRF6D80udfcwY/edit#gid=589544265

데이터는 아래와 같이 들어가는데...
`표층형	0	0	0	품사 태그	의미부류	종성 유무	읽기	타입	첫번째 품사	마지막 품사	원형	인덱스 표현`
- 표층형: 실제 단어를 입력하고
- 0, 0, 0 
- 품사 태그: NNG, NNP 등
- 의미분류: 인명, 지명, * 
- 종성유무: T, F (받침유무로 원 단어의 끝 글자 받침 유무로 T, F 입력)
- 읽기: 
- 타입: Inflect - 활용, Compound - 복합명사, Preanalysis - 기분석
- 첫번째 품사, 마지막 품사: 기분석으로 나눠지는 토큰에 대한 각 품사 입력 (mecab-ko-dic 품사 태그를 참조하여 입력)
- 원형: 토큰 들로 나눠지는 부분 +로 입력 (각 토큰: 표층형/품사태그/의미분류)
- 인덱스 표현: 토큰들로 나눠지는 부분 +로 입력 (각 토큰: 표층형/품사태그/의미부류/PositionIncrementAttribute/PositionLengthAttrigute)

```
예) 여론조사
------------------------------------
여론	NNG,정적사태,T,여론,*,*,*,*
조사	NNG,*,F,조사,*,*,*,*

사용자 사전으로 등록시, 아래와 같이 할 수 있음
------------------------------------
여론조사,,,,NNP,*,F,여론조사,Preanalysis,NNG,NNG,여론/NNG/*+조사/NNG/*
```
실제 사전 추가 후, tools/add-userdic.sh을 통해서 추가한 다음에<br>
재 빌드를 해야 추가됨 (현재 사전 작업한 장소는 나중에 /user/local/lib/mecab/dic/mecab-ko-dic 폴더에 rewrite됨)
```
$ cd ~/project/mecab-ko-dic-2.1.1-20180720 
$ vi user-dic/mecab-user-dict.csv
여론조사,,,,NNP,*,F,여론조사,Preanalysis,NNG,NNG,여론/NNG/*+조사/NNG/*

$ ./tools/add-userdic.sh
$ make install
$ ls -l /usr/local/lib/mecab/dic/mecab-ko-dic
$ sudo chown jmac:staff /usr/local/lib/mecab/dic/mecab-ko-dic/*
$ mecab
여론조사
여론조사	NNP,*,F,여론조사,Preanalysis,NNG,NNG,여론/NNG/*+조사/NNG/
EOF
```

### Mecab(은전한잎) 우선순위 높이기
현재 user-dic 밑에 있는 person.csv에 '한무경'을 추가하면 이미 단어가 '한'+'무경'으로 되어 있어서
우선순위를 높여야 함.
```
$ cd ~/project/mecab-ko-dic-2.1.1-20180720/  
$ ls -l user-dic
total 32
-rw-r--r--@ 1 jmac  staff  1593  7 20  2018 README.md
-rw-r--r--@ 1 jmac  staff   128  4 14 17:26 nnp.csv
-rw-r--r--@ 1 jmac  staff   131  4 14 17:47 person.csv
-rw-r--r--@ 1 jmac  staff   115  4 14 17:22 place.csv

$ cat user-dic/person.csv
까비,,,,NNP,인명,F,까비,*,*,*,*
한무경,,,,NNP,인명,T,한무경,*,*,*,*

# 빌드 및 적용
$ ./tools/add-userdic.sh
$ make clean
$ make install

# 실제 '한무경'의 인명이 적용되었는지 확인하면, 이미 '한'+'무경'의 단어비용이 더 높아서 먼저 나옴
$ mecab
한무경
한	MM,~가산명사,741,2660,2662,-1599,1063
무경	NNG,,1780,3534,2648,-1592,2119
EOS

```

방법은 실제 추가한 단어에 대해서 단어점수를 낮추는 방법이 있음
```
# 위에서 컴파일 해서 적용되면 '한무경'에 대한 단어 비용이 나오는데 지금 보면 '한'+'무경'이 더 높기에
# 아래에 존재하는 '한무경	NNP,인명,1788,3550,5472,-2347,3125'에 대해서 낱말비용을 낮추면 됨
$ ./tools/mecab-bestn.sh
#표현층,품사,의미부류,좌문맥ID,우문맥ID,낱말비용,연접비용,누적비용

한무경
한	MM,~가산명사,741,2660,2662,-1599,1063
무경	NNG,,1780,3534,2648,-1592,2119
EOS
한무경	NNP,인명,1788,3550,5472,-2347,3125
EOS
...

# 낮추는 방법은 위에서 추가한 인명 사전이 컴파일된 파일에서 직접 낱말비용을 낮추는 방법이 있음
# 사용자가 추가한 사전은 ~/project/mecab-ko-dic-2.1.1-20180720/user-dict/person.csv에 있으며
# 컴파일된 인명 사전은 ~/project/mecab-ko-dic-2.1.1-20180720/user-person.csv에 존재함
# 컴파일된 인명 사전의 '한무경'에서 낱말비용 5472->0으로 변경 후, make install 수행하면 됨
$ vi user-person.csv
까비,1788,3549,5472,NNP,인명,F,까비,*,*,*,*
한무경,1788,3550,0,NNP,인명,T,한무경,*,*,*,*

# 빌드 및 적용 (./tools/add-userdic.sh는 하면 안됨)
$ make clean
$ make install
```


