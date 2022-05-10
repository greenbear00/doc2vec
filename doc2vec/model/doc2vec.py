import gensim
import multiprocessing
# 아래 FAST_VERSION을 쓰면, 자체적으로 c complier를 사용하여 학습 시킴
from gensim.models.word2vec import FAST_VERSION

def train_for_model(model: gensim.models.doc2vec.Doc2Vec, docs: list):
	model.build_vocab(docs)
	model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)


def build_model() -> gensim.models.doc2vec.Doc2Vec:
	# using all available CPU cores.
	cores = multiprocessing.cpu_count()

	# https://radimrehurek.com/gensim/models/doc2vec.html
	# vector_size: Dimensionality of the feature vectors
	# window: 앞 뒤로 단어 보는거 (사이즈가 커지면, 훈련 결과로 나오는 word vectors의 성능이 높아지지만, 훈련 시간이 오래 걸림. 보통 5-10)
	# vector_size: 벡터 차원의 크기
	# alpha: learning rate
	# min_count: 학습에 사용할 최소 단어 빈도 수
	# dm: 학습방법 1 = PV-DM, 0 = PV-DBOW (거의 PV-DM이 성능이 더 잘 나옴)
	# negative: Complexity Reduction 방법, negative sampling
	# max_epochs: 최대 학습 횟수
	# 정확도 참고: Dipika Baad의 Doc2Vec 매개변수 조합에 대한 정확도
	# (
	# 	window=10,
	# size=150,
	# alpha=0.025,
	# min_alpha=0.025,
	# min_count=2,
	# dm =1,
	# negative = 5,
	# seed = 9999)
	model = gensim.models.doc2vec.Doc2Vec(vector_size=1000, min_count=1, epochs=100,
										  alpha=0.025, min_alpha=0.00025, workers=cores,
										  negative=3, window=10, dm=1, seed=9999)
	# model = gensim.models.doc2vec.Doc2Vec(vector_size=1000, min_count=2, epochs=100,
	# 									  alpha=0.025, min_alpha=0.00025, workers=cores,
	# 									  window=15, sampling_threshold=1e-5, negative_size=5, dm=1, seed=9999)

	print(type(model))

	return model
