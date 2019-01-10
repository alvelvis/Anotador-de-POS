#!/usr/bin/env python3
import estrutura_dados
from sklearn import preprocessing
from sklearn import tree
import pickle
import copy
import string
import re
import sys

def tokenizar(sentenca):

	contractions = ['da', 'das', 'do', 'dos', 'dela', 'nos', 'ao', 'na', 'aos', 'no', 'às', 'nesse', 'pelos', 'numa', 'à', 'nas', 'pela', 'pelo', 'daí', 'deste', 'neste', 'desta', 'num', 'naquele', 'daquela', 'connosco', 'destes', 'dessas', 'pelas', 'duma', 'daquele', 'deles', 'nesta', 'desse', 'dessa', 'dele', 'daqueles', 'àquele', 'disso', 'disto', 'naquela', 'daquilo', 'delas', 'nele', 'nela', 'nessa', 'nestas', 'nessas', 'noutras', 'numas', 'nelas', 'destas', 'nalguns', 'dantes', 'daqui', 'nestes', 'convosco', 'nesses', 'daquelas', 'consigo', 'dum', 'noutro', 'desses', 'do(s)', 'neles', 'nalgumas', 'nisso', 'noutra', 'naqueles', 'noutros', 'comigo', 'dali', 'lha', 'àqueles', 'àquela', 'nisto', 'naquilo']

	dettached = ['de a', 'de as', 'de o', 'de os', 'de ela', 'em os', 'a o', 'em a', 'a os', 'em o', 'a as', 'em esse', 'por os', 'em uma', 'a a', 'em as', 'por a', 'por o', 'de aí', 'de este', 'em este', 'de esta', 'em um', 'em aquele', 'de aquela', 'com nós', 'de estes', 'de essas', 'por as', 'de uma', 'de aquele', 'de eles', 'em esta', 'de esse', 'de essa', 'de ele', 'de aqueles', 'a aquele', 'de isso', 'de isto', 'em aquela', 'de aquilo', 'de elas', 'em ele', 'em ela', 'em essa', 'em estas', 'em essas', 'em outras', 'em umas', 'em elas', 'de estas', 'em alguns', 'de antes', 'de aqui', 'em estes', 'com vós', 'em esses', 'de aquelas', 'com si', 'de um', 'em outro', 'de esses', 'de os', 'em eles', 'em algumas', 'em isso', 'em outra', 'em aqueles', 'em outros', 'com mim', 'de ali', 'lhe a', 'a aqueles', 'a aquela', 'em isto', 'em aquilo']

	if len(dettached) == len(contractions):
		for i, contraction in enumerate(contractions):
			sentenca = re.sub(r'\b' + contraction + r'\b', dettached[i], sentenca, flags=re.IGNORECASE)
	else:
		print('\nNúmero de contrações é diferente do número de desmembramentos.')
		exit()

	for ponto in string.punctuation + "«»":
		sentenca = sentenca.replace(ponto, ' ' + ponto + ' ')

	print('\nSentença tokenizada:\n--------------------')
	print(sentenca)

	return sentenca.replace('\n',' <space> ').split()

def montar_tuplas(sentenca, n = 3):

	tuplas = copy.deepcopy(sentenca)
	for i, token in enumerate(sentenca):
		inicio = list()
		meio = list()
		fim = list()
		for b in range(int((n-1)/2 - i)):
			inicio.append('@')
		for b in range(int((n-1)/2 - (len(sentenca) - 1 - i))):
			fim.append('/')

		meio.append(sentenca[i])

		if i + 1 <= int((n-1)/2):
			try:
				for b in range(i):
					meio.insert(0, sentenca[i-b-1])
				for b in range(int((n-1)/2)):
					meio.append(sentenca[i+b+1])
			except: pass

		elif i + 1 > len(sentenca) - int((n-1)/2):
			try:
				for b in range(int((n-1)/2)):
					meio.insert(0, sentenca[i-b-1])
				for b in range(int((n-1)/2) - (len(sentenca) - (i+2))):
					meio.append(sentenca[i+b+1])
			except: pass

		#nem fim nem começo
		else:
			try:
				for b in range(int((n-1)/2)):
					meio.append(sentenca[i+b+1])
					meio.insert(0, sentenca[i-b-1])
			except: pass

		tuplas[i] = inicio + meio + fim

	return tuplas

def coletar_material(conllu):

	#remove metadados
	for a, sentenca in enumerate(conllu):
		conllu[a] = ["{}\t{}".format(x[1], x[3]) for x in sentenca if not '#' in x and x]

	#cria as triplas cruas
	cru = [[token.split('\t')[0] for token in sentenca] for sentenca in conllu]
	cru_solto = list()
	for sentence in cru:
		cru_solto.extend(sentence)
	palavras = montar_tuplas(cru_solto)

	#cria os rótulos para cada tripla
	rotulos = [[token.split('\t')[1] for token in sentenca] for sentenca in conllu]
	rotulos = [[classe] for classe in estrutura_dados.PrintarUD(rotulos).split('\n') if classe]

	return {"features": palavras, "labels": rotulos, "soltas": cru_solto}

def treinar(material_treino):

	conllu = estrutura_dados.LerUD(material_treino)
	material = coletar_material(conllu)

	pal = preprocessing.OneHotEncoder(handle_unknown="ignore").fit(material["features"])
	rot = preprocessing.LabelEncoder().fit(material['labels'])

	material['features'] = pal.transform(material['features'])
	material['labels'] = rot.transform(material['labels'])

	classifier = tree.DecisionTreeClassifier().fit(material["features"], material["labels"])
	acuracia = 'Acurácia: ' + str(classifier.score(material["features"], material["labels"]) * 100) + '%'
	print(acuracia)
	open(material_treino.rsplit('.', 1)[0] + '.joblib4', 'w').write(acuracia)

	with open(material_treino.rsplit('.', 1)[0] + '.joblib', 'wb') as f:
		pickle.dump(classifier, f)

	with open(material_treino.rsplit('.', 1)[0] + '.joblib2', 'wb') as f:
		pickle.dump(pal, f)

	with open(material_treino.rsplit('.', 1)[0] + '.joblib3', 'wb') as f:
		pickle.dump(rot, f)


def main(sentenca, modo = "bosque2.3_golden_train.joblib"):

	sentenca = tokenizar(sentenca)
	tripla_sentenca = montar_tuplas(sentenca)

	with open(modo, 'rb') as f:
		classifier = pickle.load(f)

	with open(modo + '2', 'rb') as f:
		pal = pickle.load(f)

	with open(modo + '3', 'rb') as f:
		rot = pickle.load(f)

	for i, tripla in enumerate(tripla_sentenca):
		predicao = classifier.predict(pal.transform([tripla]))
		proba = classifier.predict_proba(pal.transform([tripla]))
		inverso = rot.inverse_transform(predicao)

		sentenca[i] = f"{sentenca[i]}_{inverso[0]}"

	sentenca = " ".join(sentenca)

	print('\n\nAnotação:\n---------')
	print(re.sub(r'\<space\>.*?\s', '\n', sentenca))

if __name__ == "__main__":
	if len(sys.argv) == 1:
		main(input('Sentença: '))
	elif sys.argv[1] != '--train':
		main(open(sys.argv[1].replace("'", "").replace('"', '').replace("\\", '/').strip(), 'r').read(), sys.argv[2] if 2 < len(sys.argv) else "bosque2.3_golden_train.joblib")
	else:
		treinar(input("Material de treino: ").replace("'", "").replace('"', '').replace("\\", '/').strip())


