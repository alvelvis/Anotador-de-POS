import estrutura_dados
from sklearn import preprocessing
from sklearn import tree
import copy
import string
import re

def tokenizar(sentenca):

	contractions = ['da', 'das', 'do', 'dos', 'dela', 'nos', 'ao', 'na', 'aos', 'no', 'às', 'nesse', 'pelos', 'numa', 'à', 'nas', 'pela', 'pelo', 'daí', 'deste', 'neste', 'desta', 'num', 'naquele', 'daquela', 'connosco', 'destes', 'dessas', 'pelas', 'duma', 'daquele', 'deles', 'nesta', 'desse', 'dessa', 'dele', 'daqueles', 'àquele', 'disso', 'disto', 'naquela', 'daquilo', 'delas', 'nele', 'nela', 'nessa', 'nestas', 'nessas', 'noutras', 'numas', 'nelas', 'destas', 'nalguns', 'dantes', 'daqui', 'nestes', 'convosco', 'nesses', 'daquelas', 'consigo', 'dum', 'noutro', 'desses', 'do(s)', 'neles', 'nalgumas', 'nisso', 'noutra', 'naqueles', 'noutros', 'comigo', 'dali', 'lha', 'àqueles', 'àquela', 'nisto', 'naquilo']

	dettached = ['de a', 'de as', 'de o', 'de os', 'de ela', 'em os', 'a o', 'em a', 'a os', 'em o', 'a as', 'em esse', 'por os', 'em uma', 'a a', 'em as', 'por a', 'por o', 'de aí', 'de este', 'em este', 'de esta', 'em um', 'em aquele', 'de aquela', 'com nós', 'de estes', 'de essas', 'por elas', 'de uma', 'de aquele', 'de eles', 'em esta', 'de esse', 'de essa', 'de ele', 'de aqueles', 'a aquele', 'de isso', 'de isto', 'em aquela', 'de aquilo', 'de elas', 'em ele', 'em ela', 'em essa', 'em estas', 'em essas', 'em outras', 'em umas', 'em elas', 'de estas', 'em alguns', 'de antes', 'de aqui', 'em estes', 'com vós', 'em esses', 'de aquelas', 'com si', 'de um', 'em outro', 'de esses', 'de os', 'em eles', 'em algumas', 'em isso', 'em outra', 'em aqueles', 'em outros', 'com mim', 'de ali', 'lhe a', 'a aqueles', 'a aquela', 'em isto', 'em aquilo']

	if len(dettached) == len(contractions):
		for i, contraction in enumerate(contractions):
			sentenca = re.sub(r'\b' + contraction + r'\b', dettached[i], sentenca, flags=re.IGNORECASE)
	else:
		print('Número de contrações é diferente do número de transformações.')
		exit()

	for ponto in string.punctuation:
		sentenca = sentenca.replace(ponto, ' ' + ponto + ' ')

	print('Contrações desfeitas:')
	print(sentenca)

	return sentenca.split()

def coletar_material(conllu, sentence):

	#remove metadados
	for a, sentenca in enumerate(conllu):
		conllu[a] = ["{}\t{}".format(x[1], x[3]) for x in sentenca if not '#' in x and x]

	#cria as triplas cruas
	cru = [[token.split('\t')[0] for token in sentenca] for sentenca in conllu]
	palavras = copy.deepcopy(cru)
	for a, sentenca in enumerate(palavras):
		for i, token in enumerate(sentenca):
			if i == 0 and i + 1 < len(sentenca):
				palavras[a][i] = ['@', cru[a][i], cru[a][i+1]]
			elif i == 0 and i + 1 == len(sentenca):
				palavras[a][i] = ['@', cru[a][i], '/']
			elif i > 0 and i + 1 < len(sentenca):
				palavras[a][i] = [cru[a][i-1], cru[a][i], cru[a][i+1]]
			elif i > 0 and i + 1 == len(sentenca):
				palavras[a][i] = [cru[a][i-1], cru[a][i], '/']
			else:
				print(token, sentenca)
				exit()
	palavras = [tripla.split('\t') for tripla in estrutura_dados.PrintarUD(palavras).split('\n') if tripla]

	#cria os rótulos para cada tripla
	rotulos = [[token.split('\t')[1] for token in sentenca] for sentenca in conllu]
	rotulos = [[classe] for classe in estrutura_dados.PrintarUD(rotulos).split('\n') if classe]

	#verifica palavras na sentença que não têm no dataset
	print('Palavras não encontradas no dataset (recebendo POS "X"):')
	for palavra in sentence:
		if not any(palavra in x for x in palavras):
			palavras.append([palavra, palavra, palavra])
			rotulos.append(['X'])
			print(palavra)

	return {"features": palavras, "labels": rotulos}

def main(sentenca):

	sentenca = tokenizar(sentenca)

	conllu = estrutura_dados.LerUD('bosque2.3_golden_teste.conllu')
	material = coletar_material(conllu, sentenca)

	pal = preprocessing.OneHotEncoder()
	rot = preprocessing.OneHotEncoder()

	classifier = tree.DecisionTreeClassifier()

	tripla_sentenca = copy.deepcopy(sentenca)
	for i, token in enumerate(sentenca):
		if i == 0 and i + 1 < len(sentenca):
			tripla_sentenca[i] = ['@', sentenca[i], sentenca[i+1]]
		elif i == 0 and i + 1 == len(sentenca):
			tripla_sentenca[i] = ['@', sentenca[i], '/']
		elif i > 0 and i + 1 < len(sentenca):
			tripla_sentenca[i] = [sentenca[i-1], sentenca[i], sentenca[i+1]]
		elif i > 0 and i + 1 == len(sentenca):
			tripla_sentenca[i] = [sentenca[i-1], sentenca[i], '/']
		else:
			print(token, sentenca)
			exit()

	for i, tripla in enumerate(tripla_sentenca):
		predicao = classifier.fit(pal.fit_transform(material["features"]).toarray(), rot.fit_transform(material["labels"]).toarray()).predict(pal.fit(material["features"]).transform([tripla]).toarray())
		inverso = rot.inverse_transform(predicao)
		print(sentenca[i], inverso, predicao)
		sentenca[i] = "{}_{}".format(sentenca[i], inverso[0][0])

	sentenca = "\n".join(sentenca)

	print('')
	print(sentenca)

if __name__ == "__main__":
	main(input('Sentença: '))


