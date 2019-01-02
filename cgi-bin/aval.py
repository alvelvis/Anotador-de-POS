import estrutura_dados
import string

with open(input('Arquivo anotado pelo sistema: ').replace('"', '').replace("'", "").replace('\\', '/').strip(), 'r') as f:
	anotado = f.read()
	algoritmo = anotado.split('\n---------\n')[1].replace('\n', ' ').split()

conllu = estrutura_dados.LerUD(input('Arquivo golden: ').replace('"', '').replace("'", "").replace('\\', '/').strip())

conllu_normalizado = [["{}_{}".format(token[1], token[3]) for token in sentenca if not '#' in token and token and token[3] != '_'] for sentenca in conllu]

golden = list()
for sentenca in conllu_normalizado:
	for token in sentenca:
		golden.append(token)

contractions = ['da', 'das', 'do', 'dos', 'dela', 'nos', 'ao', 'na', 'aos', 'no', 'às', 'nesse', 'pelos', 'numa', 'à', 'nas', 'pela', 'pelo', 'daí', 'deste', 'neste', 'desta', 'num', 'naquele', 'daquela', 'connosco', 'destes', 'dessas', 'pelas', 'duma', 'daquele', 'deles', 'nesta', 'desse', 'dessa', 'dele', 'daqueles', 'àquele', 'disso', 'disto', 'naquela', 'daquilo', 'delas', 'nele', 'nela', 'nessa', 'nestas', 'nessas', 'noutras', 'numas', 'nelas', 'destas', 'nalguns', 'dantes', 'daqui', 'nestes', 'convosco', 'nesses', 'daquelas', 'consigo', 'dum', 'noutro', 'desses', 'do(s)', 'neles', 'nalgumas', 'nisso', 'noutra', 'naqueles', 'noutros', 'comigo', 'dali', 'lha', 'àqueles', 'àquela', 'nisto', 'naquilo']

golden_tratado = list()
for token in golden:
	for ponto in string.punctuation + "«»":
		if (ponto in token.split('_')[0] and token.split('_')[0] != ponto) or token.split('_')[0].lower() in contractions:
			break
	else:
		golden_tratado.append(token)

acertos = 0
palavras = 0
n = 0
for i in range(len(golden_tratado)):
	if algoritmo[n].split('_')[0].lower() == golden_tratado[i].split('_')[0].lower():
		palavras += 1
		if algoritmo[n].split('_')[1] == golden_tratado[i].split('_')[1]:
			acertos += 1
		n += 1
	elif i+2 < len(golden_tratado):
		while algoritmo[n].split('_')[0].lower() != golden_tratado[i].split('_')[0].lower():
			n += 1
		palavras += 1
		if algoritmo[n].split('_')[1] == golden_tratado[i].split('_')[1]:
			acertos += 1
		n += 1

print('\nResultados da avaliação\n')
print('Tokenização\n-----------')
print('Tokens no golden:', len(golden))
print('Tokens no sistema:', len(algoritmo))
print('Precisão:', str(len(golden)/len(algoritmo)*100) + '%')
print('\nAnotação de POS\n---------------')
print('Tokens iguais:', palavras)
print('Acertos de POS:', acertos)
print('Precisão:', str((acertos/palavras)*100) + '%')

