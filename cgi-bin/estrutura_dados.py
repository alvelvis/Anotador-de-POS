# -*- coding: utf-8 -*-

'''
Conjunto de funções para lidar com arquivos UD (.conllu)
Objetivo: ser importado em outros códigos do pacote ACDC-UD: "import estrutura_dados"
'''

#Abre o arquivo UD em um caminho dado ("ud_file"), segmenta suas sentenças, suas linhas, e cada uma das 10 colunas dos seus tokens
def LerUD(ud_file):

	#Determina a codificação do arquivo caso o usuário forneça a codificação (depois de dois pontos)
	#Exemplo: "meu_arquivo.conllu:utf8"
	if ':' in ud_file: codification = ud_file.split(':')[1]
	#Caso o usuário não forneça dois pontos, a codificação padrão é "utf8"
	else: codification = 'utf8'
	ud_file = ud_file.split(':')[0] #O nome do arquivo é o que vem antes de dois pontos (se não tiver dois pontos, tudo bem, ele não dá erro)

	#Abre o arquivo no caminho fornecido e segmenta as sentenças
	arquivo = open(ud_file, 'r').read().split('\n\n')
	#Agora que "arquivo" é uma lista de sentenças, apagar os itens que estiverem vazios (no caso, por exemplo, de linhas em excesso no final do arquivo, resultando em splits("\n\n") que não são de fato sentenças)
	arquivo = [x for x in arquivo if x]

	#Para cada sentença na lista de sentenças,
	for a, sentence in enumerate(arquivo):
		#Essa sentença vai ter suas linhas segmentadas
		arquivo[a] = arquivo[a].splitlines() #Repare que linha pode ser tanto os tokens quanto os metadados (linhas que começam com "# text = ", por exemplo)
		#Para cada linha já segmentada nessa dada sentença (arquivo[a]),
		for b, linha in enumerate(arquivo[a]):
			#Se a quantidade de colunas for 10, (ou seja, garante que estou lidando com tokens, e não com metadados)
			if '\t' in linha:
				#Essa linha, que na verdade é um token, (sentença arquivo[a], token[b]), vai ser dividida em colunas
				arquivo[a][b] = arquivo[a][b].split('\t')

	#Retorna uma lista com sentenças, sendo cada sentença uma lista de linhas, e cada lista de linhas, se tiver 10 colunas, uma lista de 10 colunas
	return arquivo


#Depois de feitas as alterações em um arquivo UD (variável "UD"), essa função retorna o conjunto de listas para o formato "string" normal para ser salvo em um arquivo (variável "arquivo")
def EscreverUD(UD, arquivo):

	#Determina a codificação do arquivo caso o usuário forneça a codificação (depois de dois pontos)
	#Exemplo: "meu_arquivo.conllu:utf8"
	if ':' in arquivo: codification = arquivo.split(':')[1]
	#Caso o usuário não forneça dois pontos, a codificação padrão é "utf8"
	else: codification = "utf8"
	arquivo = arquivo.split(':')[0] #O nome do arquivo é o que vem antes de dois pontos (se não tiver dois pontos, tudo bem, ele não dá erro)

	#Para cada sentença na lista "UD",
	for a, sentence in enumerate(UD):
		#Para cada linha dentro dessa sentença,
		for b, linha in enumerate(UD[a]):
			#Se for um token (tiver 10 colunas) e não for um metadado (contiver "# "),
			if isinstance(linha, list) and not '# ' in linha:
				#Reunir as colunas adicionando um "\t"
				UD[a][b] = "\t".join(UD[a][b])
		#Depois de reunir todos os tokens que devem ser reunidos, reunir todas as linhas com um "\n"
		UD[a] = "\n".join(UD[a])
	#Reunidas todas as linhas, reunir as sentenças
	UD = "\n\n".join(UD) + '\n\n' #O "\n" no final é necessário pois todo arquivo UD deve ter uma linha vazia no final, é uma exigência dos códigos de comparação, avaliação, etc.

	#Salvar :)
	open(arquivo, 'w', encoding=codification).write(UD)

#Depois de feitas as alterações em um arquivo UD (variável "UD"), essa função retorna o conjunto de listas para o formato "string" normal para ser salvo em um arquivo (variável "arquivo")
def PrintarUD(UD):

	#Para cada sentença na lista "UD",
	for a, sentence in enumerate(UD):
		#Para cada linha dentro dessa sentença,
		for b, linha in enumerate(UD[a]):
			#Se for um token (tiver 10 colunas) e não for um metadado (contiver "# "),
			if isinstance(linha, list) and not '# ' in linha:
				#Reunir as colunas adicionando um "\t"
				UD[a][b] = "\t".join(UD[a][b])
			#Depois de reunir todos os tokens que devem ser reunidos, reunir todas as linhas com um "\n"
		UD[a] = "\n".join(UD[a])
	#Reunidas todas as linhas, reunir as sentenças
	UD = "\n\n".join(UD) + '\n' #O "\n" no final é necessário pois todo arquivo UD deve ter uma linha vazia no final, é uma exigência dos códigos de comparação, avaliação, etc.

	#Salvar :)
	return UD

#Transforma acentos e caracteres especiais em underlines
def slugify(value):
	return "".join(x if x.isalnum() or x == '.' else "_" for x in value)
