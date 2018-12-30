# Anotador de POS

Primeiro experimento com Machine Learning: um anotador de classes gramaticais treinado a partir dos trigramas das palavras no material *golden*.

* [Resultados](#Resultados)
	* [Avaliação de desempenho](#Avaliação_de_desempenho)
* [Funcionamento](#Funcionamento)
	* [Bibliotecas](#Bibliotecas)
	* [Passo a passo](#Passo_a_passo)
		* [Entrada](#Entrada)
		* [tokenizar()](#tokenizar)
		* [coletar_material()](#coletar_material)
		* [main()](#main)
	* [OneHotEncoder](#OneHotEnconder)

# Resultados

Observe a seguinte sentença, retirada do corpus [Bosque](https://github.com/UniversalDependencies/UD_Portuguese-Bosque), partição de *treino*:

>Talvez entusiasmado pela festa da vitória, o Presidente russo afirmou que «chegará o dia em que a Rússia ajudará o Ocidente».

No material *golden*, a sentença está anotada da seguinte forma, dispensando-se outras informação que não as de POS:

	Talvez	ADV
	entusiasmado	VERB
	pela	_
	por	ADP
	a	DET
	festa	NOUN
	da	_
	de	ADP
	a	DET
	vitória	NOUN
	,	PUNCT
	o	DET
	Presidente	NOUN
	russo	ADJ
	afirmou	VERB
	que	SCONJ
	«	PUNCT
	chegará	VERB
	o	DET
	dia	NOUN
	em	ADP
	que	PRON
	a	DET
	Rússia	PROPN
	ajudará	VERB
	o	DET
	Ocidente	NOUN
	»	PUNCT
	.	PUNCT

A mesma sentença, anotada pelo algoritmo aqui descrito e treinado na partição *dev* do [Bosque](https://github.com/UniversalDependencies/UD_Portuguese-Bosque), apresentou a seguinte anotação de POS:

>Talvez_PROPN entusiasmado_VERB por_ADP a_DET festa_NOUN de_ADP a_DET vitória_NOUN ,_PUNCT o_DET Presidente_NOUN russo_PROPN afirmou_VERB que_PRON «_PUNCT chegará_VERB o_DET dia_PROPN em_ADP que_PRON a_DET Rússia_PROPN ajudará_VERB o_DET Ocidente_PROPN »_PUNCT ._PUNCT


Temos divergências nos tokens:

1) Talvez (ADV x PROPN): erro do algoritmo, provavelmente pela frequência de nomes de pessoas com inicial maiúscula que iniciam sentenças;
2) russo (ADJ x PROPN): erro do algoritmo, provavelmente influenciado pela quantidade de nomes de pessoas que acompanham a palavra "Presidente" no corpus;
3) Ocidente (NOUN x PROPN): ambas as anotações são possíveis.

Trata-se de um resultado similar, principalmente ao se levar em conta que a sentença anotada (retirada da partição "train" do Bosque) não figurava no material em que o algoritmo foi treinado (partição "dev").

## Avaliação de desempenho

O desempenho do anotador de POS foi avaliado comparando a anotação que ele faz das sentenças na partição "teste" do Bosque com o seu golden.

Primeiro, foi utilizado o script **limpar_conllu.py** para, a partir do arquivo **bosque2.3_golden_teste.conllu**, gerar as sentenças cruas, no arquivo **teste_limpo.txt**.

Depois, o anotador de POS foi chamado para anotar o arquivo de sentenças cruas:

	$ python3 tagger.py teste_limpo.txt

O resultado consta no arquivo **teste_anotado.txt**.

Então, a partir do script **aval.py**, calculou-se a precisão das anotações ao comparar-se o "teste" anotado pelo algoritmo e o "teste" anotado por humanos.

Como resultado, conquistamos uma precisão de %.

# Funcionamento

O anotador de POS baseia suas predições em uma Árvore de Decisões da biblioteca Scikit-Learn, em Python 3.

O material de treino do anotador é sempre um arquivo *.conllu* no formato Universal Dependencies. O código processa o arquivo conllu de tal maneira que cada unidade de treino, chamada "feature", é uma tripla, ou mais especificamente, um trigrama. Ou seja, cada palavra que alimenta o algoritmo, além de estar acompanhada da sua classe de palavra anotada no golden, tem também as duas palavras vizinhas que a acompanham. Assim, o anotador de POS, ao realizar suas tarefas, sempre prestará atenção ao contexto dos tokens.

No caso de palavras que iniciam e terminam frases, são utilizados, respectivamente, os símbolos "@" e "/" para preencher as vagas dos trigramas.

## Bibliotecas

**tagger.py** utiliza as seguintes bibliotecas:

```python
import estrutura_dados
from sklearn import preprocessing
from sklearn import tree
import copy
import string
import re
import sys
```

**estrutura_dados** é um script utilizado no projeto [ACDC-UD](http://github.com/alvelvis/ACDC-UD) para ler, escrever e printar arquivos UD;

**sklearn** é a biblioteca utilizada para transformar as palavras em códigos binários e montar a Árvore de Decisões, base do algoritmo de predição;

**copy** é utilizado para copiar as listas com os trigramas;

**string**.punctuation é o conjunto de caracteres que devemos destacar das sentenças na tokenização. Além dessa pontuação, adicionamos os símbolos "«" e "»";

**re** é utilizado na hora de destacar as contrações em Língua Portuguesa;

**sys** é a biblioteca responsável por ler a linha de comando que passamos ao programa no terminal.

## Passo a passo

### Entrada

A primeira parte do **tagger.py** é a responsável por verificar se ele anotará um dado arquivo descrito na linha de comando do terminal ou se o usuário será requisitado para digitar a sentença que deseja que seja anotada:

```python
if __name__ == "__main__":
	if len(sys.argv) == 1:
		main(input('Sentença: '))
	else:
		main(open(sys.argv[1].replace("'", "").replace('"', '').replace("\\", '/').strip(), 'r').read())
```

Ambas as opções levam à função "main()":

```python
def main(sentenca):

	sentenca = tokenizar(sentenca)

	conllu = estrutura_dados.LerUD('bosque2.3_golden_dev.conllu')
	material = coletar_material(conllu, sentenca)

	...
```

### tokenizar()

A segunda etapa é tokenizar a entrada do usuário, seja o arquivo ou a sentença.

As duas listas a seguir foram mineradas do arquivo **bosque2.3_golden_train_dev_e_teste.conllu** ao se procurar por palavras cujas POS fossem *underline* ("\_"). Ou seja, procuramos por palavras não anotadas que, portanto, eram contrações:

```python
def tokenizar(sentenca):

	contractions = ['da', 'das', 'do', 'dos', 'dela', 'nos', 'ao', 'na', 'aos', 'no', 'às', 'nesse', 'pelos', 'numa', 'à', 'nas', 'pela', 'pelo', 'daí', 'deste', 'neste', 'desta', 'num', 'naquele', 'daquela', 'connosco', 'destes', 'dessas', 'pelas', 'duma', 'daquele', 'deles', 'nesta', 'desse', 'dessa', 'dele', 'daqueles', 'àquele', 'disso', 'disto', 'naquela', 'daquilo', 'delas', 'nele', 'nela', 'nessa', 'nestas', 'nessas', 'noutras', 'numas', 'nelas', 'destas', 'nalguns', 'dantes', 'daqui', 'nestes', 'convosco', 'nesses', 'daquelas', 'consigo', 'dum', 'noutro', 'desses', 'do(s)', 'neles', 'nalgumas', 'nisso', 'noutra', 'naqueles', 'noutros', 'comigo', 'dali', 'lha', 'àqueles', 'àquela', 'nisto', 'naquilo']

	dettached = ['de a', 'de as', 'de o', 'de os', 'de ela', 'em os', 'a o', 'em a', 'a os', 'em o', 'a as', 'em esse', 'por os', 'em uma', 'a a', 'em as', 'por a', 'por o', 'de aí', 'de este', 'em este', 'de esta', 'em um', 'em aquele', 'de aquela', 'com nós', 'de estes', 'de essas', 'por elas', 'de uma', 'de aquele', 'de eles', 'em esta', 'de esse', 'de essa', 'de ele', 'de aqueles', 'a aquele', 'de isso', 'de isto', 'em aquela', 'de aquilo', 'de elas', 'em ele', 'em ela', 'em essa', 'em estas', 'em essas', 'em outras', 'em umas', 'em elas', 'de estas', 'em alguns', 'de antes', 'de aqui', 'em estes', 'com vós', 'em esses', 'de aquelas', 'com si', 'de um', 'em outro', 'de esses', 'de os', 'em eles', 'em algumas', 'em isso', 'em outra', 'em aqueles', 'em outros', 'com mim', 'de ali', 'lhe a', 'a aqueles', 'a aquela', 'em isto', 'em aquilo']

	...
```

A primeira lista foi o resultado direto da extração do arquivo golden. A segunda lista foi obtida a partir do desmembramento manual das contrações da primeira lista.

Então, ainda dentro da função "tokenizar(sentenca)", é feito o desmembramento das palavras contidas na entrada:

```python
def tokenizar(sentenca):

	...

	if len(dettached) == len(contractions):
		for i, contraction in enumerate(contractions):
			sentenca = re.sub(r'\b' + contraction + r'\b', dettached[i], sentenca, flags=re.IGNORECASE)
	else:
		print('\nNúmero de contrações é diferente do número de desmembramentos.')
		exit()

	...
```

A seguir é feito o destacamento das pontuações (inserindo-se, aqui, os hífens, portanto desmembrando também as palavras compostas, pronomes oblíquos etc.), retornando uma lista em que cada token é um ítem:

```python
def tokenizar(sentenca):

	...

	for ponto in string.punctuation + "«»":
		sentenca = sentenca.replace(ponto, ' ' + ponto + ' ')

	print('\nContrações desfeitas:')
	print(sentenca)

	return sentenca.split()

	...
```

### coletar_material()

Após a tokenização da entrada, a função "main()" chama a função "coletar_material()", responsável por "treinar" o algoritmo.

Repare que o treino tem sido feito na partição "dev" do Bosque, uma vez que, em materiais maiores, o sistema tem se tornado extremamente pesado e acarretado erros de memória.

A primeira parte da função é a responsável por retirar os metadados e as anotações do material de treino, deixando apenas as palavras (coluna 2) e suas POS (coluna 4):

```python
def coletar_material(conllu):

	#remove metadados
	for a, sentenca in enumerate(conllu):
		conllu[a] = ["{}\t{}".format(x[1], x[3]) for x in sentenca if not '#' in x and x]

	...
```

Então, é hora de criar os trigramas, não anotados. Este constituirá o material "features" que alimentará o sistema de predição:

```python
def coletar_material(conllu):

	...

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

	...
```

Depois, criamos a lista de "rótulos" (labels) que se encaixam nas "features", e retornamos um dicionário com ambas as listas para alimentar o preditor:

```python
def coletar_material(conllu):

	...

	#cria os rótulos para cada tripla
	rotulos = [[token.split('\t')[1] for token in sentenca] for sentenca in conllu]
	rotulos = [[classe] for classe in estrutura_dados.PrintarUD(rotulos).split('\n') if classe]

	return {"features": palavras, "labels": rotulos}
```

### main()

Voltando à função principal, já tendo tokenizado a entrada do programa e coletado o material de treino, é hora de preparar os trigramas do material de entrada e de, finalmente, montar a árvore de decisões.

Aqui, preparamos os trigramas da entrada do programa, de forma análoga à que usamos para montar os trigramas do material de treino:

```python
def main(sentenca):

	...

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

	...
```

Então, criamos um [OneHotEncoder](#OneHotEncoder) para cada lista (features e labels) e fazemos a predição:

```python
def main(sentenca):

	...

	pal = preprocessing.OneHotEncoder(handle_unknown="ignore")
	rot = preprocessing.OneHotEncoder(handle_unknown="ignore")

	classifier = tree.DecisionTreeClassifier()

	for i, tripla in enumerate(tripla_sentenca):
		predicao = classifier.fit(pal.fit_transform(material["features"]).toarray(), rot.fit_transform(material["labels"]).toarray()).predict(pal.fit(material["features"]).transform([tripla]).toarray())
		inverso = rot.inverse_transform(predicao)
		print(sentenca[i], inverso, predicao)
		sentenca[i] = "{}_{}".format(sentenca[i], inverso[0][0])

	...
```

No fim, juntamos as palavras com um espaço entre elas e mandamos para o terminal:

```python
def main(sentenca):

	...

	sentenca = " ".join(sentenca)

	print('')
	print(sentenca)
```

Fim do algoritmo.

## OneHotEncoder

O OneHotEncoder é necessários pois estamos lidando com *strings* tanto em relação às palavras que queremos anotar quanto aos rótulos das anotações.

O OneHotEncoder é o responsável por transformar as strings em matrizes de códigos binários, de forma que todas as palavras sejam tratadas como "independentes" e não se confundam entre si.

Observe as matrizes que representam cada uma das palavras a seguir, transformadas pelo OneHotEncoder (processo que, posteriormente, é revertido para que as palavras sejam legíveis para o usuário):

	[['NOUN']] [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
	[['VERB']] [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
	[['PROPN']] [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]