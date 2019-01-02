# Anotador de POS

Primeiro experimento com Machine Learning: um anotador de classes gramaticais treinado a partir dos trigramas das palavras no material *golden*.

O Anotador de POS pode ser testado on-line pelo endereço: http://anotador-de-pos.tronco.me/

* [Resultados](#Resultados)
	* [Avaliação de desempenho](#Avaliação-de-desempenho)
* [Funcionamento](#Funcionamento)
	* [Bibliotecas](#Bibliotecas)
	* [Passo a passo](#Passo-a-passo)
		* [Entrada](#Entrada)
		* [tokenizar()](#tokenizar)
		* [coletar_material()](#coletar_material)
		* [main()](#main)
	* [LabelEncoder](#LabelEncoder)

# Resultados

Observe a seguinte sentença, retirada do corpus [Bosque](https://github.com/UniversalDependencies/UD_Portuguese-Bosque), partição de teste:

>Talvez entusiasmado pela festa da vitória, o Presidente russo afirmou que «chegará o dia em que a Rússia ajudará o Ocidente».

No material *golden*, a sentença está anotada da seguinte forma:

	# text = Gosto de levar a sério o meu papel de consultor encartado.
	# source = CETEMPúblico n=984 sec=nd sem=94b
	# sent_id = CP984-4
	# id = 5058
	1	Gosto	gostar	VERB	_	Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin	0	root	_	_
	2	de	de	ADP	_	_	3	mark	_	_
	3	levar	levar	VERB	_	VerbForm=Inf	1	xcomp	_	_
	4	a	a	ADP	_	_	5	case	_	MWE=a_sério
	5	sério	sério	NOUN	_	_	3	xcomp	_	_
	6	o	o	DET	_	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	8	det	_	_
	7	meu	meu	DET	_	Gender=Masc|Number=Sing|PronType=Prs	8	det	_	_
	8	papel	papel	NOUN	_	Gender=Masc|Number=Sing	3	obj	_	_
	9	de	de	ADP	_	_	10	case	_	_
	10	consultor	consultor	NOUN	_	Gender=Masc|Number=Sing	8	nmod	_	_
	11	encartado	encartar	VERB	_	Gender=Masc|Number=Sing|VerbForm=Part	10	acl	_	SpaceAfter=No
	12	.	.	PUNCT	_	_	1	punct	_	_

A mesma sentença, anotada pelo algoritmo aqui descrito e treinado na partição *train* do [Bosque](https://github.com/UniversalDependencies/UD_Portuguese-Bosque), apresentou a seguinte anotação de POS:

>Gosto_NOUN de_ADP levar_VERB a_DET sério_ADJ o_DET meu_DET papel_NOUN de_ADP consultor_NOUN encartado_VERB ._PUNCT

Observamos divergênicas nos seguintes tokens:

1. Gosto: NOUN (sistema) - VERB (golden)
2. sério: ADJ (sistema) - NOUN (golden)

Trata-se de um resultado animador, principalmente ao se levar em conta que a sentença anotada (retirada da partição "teste" do Bosque) não figurava no material em que o algoritmo foi treinado (partição "train").

## Avaliação de desempenho

O desempenho do anotador de POS foi avaliado comparando a anotação que ele faz das sentenças na partição "teste" do Bosque com o seu golden.

Primeiro, foi utilizado o script **limpar_conllu.py** para, a partir do arquivo **bosque2.3_golden_teste.conllu**, gerar as sentenças cruas, no arquivo **teste_limpo.txt**.

Depois, o anotador de POS foi chamado para anotar o arquivo de sentenças cruas:

	$ python3 tagger.py ../teste_limpo.txt > ../teste_anotado.txt

O resultado consta no arquivo **teste_anotado.txt**.

Então, a partir do script **aval.py**, calculou-se a precisão das tokenizações e anotações ao comparar-se o "teste" anotado pelo algoritmo e o "teste" anotado por humanos.

Os resultados são os seguintes:

	Arquivo anotado pelo sistema: ../teste_anotado.txt
	Arquivo golden: bosque2.3_golden_teste.conllu

	Resultados da avaliação

	Tokenização
	-----------
	Tokens no golden: 10199
	Tokens no sistema: 10517
	Precisão: 96.9763240467814%

	Anotação de POS
	---------------
	Tokens iguais: 10032
	Acertos de POS: 8372
	Precisão: 83.45295055821371%


[**↥ voltar ao topo**](#anotador-de-pos)

# Funcionamento

O anotador de POS baseia suas predições em uma Árvore de Decisões da biblioteca Scikit-Learn, em Python 3.

O material de treino do anotador é sempre um arquivo *.conllu* no formato Universal Dependencies. O código processa o arquivo conllu de tal maneira que cada unidade de treino, chamada "feature", é uma tripla, ou mais especificamente, um trigrama. Ou seja, cada palavra que alimenta o algoritmo, além de estar acompanhada da sua classe gramatical anotada no golden, tem também as duas palavras vizinhas que a acompanham. Assim, o anotador de POS, ao realizar suas tarefas, sempre prestará atenção ao contexto dos tokens.

No caso de palavras que iniciam e terminam frases, são utilizados, respectivamente, os símbolos "@" e "/" para preencher as vagas dos trigramas.

## Bibliotecas

**tagger.py** utiliza as seguintes bibliotecas:

```python
import estrutura_dados
from sklearn import preprocessing
from sklearn import tree
import pickle as jb
import copy
import string
import re
import sys
```

**estrutura_dados** é um script utilizado no projeto [ACDC-UD](http://github.com/alvelvis/ACDC-UD) para ler, escrever e printar arquivos UD;

**sklearn** é a biblioteca utilizada para transformar as palavras em códigos binários e montar a Árvore de Decisões, base do algoritmo de predição;

**pickle** é o módulo responsável por salvar o modelo treinado em um arquivo ".joblib";

**copy** é utilizado para copiar as listas com os trigramas;

**string**.punctuation é o conjunto de caracteres que devemos destacar das sentenças na tokenização. Além dessa pontuação, adicionamos os símbolos "«" e "»";

**re** é utilizado na hora de destacar as contrações em Língua Portuguesa;

**sys** é a biblioteca responsável por ler a linha de comando que passamos ao programa no terminal.

[**↥ voltar ao topo**](#anotador-de-pos)

## Passo a passo

### Entrada

A primeira parte do **tagger.py** é a responsável por verificar se ele anotará um dado arquivo descrito na linha de comando do terminal, ou se o usuário será requisitado para digitar a sentença que deseja que seja anotada. Verifica-se também a presença do parâmetro "--train":

```python
if __name__ == "__main__":
	if len(sys.argv) == 1:
		main(input('Sentença: '))
	elif sys.argv[1] != '--train':
		main(open(sys.argv[1].replace("'", "").replace('"', '').replace("\\", '/').strip(), 'r').read(), sys.argv[2] if 2 < len(sys.argv) else "bosque2.3_golden_train.joblib")
	else:
		treinar(input("Material de treino: ").replace("'", "").replace('"', '').replace("\\", '/').strip())
```

Ambas as opções levam à função "main()":

```python
def main(sentenca, modo = "bosque2.3_golden_train.joblib"):

	sentenca = tokenizar(sentenca)
	tripla_sentenca = montar_tuplas(sentenca)

	...

```

### tokenizar()

A segunda etapa é tokenizar a entrada do usuário, seja o arquivo ou a sentença.

A primeira lista a seguir foi minerada do arquivo **bosque2.3_golden_train_dev_e_teste.conllu** ao se procurar por palavras cujas POS fossem *underline* ("\_"). Ou seja, procuramos por palavras não anotadas que, portanto, eram contrações:

```python
def tokenizar(sentenca):

	contractions = ['da', 'das', 'do', 'dos', 'dela', 'nos', 'ao', 'na', 'aos', 'no', 'às', 'nesse', 'pelos', 'numa', 'à', 'nas', 'pela', 'pelo', 'daí', 'deste', 'neste', 'desta', 'num', 'naquele', 'daquela', 'connosco', 'destes', 'dessas', 'pelas', 'duma', 'daquele', 'deles', 'nesta', 'desse', 'dessa', 'dele', 'daqueles', 'àquele', 'disso', 'disto', 'naquela', 'daquilo', 'delas', 'nele', 'nela', 'nessa', 'nestas', 'nessas', 'noutras', 'numas', 'nelas', 'destas', 'nalguns', 'dantes', 'daqui', 'nestes', 'convosco', 'nesses', 'daquelas', 'consigo', 'dum', 'noutro', 'desses', 'do(s)', 'neles', 'nalgumas', 'nisso', 'noutra', 'naqueles', 'noutros', 'comigo', 'dali', 'lha', 'àqueles', 'àquela', 'nisto', 'naquilo']

	dettached = ['de a', 'de as', 'de o', 'de os', 'de ela', 'em os', 'a o', 'em a', 'a os', 'em o', 'a as', 'em esse', 'por os', 'em uma', 'a a', 'em as', 'por a', 'por o', 'de aí', 'de este', 'em este', 'de esta', 'em um', 'em aquele', 'de aquela', 'com nós', 'de estes', 'de essas', 'por as', 'de uma', 'de aquele', 'de eles', 'em esta', 'de esse', 'de essa', 'de ele', 'de aqueles', 'a aquele', 'de isso', 'de isto', 'em aquela', 'de aquilo', 'de elas', 'em ele', 'em ela', 'em essa', 'em estas', 'em essas', 'em outras', 'em umas', 'em elas', 'de estas', 'em alguns', 'de antes', 'de aqui', 'em estes', 'com vós', 'em esses', 'de aquelas', 'com si', 'de um', 'em outro', 'de esses', 'de os', 'em eles', 'em algumas', 'em isso', 'em outra', 'em aqueles', 'em outros', 'com mim', 'de ali', 'lhe a', 'a aqueles', 'a aquela', 'em isto', 'em aquilo']

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

### main()

Voltando à função principal, já tendo tokenizado a entrada do programa, carregamos o material previamente treinado (parâmetro "--train"). Então, finalmente, montamos a árvore de decisões.


```python
def main(sentenca, modo = "bosque2.3_golden_train.joblib"):

	...

	classifier = jb.loads(open(modo, 'rb').read())
	pal = jb.loads(open(modo + '2', 'rb').read())
	rot = jb.loads(open(modo + '3', 'rb').read())

	for i, tripla in enumerate(tripla_sentenca):
		predicao = classifier.predict(pal.transform([tripla]))
		inverso = rot.inverse_transform(predicao)

		sentenca[i] = "{}_{}".format(sentenca[i], inverso[0])

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

[**↥ voltar ao topo**](#anotador-de-pos)

## LabelEncoder e OneHotEncoder

O LabelEncoder e o OneHotEncoder são necessários pois estamos lidando com *strings* tanto em relação às palavras que queremos anotar quanto aos rótulos das anotações.

O LabelEncoder é o responsável por transformar os rótulos (as classes gramaticais) em números, e o OneHotEncoder trata das "features" (o n-gramas do material de treino) de forma que todas as palavras sejam tratadas como "independentes" e não se confundam entre si (!verificar veracidade da informação).

[**↥ voltar ao topo**](#anotador-de-pos)
