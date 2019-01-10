#!/usr/bin/env python3
print('Content-type:text/html\n\n')

import os
import cgi, cgitb
cgitb.enable()
import estrutura_dados
from sklearn import preprocessing
from sklearn import tree
import pickle as jb
import copy
import string
import re
import sys
import os

html = '<html><head><meta name="viewport" charset="UTF-8" http-equiv="content-type" content="text/html, width=device-width, initial-scale=1.0"><title>Anotador de POS</title></head><body style="margin: auto; width: 80%; height: 60%;">'

if os.environ['REQUEST_METHOD'] != 'POST' and not 'conllu' in cgi.FieldStorage():
	html += '''<h1>Anotador de POS</h1><hr><a href="http://github.com/alvelvis/anotador-de-pos" target="_blank">Página do projeto no GitHub</a><br><br><br><center><form action="index.py" method=POST><textarea style="width:60%; height:40%;" name="sentence" placeholder="Insira aqui o texto a ser anotado..." required></textarea><br><br>'''
	html += '<input type=text name="modelo" placeholder="Escolha um modelo..." list=cars><datalist id=cars>'
	for item in os.listdir('../cgi-bin'):
		if '.' in item and item.rsplit('.', 1)[1] == 'joblib':
			html += '<option>' + item + '</option>'
	html += '''</datalist><br><br><input onclick="document.getElementById('complete').value = 'Anotando...'" id=complete name="complete" type=submit value="Anotar part-of-speech"></form></center>'''

#POST
elif os.environ['REQUEST_METHOD'] == 'POST':
	import tagger
	form = cgi.FieldStorage()
	modo = form['modelo'].value

	print(html)
	print('<br>Modelo:', modo, '<br>')
	if os.path.isfile(modo + '4'): print(open(modo + '4', 'r').read())
	print('<pre>')
	tagger.main(form['sentence'].value, modo)
	print('</pre><br><hr><a href="index.py">Voltar</a>')

elif 'conllu' in cgi.FieldStorage():
	import tagger
	tagger.treinar(cgi.FieldStorage()['conllu'].value)

html += '</body></html>'
print(html)
