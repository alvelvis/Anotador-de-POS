#!/usr/bin/python3
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

html = '<html><head><meta name="viewport" charset="UTF-8" http-equiv="content-type" content="text/html, width=device-width, initial-scale=1.0"><title>Anotador de POS</title></head><body style="margin: auto; width: 80%; height: 60%;">'

if os.environ['REQUEST_METHOD'] != 'POST':
	html += '''<h1>Anotador de POS</h1><hr><a href="http://github.com/alvelvis/anotador-de-pos" target="_blank">Página do projeto no GitHub</a><br><br><br><center><form action="index.py" method=POST><textarea style="width:60%; height:40%;" name="sentence" placeholder="Insira aqui o texto a ser anotado..." required></textarea><br><br><input onclick="document.getElementById('fast').value = 'Anotando...'" id=fast name="fast" type=submit value="Anotação rápida"> <input onclick="document.getElementById('complete').value = 'Anotando...'" id=complete name="complete" type=submit value="Anotação completa"></form></center>'''

#POST
else:
	import tagger
	form = cgi.FieldStorage()
	modo = "bosque2.3_golden_train.joblib" if "complete" in form else "bosque2.3_golden_dev.joblib"

	print(html)
	print('<pre>')
	tagger.main(form['sentence'].value, modo)
	print('</pre><br><hr><a href="index.py">Voltar</a>')

html += '</body></html>'
print(html)
