#!/bin/bash
python eval_pascal_voc_new.py -t frost -v D -th 0.9 -r 
python eval_pascal_voc_new.py -t frost -v D -th 0.7 -r 
python eval_pascal_voc_new.py -t frost -v D -th 0.3 -r 

python eval_pascal_voc_new.py -t frost -v I -th 0.9 -r 
python eval_pascal_voc_new.py -t frost -v I -th 0.7 -r 
python eval_pascal_voc_new.py -t frost -v I -th 0.6 -r 
python eval_pascal_voc_new.py -t frost -v I -th 0.3 -r 

python eval_pascal_voc_new.py -t brightness -v D -th 0.9 -r 
python eval_pascal_voc_new.py -t brightness -v D -th 0.8 -r 
python eval_pascal_voc_new.py -t brightness -v D -th 0.7 -r 
python eval_pascal_voc_new.py -t brightness -v D -th 0.2 -r 

python eval_pascal_voc_new.py -t brightness -v I -th 0.9 -r 
python eval_pascal_voc_new.py -t brightness -v I -th 0.8 -r 
python eval_pascal_voc_new.py -t brightness -v I -th 0.7 -r 
python eval_pascal_voc_new.py -t brightness -v I -th 0.5 -r 
python eval_pascal_voc_new.py -t brightness -v I -th 0.2 -r 