#!/bin/bash
python eval_pascal_voc_new.py -t frost -v D -th 0.9 -r 1
python eval_pascal_voc_new.py -t frost -v D -th 0.7 -r 1
python eval_pascal_voc_new.py -t frost -v D -th 0.3 -r 1

python eval_pascal_voc_new.py -t frost -v I -th 0.9 -r 1
python eval_pascal_voc_new.py -t frost -v I -th 0.7 -r 1
python eval_pascal_voc_new.py -t frost -v I -th 0.6 -r 1
python eval_pascal_voc_new.py -t frost -v I -th 0.3 -r 1

python eval_pascal_voc_new.py -t brightness -v D -th 0.9 -r 1
python eval_pascal_voc_new.py -t brightness -v D -th 0.8 -r 1
python eval_pascal_voc_new.py -t brightness -v D -th 0.7 -r 1
python eval_pascal_voc_new.py -t brightness -v D -th 0.2 -r 1

python eval_pascal_voc_new.py -t brightness -v I -th 0.9 -r 1
python eval_pascal_voc_new.py -t brightness -v I -th 0.8 -r 1
python eval_pascal_voc_new.py -t brightness -v I -th 0.7 -r 1
python eval_pascal_voc_new.py -t brightness -v I -th 0.5 -r 1
python eval_pascal_voc_new.py -t brightness -v I -th 0.2 -r 1