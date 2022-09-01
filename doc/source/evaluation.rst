*****************************
Additional Evaluation Results
*****************************

In addition to the image transformation adding artificial frost shown in the paper, we also conducted an additional set of evaluation using the image transformation changing brightness. 
Because of page limit of the paper, we show them here.

RQ1
====
Comparison of the human tolerated threshold estimated either directly or by composing the subtask thresholds. :math:`t_{c}` and :math:`t_{p}` are for *correctness-preservation* (:math:`cp`) and *prediction-preservation* (:math:`pp`), respectively. 

.. image:: images/RQ1_brightness.png
  :alt: results for RQ1 specific to brightness

As we can see in the image, our composed threshold is always the lowerbound threshold.

RQ2
====
In the following table, we have for the transformation changing brightness, the comparison of reliability evaluation of object detection and instance segmentation MVCs with our checking method using the SoTa benchmark dataset PASCAL VOC-C [PASCALVOC-C]_.

.. image:: images/rq2b.png
  :alt: RQ2 table with brightness

With the results highlighted in blue boxes, we can see that MVCs R50-C4-3x and R101-C4-3x have the same *AP* values but R101-C4-3x is more reliable to brightness than R50-C4-3x. Then, by checking reliability of subtasks, we can see that, highlighted with black boxes, although R50-C4-1x and R101-DC5-3x have similar *AP* values, R101-DC5-3x is more reliable for the subtask :math:`\mathbf{v}_{C|L}` and R50-C4-1x is more reliable for the subtask :math:`\mathbf{v}_L`. Also, as shown in red boxes, MVC R101-C4-3x has higher *AP* value than R50-C4-1x for instance segmentation, but it is less reliable against the transformation brightness. Additionally, for the object class bird, the reliability distance for :math:`pp_{\mathbf{V}_D}` and :math:`pp_{\mathbf{V}_I}` are all negative for all the MVCs. This suggests that these MVCs can preserve the prediction on original images better when there are more brightness adjustments in the images, meaning these MVCs are less reliable with less visual change in the images. These new reliability gaps identified with our checking method support our conclusion for RQ2.

RQ3
====
In the folowing table, we compare the average runtime and peak memory used to check the satisfaction of requirements against the image transformation changing brightness for object detection and instance segmentation W/O reusing decomposition analysis results.

.. image:: images/r3b.png
  :alt: RQ3 table with brightness

Similar to results shown in the paper for the transformation adding artificial frost, we can see on the above table that the peak memory is not affected significantly. In total, checking :math:`cp` satisfaction for instance segmentation  by reusing results from object detection took :math:`81.53` seconds while checking without reuse took :math:`317.23` seconds; checking :math:`pp` took :math:`31.33` seconds with reuse and :math:`476.97` seconds without.
On average, reuse decreased runtime by :math:`86\%`, supporting our conclusion for RQ3. 



..  [PASCALVOC-C] Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming `link <https://arxiv.org/abs/1907.07484>`_.
