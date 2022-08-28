*********************
Parameter Estimation
*********************

We begin with an estimation of the threshold values (:math:`t_c, t_p`) for each subtask :math:`v_i` from the human performance data, following the state-of-the-art estimation method (see Section Background).
To compute the human-tolerated range of visual changes, state-of-the-art uses the binomial statistical test, which is specific to the task metric. 
E.g., for the metric Precision-Recall curve :math:`PR` (used for object detection and instance segmentation), obtained with the human experiment data, the binomial test is performed on each point of the curve, checking if :math:`PR` for transformed images is below that of the original images with sufficient statistical significance.
Since in the empirical studies one does not have the data of all the points of the :math:`PR` curve, the binomial test is performed only in the curve locations where sufficient data is available. 
To define binomial test in a point of a :math:`PR` curve, we need to define two binomial tests, for precision and recall at that point, respectively.
Specifically, for each individual binomial test, we follow the procedure of state-of-the-art to estimate the human-tolerated range of visual changes, :math:`\Delta_v` (see Section Background), and pick the minimum range for the :math:`PR` curve binomial test, i.e., :math:`\Delta^{PR}_v = min{(\Delta^{prec}_v, \Delta^{rec}_v})`.
The resulting :math:`t_c` and :math:`t_p` thresholds for the entire :math:`PR` curve are obtained by considering the visual change :math:`\Delta^{PR}_v` value at the curve point that has been obtained with the largest amount of human data.

Although not shown in the paper, we also conducted experiments with the transformation brightness and obtained thresholds as below:

Estimated Thresholds for brightness
-----------------------------------
.. image:: images/brightess_thresholds.png
  :alt: thresholds estimated for brightness
