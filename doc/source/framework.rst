***********************
DecompoVision Framework
***********************

In this website, we provide the supplementary material for our ICSE’2023 paper. 
Please find in the figure below an overview of the approach that we propose and a summary that is linked to the supplementary material sections.

.. image:: images/decompoFram.png
  :alt: DecompoVision framework worflow
  
  

Overview
--------
Based on our complex vision task decomposition, we develop a modular reliability framework DecompoVision (shown above), which builds on top of the state of the art reliability framework for image classfication [ICRAF]_.
Our framework first decomposes a complex vision task (c-task) into atomic subtasks (Step I).
Then, for each subtask independently, given an image transformation simulating scene changes, we use human performance data to generate reliability requirements (Step II.a and II.b).
Consequently, we compose the individual subtask requirements to get the requirements for the c-task (Step II.c).
Finally, we propose a checking method for both the overall c-task and the subtask requirements, which enables *failure localization* in the sequence of subtasks.
Note that the modularity of Decompovision allows us to reuse human performance data, requirements specifications, and analysis artifacts for shared subtasks across different c-tasks.


..  [ICRAF] If a Human Can See It, So Should Your System: Reliability Requirements for Machine Vision Components `link <https://arxiv.org/abs/2202.03930/>`_.



