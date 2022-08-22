****************
Mturk Experiment
****************

Table of Contents
=================

1. :ref:`Introduction`
2. :ref:`Classification`
3. :ref:`Draw Bounding Box`


Introduction
============


The mturk experiments are built with vue + vite.

Mturk only allows user to upload a single html file. You could write everything into a single html file using vanilla javascript, but this quickly gets super complicated.

The file gets too long and hard to maintain. We then developed an easy way to build complex experiments with Vue (or any other modern single page application framework).

How It Works
------------


`npm run build` generates a `dist` folder containing a `index.html` and this is the file you want to copy to Mturk.

But before copying the content to mturk, you need to upload all the dependencies (static assets such as css, js, images, etc) to AWS S3.

AWS S3 serves as a CDN for the static assets, the `index.html` downloads them when web page loads.

By default the generated `index.html` references the dependency files from local folder, you have to prefix the path with the S3 bucket url.

For example, the original generated code is the following

.. code-block::

       <link rel="stylesheet" href="./assets/index.1431c11d.css">

You need to change it to

.. code-block::

       <link rel="stylesheet" href="https://bucket-name.s3.us-east-2.amazonaws.com/classification/assets/index.1431c11d.css">


Fortunately, we provide a patch script to update `href` automatically. It's called `patch-s3.sh`.

We also provide a Makefile containing all the commands you will need from building, patching, to uploading.

Simply run `make`, and everything will be done for you.

If you were to build your own project, don't forget to change the urls in `patch-s3.sh` and `Makefile`.

Classification
==============


.. raw:: html

    <video width="600" controls controlsList="nodownload">
    <source
        src="https://mturk-host.s3.us-east-2.amazonaws.com/instructions/classification-demo.mov"
        type="video/mp4"
    />
    Your browser does not support the video tag.
    </video>



Draw Bounding Box
=================

.. raw:: html

    <video width="600" controls controlsList="nodownload">
    <source
        src="https://mturk-host.s3.us-east-2.amazonaws.com/instructions/detection/detection-video-demo.mp4"
        type="video/mp4"
    />
    Your browser does not support the video tag.
    </video>
