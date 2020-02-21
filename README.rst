FORCE BDSS Nevergrad Plugin
---------------------------

.. image:: https://travis-ci.com/force-h2020/force-bdss-plugin-nevergrad.svg?branch=master
   :target: https://travis-ci.com/force-h2020/force-bdss-plugin-nevergrad
   :alt: Build status

.. image:: http://codecov.io/github/force-h2020/force-bdss-plugin-nevergrad/coverage.svg?branch=master
   :target: http://codecov.io/github/force-h2020/force-bdss-plugin-nevergrad?branch=master
   :alt: Coverage status

This repository contains the implementation of a plugin for the Business Decision Support System (BDSS), contributing
the `Nevergrad <https://github.com/facebookresearch/nevergrad>`_ gradient free optimization package.
It is implemented under the Formulations and Computational Engineering (FORCE) project within Horizon 2020
(`NMBP-23-2016/721027 <https://www.the-force-project.eu>`_).

The ``NevergradPlugin`` class contributes BDSS MCO objects, including a ``BaseOptimizerEngine`` implementation
that acts as a wrapper around the ``nevergrad`` package, and a stand alone ``BaseMCO`` subclass that can
be used for any use case.

Installation
-------------
Installation requirements include an up-to-date version of ``force-bdss``. Additional modules that can contribute to the ``force-wfmanager`` UI are also included,
but a local version of ``force-wfmanager`` is not required in order to complete the
installation.


To install ``force-bdss`` and the ``force-wfmanager``, please see the following
`instructions <https://github.com/force-h2020/force-bdss/blob/master/doc/source/installation.rst>`_.

After completing at least the ``force-bdss`` installation steps, clone the git repository::

    git clone https://github.com/force-h2020/force-bdss-plugin-nevergrad

the enter the source directory and run::

    python -m ci install

This will allow install the plugin in the ``force-py36`` edm environment, allowing the contributed
BDSS objects to be visible by both ``force-bdss`` and ``force-wfmanager`` applications.

Documentation
-------------

To build the Sphinx documentation in the ``doc/build`` directory run::

    python -m ci docs
