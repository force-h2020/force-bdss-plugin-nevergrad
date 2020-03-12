FORCE Nevergrad Plugin Changelog
================================

Release 0.1.0
-------------

Released: 12/03/2020

Release notes
~~~~~~~~~~~~~

Version 0.1.0 is the inital release of the ``force-bdss-plugin-nevergrad`` package.

The following people contributed code changes for this release:

* Frank Longford
* Petr Kungurtsev

Features
~~~~~~~~
* BDSS plugin ``NevergradPlugin`` (#4), contributing with stand alone BDSS objects that can be
  used and customised 'out-of-the-box'
* Pinned to Nevergrad stable commit on master branch, using API from 0.3.2 (#4, #8)
* One BDSS ``OptimizerEngine`` subclass (#4, #13): ``NevergradOptimizerEngine``
* One BDSS ``MCO`` subclass (#4, #5, #10): ``NevergradMCO``
* Documentation included describing gradient free methods and Nevergrad API (#12)