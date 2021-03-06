.. bubble_collisions documentation master file, created by
   sphinx-quickstart on Mon Aug  4 12:40:21 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the bubble_collisions documentation!
===============================================

The ``bubble_collisions`` package contains (or will contain) everything needed to simulate and analyze collisions between bubble universes in an eternal inflation framework. The core code is written in C with a Python interface, and there are several additional Python scripts for analysis.

The original code was described in `arXiv:1312.1357`_, and later used in `arXiv:1407.2950`_ and `arXiv:1508.03641`_.

Note that in some places I use *N* to indicate the time variable, which matches the notation in the papers, and in some places I use *t*, which is much less ambiguous. The *N* variable can also refer to the relative time step in a simulation region (:math:`\Delta t \propto 2^N`).

.. _`arXiv:1312.1357`: http://arxiv.org/abs/abs/1312.1357
.. _`arXiv:1407.2950`: http://arxiv.org/abs/abs/1407.2950
.. _`arXiv:1508.03641`: https://arxiv.org/abs/1508.03641v1


To do
=====

- [x] Create documentation for collisionRunner
- [x] Create documentation for derivsAndSmoothing
- [x] Merge to CosmoTransitions v2
- [x] Reimplement coordsAndGeos.py
- [x] Update full_sky.py (minor) and add documentation
- [x] Update perturbation_fits.py (minor) and add documentation
- [ ] Write up the usage section of the overview.

Contents
========

.. toctree::
   :maxdepth: 2

   Overview
   simulation
   models
   collisionRunner
   derivsAndSmoothing
   geodesics
   bubble_analytics
   full_sky
   perturbation_fits


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

