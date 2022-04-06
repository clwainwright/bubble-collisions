Package Overview
================

Installation
------------

Dependencies
~~~~~~~~~~~~

There are two routes to installing all of the dependencies need to run the *bubble_collisions* package. The easy way is to just download and install a large scientific python distribution. Both the `Enthought Python Distribution`_ and the `Anaconda`_ should work well.

.. _Enthought Python Distribution: https://www.enthought.com/products/epd/
.. _Anaconda: https://store.continuum.io/cshop/anaconda/

If you want to have more control over what your installation, you can use the Python Package Index to install all of the dependencies. First, make sure you have a recent (v2.7) version of Python running on your system. This comes standard on OS X Mavericks, or it can be downloaded from python.org.
Next install *pip*, a python package manager. Just run ``easy_install pip`` from the command line (you might need to use ``sudo``). If you want to maintain a clean installation, you should also install `virtual environments`_ (``pip install virtualenv``). This will let you create a dedicated python environment just for this project, without worrying about polluting namespace or breaking packages in other projects.

.. _virtual environments: http://docs.python-guide.org/en/latest/dev/virtualenvs/

Once you have python installed and the environment setup (``virtualenv my_env; source my_env/bin/activate``), you're ready to start installing dependencies. The following are required::

    pip install numpy --upgrade # should be at least v1.7
    pip install scipy
    pip install ipython[notebook]
    pip install matplotlib

You may need to first install fortran compilers to get these working (if running OS X, these are part of the Xcode developer tools). 


Package installation
~~~~~~~~~~~~~~~~~~~~

Once the dependencies are installed, installing the actual package is very simple. Just navigate to *adaptiveGrid* folder and run ``python setup.py install``. This should compile and install the package into the *site-packages* folder in your python distribution. After that, you should be able to run ``import bubble_collisions`` from a python prompt even outside of this directory.

Additionally, if you want to recreate this documentation, download sphinx (``pip install sphinx; pip install sphinxcontrib-napoleon``), navigate to the *docs* directory, and ``make html``.


Usage
-----

*To fill in later*