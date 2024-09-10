:notoc:

.. _examples:

********
Examples
********

Pelicun examples are listed in the following index.

.. attention::

   These documentation pages are brand new and in active development.
   Increasing the set of examples is a very high priority.
   Please come back soon!

Complete list
-------------

+-----------+---------------------------------------------------------+
|Example    |Description                                              |
+===========+=========================================================+
|`E2`_      |This example demonstrates the seismic performance        |
|           |assessment of a steel moment frame structure using the   |
|           |FEMA P-58 methodology.                                   |
+-----------+---------------------------------------------------------+

Grouped by feature
------------------

The following sections group the examples above based on the specific features they illustrate, helping you pick the ones most relevant to what you are looking for.

.. dropdown:: Asset type

   +------------------------------------------------------+---------------------------------------------------------+
   |Feature                                               |Examples                                                 |
   +======================================================+=========================================================+
   |Building                                              |`E2`_                                                    |
   +------------------------------------------------------+---------------------------------------------------------+

.. dropdown:: Demand Simulation

   +------------------------------------------------------+---------------------------------------------------------+
   |Feature                                               |Examples                                                 |
   +======================================================+=========================================================+
   |:ref:`Model calibration <fo_calibration>`             |`E2`_                                                    |
   +------------------------------------------------------+---------------------------------------------------------+
   |:ref:`RID|PID inference <fo_pidrid>`                  |`E2`_                                                    |
   +------------------------------------------------------+---------------------------------------------------------+
   |:ref:`Sample expansion <fo_sample_expansion>`         |`E2`_                                                    |
   +------------------------------------------------------+---------------------------------------------------------+


.. dropdown:: Loss estimation

   +------------------------------------------------------+---------------------------------------------------------+
   |Feature                                               |Examples                                                 |
   +======================================================+=========================================================+
   |:ref:`Loss maps <fo_loss_maps>`                       |`E2`_                                                    |
   +------------------------------------------------------+---------------------------------------------------------+
   |:ref:`Active decision variables <fo_active_dvs>`      |`E2`_                                                    |
   +------------------------------------------------------+---------------------------------------------------------+
   |:ref:`Loss aggregation <fo_loss_aggregation>`         |`E2`_                                                    |
   +------------------------------------------------------+---------------------------------------------------------+

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/example_2.pct.py

.. _E2: notebooks/example_2.pct.py
