Introduction
============

**Py3DFreeHandUS** is a package for processing data acquired *simultaneously* by ultra-sonographic systems (**US**) and marker-based optoelectronic systems.
The `first ones <http://en.wikipedia.org/wiki/Medical_ultrasonography>`_ being medical devices used for visualizing subcutaneous body structures 
including tendons, muscles, joints, vessels and internal organs. The `second ones <http://www2.brooklyn.liu.edu/bbut04/adamcenter/Instrumented%20Analysis%20Website/>`_ 
being a based on a collection cameras able to reconstruct the 3D position in space of special tiny objects called *markers* (small grey balls in the image below).
These systems are mostly famous for allowing the recording of human motion for special effects purpose in cinema.
By combining these two measurement devices, it is possible to reconstruct the real 3D morphology of body structures such as muscles.

The picture below shows the instrumental setup:

.. image:: calf_acquisition.png
   :scale: 50 %

This tool was developed by the University of Leuven (Belgium) through a collaboration between the following groups:

1. `CMAL-P <http://www.uzleuven.be/en/laboratory-for-clinical-movementanalysis/research>`_ (Clinical Movement Analysis Laboratory, UZ Pellenberg)
2. `PMA <http://www.mech.kuleuven.be/en/pma/>`_ (Department of Production engineering, Machine design and Automation)
3. `FaBeR <https://faber.kuleuven.be/english/>`_ (Faculty of Kinesiology and Rehabilitation Sciences)

and, at the moment, it is mainly used in the framework of this research project:

**"3D ultrasound system for morphological measurements"** (click `here <https://gbiomed.kuleuven.be/english/research/50000743/nrrg1/pr.htm#3D%20ultrasound>`_ for details).

All the implemented algorithms are based on the following articles:

.. [Ref1] Hsu PW, Prager RW, Gee AH, Treece GM. Freehand 3D Ultrasound Calibration: A Review, chapter 3, pages 47-84. Springer Berlin Heidelberg, Berlin, Heidelberg, 2009.
.. [Ref2] Prager RW, Rohling RN, Gee AH, Berman L. Rapid calibration for 3-D freehand ultrasound. Ultrasound Med Biol. 1998 Jul;24(6):855-69. PubMed PMID: 9740387.
.. [Ref3] Wein W, Khamene A. Image-Based Method for In-Vivo Freehand Ultrasound Calibration. Medical Imaging 2008. Imaging and Signal Processing.

It is strongly recommended to read the referenced articles above before using the package.