Workflow for 3D volume reconstruction
=====================================

The following chapter **summarizes** the most important concepts described in articles [Ref1]_, [Ref2]_, [Ref3]_.
It is **strongly recommended** to read those first.

Briefly, there are two different measurement systems involved:

1. US (*UltraSound*) system: this system, based on ultra-sounds reflexion, is able to acquire 2D images regarding 
subcutaneous body structures including tendons, muscles, joints, vessels and internal organs. This system is 
normally composed by a **probe**, containing the sensors able to record reflected ultrasounds. The pixels of each image
can be expressed in the **scan reference frame P**. Normally, all the acquired images are stored into **DICOM** files.

2. Marker-based optoelectronic system (*opto*): an ensemble of cameras capturing, at the same time, the 3D position of 
(active or passive) **markers**. The position is expressed with respect to a **global reference frame T**. The positions
of the markers is normally stored in a binary **C3D** file.

The start-acquisition and stop-acquisition triggers for both devices are supposed to be **hardware-synchronized**.
The two systems have different **acquisition frequencies**, normally the opto system one being higher.

By putting some markers on the US probe, it is possible to construct a **probe reference frame R**.
Thus, it is possible to know, for very time frame of the combined data acquisition, the **attitude**
(rotation + translation) of the US probe with respect to **T**. Since the position of the 2D scans
with respect to **R** can be found by a procedure called **Calibration**, then it also possible to know
the attitude of each 2D scan (and each pixel into it) into **T**.

Let the attitude from frame **A** to **B** be expressed by a 4x4 matrix containing rotation matrix and position vector: :math:`^{B}T_{A}`. 
Let :math:`^{H}\bar{x}` be the pixel coordinates vector in a generic reference frame **H**.

What mentioned above is expressed by the following equation: :math:`^{T}\bar{x}\ =\ ^{T}T_{R}\ ^{R}T_{P}\ ^{P}\bar{x}`.

These are the steps performed during a full processing session:

Devices delay estimation and compensation
-----------------------------------------

Although we strongly suggest to hardware-trigger both devices with a common start-stop acquisition trigger, this is not always
possible or anyway there is still a time delay to be compensated later. We provide the possibility to estimate and compensate 
this time delay.

Probe spatial calibration
-------------------------

This is aimed at calculating :math:`^{R}T_{P}` (see [Ref2]_). Briefly, it is based on solving a system of equations similar to the 
one above (by expressing :math:`^{T}\bar{x}` in a more convenient **calibration phantom reference frame C**) and imposing 
:math:`^{C}\bar{x}` to respect some constraints.

Below a schematic overview of the calibration protocol. US probe is scanning the bottom of a water tank. Here, the constraint is 
that, for all the US image, that a the vertical coordinate of :math:`^{C}\bar{x}` is always 0 for all the time frames.

.. image:: probe_calib.png
   :scale: 50 %


Calibration quality assessment
------------------------------

This is the process of estimating both **precision** and **accuracy** of the calibration phase. **Precision** gives an indication
of the dispersion of measures around their mean. **Accuracy** gives an indication of the difference between the mean of the measures
and the real value. For details, read [Ref1]_. This measure can be, for instance, the known position of a point in space (*Point accuracy*)
or the known dimension of an object (*Distance accuracy*).


3D Voxel reconstruction
-----------------------

Here, the 2D US scans are "aligned" in the 3D space by using the equation above. A **3D voxel-array** is created, containing 
the grey values of all the repositioned original pixels. The voxel-array (a parallelepipedon) should be the smallest one containing
the sequence of realigned scans, in order to avoid RAM waste.


Gaps filling
------------

After all the scans are correctly positioned in the 3D space, there are inevitably "gaps" in the voxel-array, i.e., voxels for which
the grey value is unknown. The quick-and-dirty way, known as **VNN** (*Voxel Nearest Neighbour*), consists of filling a gap by using 
the closest voxel having an assigned grey value. But other more sophisticated techniques are available. 


Further analysis
----------------

The gaps-filled voxel-array can be served for other further analysis, such as **features extraction** like **body structure border extraction**
and **volume calculation**.

Below, we show the result of manual border identification and surface mesh reconstruction of a human calf muscle (by using `MeVisLab <http://www.mevislab.de/>`_):

.. image:: 3d_mesh_muscle.png
