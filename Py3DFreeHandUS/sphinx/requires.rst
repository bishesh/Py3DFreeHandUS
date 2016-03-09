Requirements
============

We have used Python 2.7 for the implementation of Py3DFreeHandUS. It requires the following external packages:

- NumPy
- SciPy (0.11.0+)
- matplotlib
- `SymPy <http://sympy.org/en/index.html>`_
- `pydicom <https://code.google.com/p/pydicom/>`_
- `btk <https://code.google.com/p/b-tk/>`_
- VTK
- OpenCV (2.4.9+)
- Cython + gcc (optional, but stringly suggested for reducing some computation bottlenecks)
- `MoviePy <http://zulko.github.io/moviepy/>`_ (**ffdshow** codecs required)
- scikit-image

All-in packages, such as `Anaconda Python <https://store.continuum.io/cshop/anaconda/>`_, already include most of the external dependencies.

In the very likely case the ultrasound files are large (GBytes), it is **strongly** suggested to use a 64-bit Python distribution.

We carefully paid *much* attention about critical aspects such as vectorization, preallocation, RAM consumption,... 
to reach realistic computation time on commodity hardware, we still suggest to use a machine with **1.6+ GHz and 4+ GB RAM**.
However, this mainly depends on the factors desribed in :ref:`when-mem-error`.

The tool was tested with data recorded by the following systems:

- US: `Telemed <http://www.telemedultrasound.com/?lang=en>`_, `ESAOTE <http://www.esaote.it/>`_
- Optoelectronic: `Optitrack <http://www.naturalpoint.com/optitrack/>`_, `Vicon <http://vicon.com/>`_

Theoretically, *any* US or optoelectronic device able to export DICOM and C3D files can be used.
We suggest both devices to be **hardware-synchronized** for start and stop acquisition triggers. However, we provided time delay estimation and compensation techniques.

The file formats tested are:

- US: **uncompressed** DICOM. Being uncompressed is a must, since at the moment ``pydicom`` doesn't support compressed formats. We used `MeVisLab <http://www.mevislab.de/>`_ for this.
- Optoelectronic: C3D. 

Normally, US systems are also able to export images sequences in AVI format. When compressed, these files are much smaller than uncompressed DICOM (or AVI) files. But, depending on the US machine,
compression can be *lossy*, not ideal for high-quality 3D morphology reconstruction. Plus, AVI files cannot contain mata-data like *Frame Time Vector*, necessary for re-synching optoelectronic
data on US data (see method ``process.Process.setDataSourceProperties()``)


