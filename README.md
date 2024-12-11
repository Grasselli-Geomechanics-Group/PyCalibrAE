# [PyCalibrAE](https://github.com/Grasselli-Geomechanics-Group/PyCalibrAE.git)

Toolbox for broadband calibration of acoustic emission sensors, see accompanying paper [Broadband Calibration of Acoustic Emission Sensors: Toolbox, Applications, and Key Insights](). 
If you use this package, please cite XXX.

This toolbox is a broadband calibration toolbox for absolute sensor calibration using theoretical Hertzian ball impacts on a transfer medium.
We utilize the spectral-element method (SEM) with the open-source [SPECFEM3D Cartesian](https://github.com/SPECFEM/specfem3d) framework to simulate high-resolution seismic wave propagation. 
By modeling point loading on an aluminum plate, Green's function is obtained at multiple receiver locations, enabling flexible calibration with any source mechanism (e.g. glass capillary, ball impact, pulse excitation).
Alternatively, we provide a translated Python version of the generalized ray theory (GRT) Green's function solution made available as a separate toolbox; see [PyPlateSolution](https://github.com/Grasselli-Geomechanics-Group/PyPlateSolution.git).
By making these toolboxes and methodologies widely accessible, we aim to enhance the understanding of acoustic emissions across diverse fields.

This toolbox contains two main functions. 
1. The `instrument_response_algorithm.py` calculates the instrument response and
corresponding error at a single location for an `N` number of identical ball impacts. 
Firstly, run this script for all ball impacts at a given sensor location. 
The calculated instrument response will be output in the `PyCalibrAE/out/` folder as a `pickle` module
(e.g. R15a_0.50mm_chC.pickle). 
2. The `instrument_response_stitch.py` stitches different ball impacts (e.g. 0.5mm, 0.66mm, 2.5mm)
to create a broadband calibration curve. The individual `.pickle` files across different ball impacts (e.g. R15a_0.50mm_chC.pickle, R15a_0.66mm_chC.pickle, R15a_2.50mm_chC.pickle)
are stitched together to form a single broadband calibration curve (e.g. R15a_chC.pickle).
3. The `function_modules.py` contains various functions to help automatically align waveform, read the
data format, etc.

Due to repository limits, the example sensor data [sensor_data](https://doi.org/10.5683/SP3/II56AM) and simulated SEM results [SEM_data](https://doi.org/10.5683/SP3/71FAOR) must be downloaded from the [University of Toronto Dataverse](https://borealisdata.ca/dataverse/PyCalibrAE).
Both `sensor_data` and `SEM_data` directories should be placed within the working `PyCalibrAE` folder. 
1. The general simulated SEM results are in the `SEM_data` folder. These are the theoretical velocity responses (.semv) due to the Heaviside force-time function at
a given incident angle (`.POS00-90`) for the X (`.FXX`), Y (`.FXY`) and Z (`.FXZ`) components.
2. The sensor data are located within the `sensor_data` folder. A subfolder is used for a given sensor (e.g. R15a) within this folder.
Within the individual sensor folder (PyCalibrAE/R15a/), a subfolder contains all the ball impact data for that given ball size 
(e.g. PyCalibrAE/R15a/0.5mm/.mat). The `.mat` files are the output MATLAB data files from the PicoScope 4824A data acquisition software [PicoScope 7](https://www.picotech.com/products/picoscope-7-software).
Additional information, such as the `ball_diameter` (mm), `drop_height` (cm), and acquisition `voltage_range` (V), has been manually appended to the `.mat` files.