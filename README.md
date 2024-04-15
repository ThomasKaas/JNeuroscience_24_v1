## Analysis Train Data (Patch-Clamp)

### General
This repository contains all the code used to analyze train data (e.g. postsynaptic currents evoked by extracellular stimulation) obtained by somatic patch-clamp recordings in iPSC-derived Human neurons in vitro as is presented in the following publication: 

"Human iPSC-derived neurons with reliable synapses and large presynaptic action potentials" Torsten Bullmann*, Thomas Kaas*, Andreas Ritzau-Jost*, Anne Wöhner, Toni Kirmann, Filiz Sila Rizalar,Max Holzer, Jana Nerlich, Dmytro Puchkov,Christian Geis, Jens Eilers, Robert J. Kittel, Thomas Arendt, Volker Haucke, Stefan Hallermann#. *These authors contributed equally. #Corresponding author.

For more information concerning the code contact: thomas.kaas.science@gmail.com.

### Requirements
The code was run on/with 
- Mac OS Montery (12.2.1)
- Individual Edition of Anaconda (https://www.anaconda.com) for the use of JuyiterNotebooks (6.3.0) and running the base/root environment provided by Anaconda 2.4.0

This code relies on the module "heka_reader.py" (https://github.com/campagnola/heka_reader), needed for reading .dat-files with Python.

### Run Examples
There are several examples on the final analysis that was performed. For that "analysis_example.ipnyb" can be run. 
Before trying out the analysis example data needs to be download from a seperate cloud-storage (https://drive.google.com/drive/folders/1aO84hQwlLNNjG7lRgpppHrgUeYzB5ZAL?usp=sharing). Copy the data under "/data_example". 

### Description of the files provided under "functions/"
The file "train_analysis_functions.py" contains all the procedures designed to perform analysis on a rather large dataset of train recordings.
The file "parameters_train_analysis.py" defines key parameters for analysis as well as the path to import raw data and meta-data as well as the path for exporting results. It is imported via "parameters_train_analysis.py". 

