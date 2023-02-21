## apipeline
Direction-dependent calibration for Apertif. The pipeline is based on the LOFAR Default Preprocessing Pipeline (DPPP/DP3 van Diepen et al. 2018) and the wsclean imaging package (Offringa et al. 2014). It is recommended to use Docker container with the LOFAR tools installed. See the Dockerfile. 

Usage: 
```
imcal.py [-h] [-c CONFIGFILE] [-o OUTBASE] [-s STEPS] [-f] msin

DDCal Inputs

positional arguments:
  msin                  MS file to process

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIGFILE, --config CONFIGFILE
                        Config file
  -o OUTBASE, --outbase OUTBASE
                        output prefix
  -s STEPS, --steps STEPS
                        steps to run. Example: "nvss,mask,dical,ddcal"
  -f, --force           Overwrite the existing files
```

The config file `imcal.yml` has sections for every step with the corresponding parameters.
Most of the parameters have explicit comments inside the config file. 
For example the initial section *split* allows splitting the data by providing _startchan_ and _nchan_ parameters. 
The next section, *nvss*, can be used for preliminary phase calibration against the NVSS catalog... 

For more information see:

`wsclean` page: https://wsclean.readthedocs.io

`DPPP` page: https://support.astron.nl/LOFARImagingCookbook/dppp.html




Citation: *Kutkin et al. 2023 (TBA)*

