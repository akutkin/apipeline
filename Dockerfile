FROM ubuntu:20.04 as builder

# This Docker image builds the dependencies for the Rapthor pipeline.
# It lives on the head of its dependencies.

# Install all build-time dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \ 
    apt-get update && \
    apt-get install -y \
        bison \
        build-essential \
        casacore-data \
        casacore-dev \
        cmake \
        flex \
        gfortran \ 
        git \
        libblas-dev \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-numpy-dev \
        libboost-program-options-dev \
        libboost-python-dev \
        libboost-system-dev \
        libboost-test-dev \
        libcfitsio-dev \
        libfftw3-dev \
        libgsl-dev \
        libgtkmm-3.0-dev \
        libhdf5-serial-dev \
        liblapack-dev \
        liblua5.3-dev \
        libncurses5-dev \
        libpng-dev \
        libpython3-dev \
        pkg-config \
        python3 \
        python3-casacore \
        python3-numpy \
        wcslib-dev \
        wget && \
    mkdir -p /src

WORKDIR /src

# Build portable binaries by default
ARG PORTABLE=TRUE

ARG LOFARSTMAN_VERSION=master
RUN git clone --depth 1 --branch ${LOFARSTMAN_VERSION} \
        https://github.com/lofar-astron/LofarStMan && \
    mkdir LofarStMan/build && \
    cd LofarStMan/build && \
    cmake .. -DPORTABLE=${PORTABLE} && \
    make install -j`nproc`

ARG DYSCO_VERSION=master
RUN git clone --depth 1 --branch ${DYSCO_VERSION} \
        https://github.com/aroffringa/dysco.git && \
    mkdir dysco/build && \
    cd dysco/build && \
    cmake .. -DPORTABLE=${PORTABLE} && \
    make install -j`nproc`
    
ARG IDG_VERSION=master
# IDG doesn't work with --depth 1, because it needs all branches to
# determine its version :-(
RUN git clone --branch ${IDG_VERSION} \
        https://git.astron.nl/RD/idg.git && \
    mkdir idg/build && \
    cd idg/build && \
    cmake .. && \
    make install -j`nproc`

ARG AOFLAGGER_VERSION=master
RUN git clone --branch ${AOFLAGGER_VERSION} \
        https://gitlab.com/aroffringa/aoflagger.git && \
    mkdir aoflagger/build && \
    cd aoflagger && git fetch && git checkout bfb3978e734911457555ac244d255f4b4ce6df68 && \
    cd build && \
    cmake .. -DPORTABLE=${PORTABLE} && \
    make install -j`nproc`

ARG LOFARBEAM_VERSION=master
RUN git clone  --depth 1 --branch ${LOFARBEAM_VERSION} \
        https://github.com/lofar-astron/LOFARBeam.git && \
    mkdir LOFARBeam/build && \
    cd LOFARBeam/build && \
    cmake .. && \
    make install -j`nproc`

ARG EVERYBEAM_VERSION=master
RUN git clone --depth 1  --branch ${EVERYBEAM_VERSION} \
        https://git.astron.nl/RD/EveryBeam.git && \
    mkdir EveryBeam/build && \
    cd EveryBeam/build && \
    cmake .. && \
    make install -j`nproc`

ARG SAGECAL_VERSION=master
RUN git clone --depth 1 --branch ${SAGECAL_VERSION} \
        https://github.com/nlesc-dirac/sagecal && \
    mkdir sagecal/build && \
    cd sagecal/build && \
    cmake .. -DLIB_ONLY=1 && \
    make install -j`nproc`

ARG DP3_VERSION=master
RUN git clone --depth 1 --branch ${DP3_VERSION} \
        https://git.astron.nl/RD/DP3.git && \
    mkdir DP3/build && \
    cd DP3/build && \
    cmake .. -DPORTABLE=${PORTABLE} -DLIBDIRAC_PREFIX=/usr/local/ && \
    make install -j`nproc`

ARG WSCLEAN_VERSION=master
RUN git clone --depth 1 --branch ${WSCLEAN_VERSION} \
        https://gitlab.com/aroffringa/wsclean.git && \
    mkdir wsclean/build && \
    cd wsclean/build && \
    cmake .. -DPORTABLE=${PORTABLE} && \
    make install -j`nproc`


# kvis
#RUN cd /src && \
#    apt-get install -y libxaw7 && \
#    wget ftp://ftp.atnf.csiro.au/pub/software/karma/karma-1.7.25-common.tar.bz2 && \
#    wget ftp://ftp.atnf.csiro.au/pub/software/karma/karma-1.7.25-amd64_Linux_libc6.3.tar.bz2 && \
#    tar -xvf karma-1.7.25-amd64_Linux_libc6.3.tar.bz2 && \
#    tar -xvf karma-1.7.25-common.tar.bz2 && \
#    mv karma-1.7.25 /usr/local/karma && \
#    ln -s /usr/local/karma/amd64_Linux_libc6.3/bin/./kvis /usr/local/bin/kvis && \
#    ln -s /usr/local/karma/amd64_Linux_libc6.3/bin/./kshell /usr/local/bin/kshell && \
#    ln -s /usr/local/karma/amd64_Linux_libc6.3/bin/./kpvslice /usr/local/bin/kpvslice
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/karma/amd64_Linux_libc6.3/lib/
#ENV KARMABASE="/usr/local/karma/amd64_Linux_libc6.3"

    
    
# Do not use `pip` from the Debian repository, but fetch it from PyPA.
# This way, we are sure that the latest versions of `pip`, `setuptools`, and
# `wheel` are installed in /usr/local, the only directory we're going to copy
# over to the next build stage.
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Install required python packages
RUN python3 -m pip install --no-cache-dir --upgrade \
    cwl-runner \
    cwltool

# AO tools
RUN \
    cd /src && \
    git clone https://github.com/aroffringa/modeltools.git modeltools && \
    cd modeltools && \
    mkdir build && \
    cd build && \
    cmake -DPORTABLE=True ../ && \
    make -j4 && \
    cp bbs2model cluster editmodel render /usr/local/bin/ && \
    cd
    
# Tom's stuff
ADD makemask /src/makemask
RUN cd /src/makemask && \
    gcc makeMaskFits.c -o makeMaskFits -lcfitsio -lm && \
    gcc makeNoiseMapFits.c -o makeNoiseMapFits -lcfitsio -lm && \
    gcc makeNoiseMapFitsLow.c -o makeNoiseMapFitsLow -lcfitsio -lm && \
    gcc makeCombMaskFits.c -o makeCombMaskFits -lcfitsio -lm && \
    cp makeMaskFits makeNoiseMapFits makeCombMaskFits makeNoiseMapFitsLow /usr/local/bin/ && \
    cd

#    gcc getMaxFits.c -o getMaxFits -lcfitsio -lm && \    
#    gcc locNoiseMed.c -o locNoiseMed -lcfitsio -lm && \
#    gcc cookbook.c -o cookbook -lcfitsio -lm && \
    
#---------------------------------------------------------------------------
# The image will now be rebuilt without adding the sources, in order to
# reduce the size of the image.
#---------------------------------------------------------------------------
FROM ubuntu:20.04 as runner
RUN mkdir /src
COPY --from=builder /usr/local /usr/local
RUN chmod +rx /usr/local/bin/*

SHELL ["/bin/bash", "-c"]

# Set default versions. Can be overridden from the command-line
ARG LOFARSTMAN_VERSION=master
ARG DYSCO_VERSION=master
ARG IDG_VERSION=master
ARG AOFLAGGER_VERSION=master
ARG LOFARBEAM_VERSION=master
ARG EVERYBEAM_VERSION=master
ARG DP3_VERSION=master
ARG WSCLEAN_VERSION=master


# Only install run-time required packages 
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
        casacore-tools \
        libatkmm-1.6-1v5 \
        libboost-date-time1.71.0 \
        libboost-filesystem1.71.0 \
        libboost-program-options1.71.0 \
        libboost-python1.71.0 \
        libcairomm-1.0-1v5 \
        libcasa-casa4 \
        libcasa-fits4 \
        libcasa-measures4 \
        libcasa-ms4 \
        libcasa-python3-4 \
        libcasa-scimath4 \
        libcasa-tables4 \
        libcfitsio8 \
        libfftw3-double3 \
        libfftw3-single3 \
        libglibmm-2.4-1v5 \
        libgomp1 \
        libgsl23 \
        libgtkmm-3.0-1v5 \
        libhdf5-103 \
        libhdf5-cpp-103 \
        liblapack3 \
        liblua5.3-0 \
        libpangomm-1.4-1v5 \
        libpng16-16 \
        libpython3.8 \
        libsigc++-2.0-0v5 \
        libstdc++6 \
        nodejs \
        python3 \
        python3-casacore \
        python3-distutils \
        wget \
        git && \
    rm -rf /var/lib/apt/lists/*


# Install WSRT Measures (extra casacore data)
# Note: The file on the ftp site is updated daily. When warnings regarding leap
# seconds appear, ignore them or regenerate the docker image.
RUN wget -q -O /WSRT_Measures.ztar \
        ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar && \
    cd /var/lib/casacore/data && \
    tar xfz /WSRT_Measures.ztar && \
    rm /WSRT_Measures.ztar

# Some python stuff
RUN python3 -m pip install h5py pandas pyyaml astropy matplotlib scipy shapely bdsf ipython
#    cd /src && \
#   git clone https://github.com/lofar-astron/PyBDSF.git && \
#  cd /src/PyBDSF && \
#    python3 -m pip install . && \
#    cd    
    
# AImCal
ADD imcal.py /opt/imcal.py
ADD cluster.py /opt/cluster.py
ADD nvss_cutout.py /opt/nvss_cutout.py
ADD imcal.yml /opt/imcal.yml
ADD nvss.csv.zip /opt/nvss.csv.zip
RUN ln -s /opt/imcal.py /usr/local/bin/imcal.py


# Try to run the compiled tools to make sure they run without
# a problem (e.g. no missing libraries).

#RUN aoflagger --version && \
#    DP3 --version && \
#    wsclean --version

# Clean
#RUN rm -rf /software/*





