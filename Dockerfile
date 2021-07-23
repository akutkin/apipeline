FROM ubuntu:20.04

RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y \
    vim \
    git \
    cmake \
    build-essential \
    g++ \
    pkg-config \
    casacore-data casacore-dev \
    libblas-dev liblapack-dev \
    libpython3-dev \
    libboost-date-time-dev libboost-test-dev \
    libboost-program-options-dev libboost-system-dev libboost-filesystem-dev \
    libcfitsio-dev \
    libfftw3-dev \
    libgtkmm-3.0-dev \
    libgsl-dev \
    libhdf5-dev \
    liblua5.3-dev \
    libopenmpi-dev \
    libpng-dev \
    pkg-config \
    python3 \
    python3-dev python3-numpy \
    python3-sphinx \
    python3-pip \
    wget

# EveryBeam
RUN \
  mkdir /software && \
  cd /software && \
  git clone https://git.astron.nl/RD/EveryBeam.git && \
  mkdir /software/EveryBeam/build && \
  cd /software/EveryBeam/build && \
  cmake ../ && \
  make install -j`nproc --all`

# Dysco
RUN \
  cd /software && \
  git clone https://github.com/aroffringa/dysco.git && \
  mkdir /software/dysco/build && \
  cd /software/dysco/build && \
  cmake ../ && \
  make install -j`nproc --all`

# IDG
RUN \
  cd /software && \
  git clone https://git.astron.nl/RD/idg.git && \
  mkdir /software/idg/build && \
  cd /software/idg/build && \
  cmake ../ && \
  make install -j`nproc --all`

# WSClean
RUN \
  cd /software && \
  git clone https://gitlab.com/aroffringa/wsclean.git && \
  mkdir /software/wsclean/build && \
  cd /software/wsclean/build && \
  cmake ../ && \
  make install -j`nproc --all` && \
  wsclean --version

# AOFlagger
RUN \
  cd /software && \
  git clone https://gitlab.com/aroffringa/aoflagger.git && \
  mkdir /software/aoflagger/build && \
  cd /software/aoflagger/build && \
  cmake ../ && \
  make install -j`nproc --all` && \
  aoflagger --version

# DP3
RUN \
  cd /software && git clone https://git.astron.nl/RD/DP3.git && \
  mkdir /software/DP3/build && \
  cd /software/DP3/build && \
  cmake ../ && \
  make install -j`nproc --all` && \
  DP3 -v

# Install WSRT Measures (extra casacore data, for integration tests)
# Note: The file on the ftp site is updated daily. When warnings regarding leap
# seconds appear, ignore them or regenerate the docker image.
RUN \
  wget -O /WSRT_Measures.ztar ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar && \
  cd /var/lib/casacore/data && \
  tar xfz /WSRT_Measures.ztar && \
  rm /WSRT_Measures.ztar

# PyBDSF
RUN \
  apt-get install -y gfortran libboost-python-dev libboost-numpy-dev python3-setuptools python3-ipython ipython3 && \
  apt-get remove -y python3-numpy && \
  python3 -m pip install astropy matplotlib numpy scipy && \
  cd /software && \
  git clone https://github.com/lofar-astron/PyBDSF.git && \
  cd /software/PyBDSF && \
  python3 -m pip install . && \
  apt-get install -y casacore-data casacore-dev python3-numpy && \
  cd

########################################################################################
# AO tools  
RUN \
    cd /software && \
    git clone https://github.com/aroffringa/modeltools.git modeltools && \
    cd modeltools && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j4 && \
    cp bbs2model cluster editmodel render /usr/local/bin/
    
# Python stuff
RUN python3 -m pip install h5py pandas pyyaml

# Tom's stuff
ADD makemask /software/makemask
RUN cd /software/makemask && \
    gcc makeMaskFits.c -o makeMaskFits -lcfitsio -lm && \
    gcc makeNoiseMapFits.c -o makeNoiseMapFits -lcfitsio -lm && \
    cp makeMaskFits makeNoiseMapFits /usr/local/bin/
    
# kvis
#RUN cd /software && \
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


# Imcal
ADD imcal.py /opt/imcal.py
ADD imcal.yml /opt/imcal.yml
RUN ln -s /opt/imcal.py /usr/local/bin/imcal.py
    
#Clean 
RUN rm -rf /software/*




    