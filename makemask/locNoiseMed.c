/*
$Id: mirRenzo.c,v 1.6 2009/04/06 10:58:50 oosterloo Exp oosterloo $ 
*/


#include <port.h>
#include <mirutil.h>
#include <miriad.h>




int main (int argc, char *argv[]) {

  char  inpFilename[80];
  char  inpFilename2[80];
  char  outFilename[80];
  char  outFilename2[80];
  char  charBuff[80];
  
  int   xlen, ylen, zlen, n, m, i, j, k;
  int   boxSize,  ind;
  long int numP;
  
  float   maskVa, mean, sig;
  float  tmpAr[1024];
  
  
  float sia, value;  
  long  index;  
  float *sigs;
  char *timeString;

  
  DataCube  *inpCube, *inpCube2, *outCube, *outCube2;
  
  
  /* init command line */
  keyini_c(argc, argv);
  
  /* get name of input file */
  keya_c("in", inpFilename, "bla");
  /* get name of input file */
  keya_c("inima", inpFilename2, "bla");

  /* get name of output file */
  keya_c("out", outFilename, "bla");
  /* get name of output file */
  keya_c("outrat", outFilename2, "bla");

  /* we have read all input */
  keyfin_c();

  inpCube = newDataCube();
  
  /* open output file */
  openDataCube(inpFilename, inpCube, "old");
  readDataCube(inpCube, FALSE);
  xlen = inpCube->axes[0];
  ylen = inpCube->axes[1];
  zlen = inpCube->axes[2];

  inpCube2 = newDataCube();
  
  /* open output file */
  openDataCube(inpFilename2, inpCube2, "old");
  readDataCube(inpCube2, FALSE);

  boxSize=10*4/3;
  
  outCube = cloneDataCube(inpCube);
  openDataCube(outFilename, outCube, "new");
  outCube2 = cloneDataCube(inpCube);
  openDataCube(outFilename2, outCube2, "new");
  
  outCube->mask = NULL;
  outCube2->mask = NULL;
  /* copy complete (?) header */
  mirHeadCopy(inpCube->fileHandle, outCube->fileHandle);
  mirHeadCopy(inpCube->fileHandle, outCube2->fileHandle);

  numP = xlen*ylen*zlen;
  
  for (i = 0; i < numP; i++) {
    outCube->data[i]= 0.0;
    outCube2->data[i]= 0.0;
  }


  numP = (2*boxSize+1)*(2*boxSize+1);
  for (i = 0; i < zlen; i++) {
    for (j = boxSize; j < xlen-boxSize; j=j+5) {
      for (k = boxSize; k < ylen-boxSize; k= k+5) {

        ind = 0;
        for (m = -boxSize; m <= boxSize; m++) {
          for (n = -boxSize; n <= boxSize; n++) {
            index = j+n + xlen*(k+m) + xlen*ylen*i;
            tmpAr[ind] = fabs(inpCube->data[index]);
            ind++;
          }
        }
        mirRealSort(numP, tmpAr);
        
        sig = tmpAr[numP/2]*1.4826;
	for (m = -2; m < 3; m++) {
          for (n = -2; n < 3; n++) {
            index = j+m + xlen*(k+n) + xlen*ylen*(i);
            outCube->data[index]= sig;
            if (sig > 0.0) {
              outCube2->data[index]= inpCube2->data[index]/sig;
            }
	  }
	}
      }
    }
  }



  writeDataCube(outCube);
  writeDataCube(outCube2);
  
  /* sign off */
  closeDataCube(inpCube);
  closeDataCube(inpCube2);
  closeDataCube(outCube);
  closeDataCube(outCube2);
  
  return 0;
  
}

