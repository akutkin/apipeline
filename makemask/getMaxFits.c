#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
  protgram to compute the 
*/
#include <fitsio.h>



int main(int argc, char **argv) {

  fitsfile *inImage;      /* pointer to input image */


  int status;
  int ii;
    
  int maxdim;
  long  nelements;
  long naxes[10];
  int bitpix = FLOAT_IMG ; /* output image is floats; BITPIX = -32*/
    
  float *imArray;
    
  int nullval ;
  int anynull;
  float sig;
  
  status = 0;
  maxdim = 10;

        
  /* open the existing image */
  fits_open_file(&inImage, argv[1], READONLY, &status) ;
    
  /* get size of image. assume x and y are axes 0 and 1 */  
  fits_get_img_size(inImage, maxdim, naxes, &status);

  /* number of pixels in image */
  nelements = naxes[0]*naxes[1];
  /* allocate memory for image */
  imArray = (float *)malloc(nelements*sizeof(float));
  /* set maimum to something low */
  sig = -1100100101001.0;

  /* don't check for null values */
  nullval = 0;

  /* read the image */
  fits_read_img(inImage, TFLOAT, 1, nelements, &nullval,
                imArray, &anynull, &status) ;

  /* find maximum value */
  for (ii = 0; ii < nelements; ii++) {
    if (imArray[ii] > sig) {
      sig = imArray[ii];
    }
  }

  /* print maximum */
  printf("%12.8f\n",sig);
    
  /* close all files */
  fits_close_file(inImage,&status);




}
