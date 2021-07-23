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

  fitsfile *outMask;      /* pointer to output mask */

  int status, morekeys, hdutype;
  int  i, j, ind;
  int naxis;
  int xlen, ylen;
    
  int maxdim;
  long  nelements, index;
  long naxes[10];
  int bitpix = SHORT_IMG ; /* output image is ints; BITPIX = -16*/
    
  float *imArray;
  int   *maskArray;
     
  float clipVal;
  int nullval ;
  int anynull;
   
  status = 0;
  maxdim = 10;
  clipVal = atof(argv[3]);
 
          
  /* open the existing image */
  fits_open_file(&inImage, argv[1], READONLY, &status) ;

    
  /* get number of dimensions */
  fits_get_img_dim(inImage, &naxis, &status);
  /* get size of image. assume x and y are axes 0 and 1 */  
  fits_get_img_size(inImage, maxdim, naxes, &status);

  /* number of pixels in image */
  nelements = naxes[0]*naxes[1];
  xlen = naxes[0];
  ylen = naxes[1];

  /* allocare memeory for the arrays */
  imArray = (float *)malloc(nelements*sizeof(float));
  maskArray = (int *)malloc(nelements*sizeof(int));

  /* don't check for null values */
  nullval = 0;

  /* read the image */
  fits_read_img(inImage, TFLOAT, 1, nelements, &nullval,
                imArray, &anynull, &status) ;

  /* loop over image */
  for (i = 0; i < xlen; i++) {
    for (j = 0; j < ylen;  j++) {
      index = i + xlen*(j) ;
      if (imArray[index] > clipVal) {
         maskArray[index]= 1;
      } else {
        maskArray[index]= 0;
      }
    }
  }

  /* delete output mask file */
  remove(argv[2]);
  /* create output mask file, copy header and write maskfits
     file */
  fits_create_file(&outMask, argv[2], &status);
  fits_copy_hdu(inImage, outMask, 0, &status);
  fits_write_img(outMask,TINT,1,nelements,maskArray,&status);

  /* close all files */
  fits_close_file(inImage,&status);
  fits_close_file(outMask,&status);




}
