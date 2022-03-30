#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
  protgram to compute the 
*/
#include "fitsio.h"



int main(int argc, char **argv) {

  fitsfile *inImage1;      /* pointer to input image */
  fitsfile *inImage2;      /* pointer to input image */

  fitsfile *outMask;      /* pointer to output mask */

  int status, morekeys, hdutype;
  int  i, j, ind;
  int naxis;
  int xlen, ylen;
    
  int maxdim;
  long  nelements, index;
  long naxes[10];
  int bitpix = SHORT_IMG ; /* output image is ints; BITPIX = -16*/
    
  float *imArray1;
  float *imArray2;
  int   *maskArray;
     
  float clipVal1;
  float clipVal2;
  int nullval ;
  int anynull;
   
  status = 0;
  maxdim = 10;

  if (argc != 6) {
    puts("makeCombMaskFits image1 image2 outmask clip1 clip2 ");
    exit(1);
  }
  clipVal1 = atof(argv[4]);
  clipVal2 = atof(argv[5]);
 
          
  /* open the existing image */
  fits_open_file(&inImage1, argv[1], READONLY, &status) ;
  fits_open_file(&inImage2, argv[2], READONLY, &status) ;

    
  /* get number of dimensions */
  fits_get_img_dim(inImage1, &naxis, &status);
  /* get size of image. assume x and y are axes 0 and 1 */  
  fits_get_img_size(inImage1, maxdim, naxes, &status);

  /* number of pixels in image */
  nelements = naxes[0]*naxes[1];
  xlen = naxes[0];
  ylen = naxes[1];

  /* allocare memeory for the arrays */
  imArray1 = (float *)malloc(nelements*sizeof(float));
  imArray2 = (float *)malloc(nelements*sizeof(float));
  maskArray = (int *)malloc(nelements*sizeof(int));

  /* don't check for null values */
  nullval = 0;

  /* read the images */
  fits_read_img(inImage1, TFLOAT, 1, nelements, &nullval,
                imArray1, &anynull, &status) ;
  fits_read_img(inImage2, TFLOAT, 1, nelements, &nullval,
                imArray2, &anynull, &status) ;

  /* loop over image */
  for (i = 0; i < xlen; i++) {
    for (j = 0; j < ylen;  j++) {
      index = i + xlen*(j) ;
      if (imArray1[index] > clipVal1 || imArray2[index] > clipVal2) {
         maskArray[index]= 1;
      } else {
        maskArray[index]= 0;
      }
    }
  }

  /* delete output mask file */
  remove(argv[3]);
  /* create output mask file, copy header and write maskfits
     file */
  fits_create_file(&outMask, argv[3], &status);
  fits_copy_hdu(inImage1, outMask, 0, &status);
  fits_write_img(outMask,TINT,1,nelements,maskArray,&status);

  /* close all files */
  fits_close_file(inImage1,&status);
  fits_close_file(inImage2,&status);
  fits_close_file(outMask,&status);




}
