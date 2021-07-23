#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
  protgram to compute the 
*/
#include <fitsio.h>

int comp(const void *a,const void *b) {
float *x = (float *) a;
float *y = (float *) b;
// return *x - *y; // this is WRONG...
if (*x < *y) return -1;
else if (*x > *y) return 1; return 0;
}

int main(int argc, char **argv) {

  fitsfile *inImage;      /* pointer to input image */
  fitsfile *inRes;        /* pointer to input residual */

  fitsfile *outNoise;     /* pointer to output noise image */
  fitsfile *outSN;        /* pointer to output S/N image */

  int status, morekeys, hdutype;
  int ii, jj,  numP, i, j, n, m, ind;
  int naxis;
  int xlen, ylen;
    
  int maxdim;
  long  nelements, index;;
  long naxes[10];
  int bitpix = FLOAT_IMG ; /* output image is floats; BITPIX = -32*/
    
  float *imArray, *resArray, *noiseArray, *snArray;
  float tmpAr[1024];
    
  int boxsize;
  int nullval ;
  int anynull;
  float sig;
  
  status = 0;
  maxdim = 10;
  boxsize = 13;
        
  /* open the existing image */
  fits_open_file(&inImage, argv[1], READONLY, &status) ;
  /* open the residual image */
  fits_open_file(&inRes, argv[2], READONLY, &status) ;
    
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
  resArray = (float *)malloc(nelements*sizeof(float));
  noiseArray = (float *)malloc(nelements*sizeof(float));
  snArray = (float *)malloc(nelements*sizeof(float));

  /* don't check for null values */
  nullval = 0;

  /* read the image */
  fits_read_img(inImage, TFLOAT, 1, nelements, &nullval,
                imArray, &anynull, &status) ;
  /* read the residual */
  fits_read_img(inRes, TFLOAT, 1, nelements, &nullval,
               resArray, &anynull, &status) ;

  /* set outpit noise array to zero */
  for (ii = 0; ii < nelements; ii++) {
    noiseArray[ii] = 0.0;
  }

  /* number of pixels in the box for which we compute the median */
  numP = (2*boxsize+1)*(2*boxsize+1);
  /* compute index that will point to the median */
  numP = numP/2;

  /* loop over image in steps of 5 pixels */
  for (i = boxsize; i < xlen-boxsize; i=i+5) {
    for (j = boxsize; j < ylen-boxsize; j= j+5) {

      /* get the median for the current box */

      /* reset index */
      ind = 0;
      /* loop over box, putting absolute value of the image value in temp array */
      for (m = -boxsize; m <= boxsize; m++) {
        for (n = -boxsize; n <= boxsize; n++) {
          index = i+n + xlen*(j+m);
          tmpAr[ind] = fabs(resArray[index]);
          ind++;
        }
      }

      /* sort the temp array and take the central value */
      qsort(tmpAr,ind,sizeof(float),comp);
      /* multiply by 1.4826 to turn the median absolute value into sigma */
      sig = tmpAr[numP]*1.4826;
      /* and put this value in the output noise image */
      for (m = -2; m < 3; m++) {
        for (n = -2; n < 3; n++) {
          index = i+m + xlen*(j+n) ;
          noiseArray[index]= sig;
          /* if not zero, compute the S/N, i.e. the image value / noise value */
          if (sig > 0.0) {
            snArray[index]= imArray[index]/sig;
          }

        }
      }
    }
  }

  /* delete output noise file */
  remove(argv[3]);
  /* create output noise file, copy header and write noise values into fits
     file */
  fits_create_file(&outNoise, argv[3], &status);
  fits_copy_hdu(inRes, outNoise, 0, &status);
  fits_write_img(outNoise,TFLOAT,1,nelements,noiseArray,&status);

  /* same as above, but now for S/N image */
  remove(argv[4]);
  
  fits_create_file(&outSN, argv[4], &status);
  fits_copy_hdu(inRes, outSN, 0, &status);
  fits_write_img(outSN,TFLOAT,1,nelements,snArray,&status);

  /* close all files */
  fits_close_file(inImage,&status);
  fits_close_file(inRes,&status);
  fits_close_file(outNoise,&status);
  fits_close_file(outSN,&status);



}
