// File       : common.h
// Description: Common headers
// Copyright 2020 ETH Zurich. All Rights Reserved.
#ifndef COMMON_H_PUN0BOLA
#define COMMON_H_PUN0BOLA

#ifndef _HTILE_
#define _HTILE_ 128 // default horizontal tile size
#endif /* _HTILE_ */

#ifndef _VTILE_
#define _VTILE_ 16 // default vertical tile size
#endif /* _VTILE_ */

#ifdef _SINGLE_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif /* _SINGLE_PRECISION_ */

#endif /* COMMON_H_PUN0BOLA */
