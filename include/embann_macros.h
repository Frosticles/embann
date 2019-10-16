// SPDX-License-Identifier: GPL-2.0-only
/*
    Embann_macros.h - EMbedded Backpropogating Artificial Neural Network.
    Copyright Peter Frost 2019
*/

#ifndef Embann_macros_h
#define Embann_macros_h

#ifndef EOK
#define EOK 0 /* Provides more clarity than just writing return 0; */
#endif// EOK



/* Random float between -1 and 1 */
#define RAND_WEIGHT() (((float)random() / (RAND_MAX / 2)) - 1)
/* Calculate number of elements in an array */
#define NUM_ARRAY_ELEMENTS(x) (sizeof(x) / sizeof((x)[0]))





#ifdef CONFIG_ERROR_CHECK_SET_ERRNO
    #define EMBANN_ERROR_CHECK(x) *(embann_getErrno()) = (x);     \
        if (*(embann_getErrno()) != EOK) {                        \
            printf("ERROR: %d", *(embann_getErrno()));            \
        }

#elif defined(CONFIG_ERROR_CHECK_ABORT)
    #define EMBANN_ERROR_CHECK(x) *(embann_getErrno()) = (x);     \
            if (*(embann_getErrno()) != EOK) {                    \
                abort();                                          \
            }                                                     \

#elif defined(CONFIG_ERROR_CHECK_RETURN)
    #define EMBANN_ERROR_CHECK(x) *(embann_getErrno()) = (x); return *(embann_getErrno());

#endif




#ifdef CONFIG_MALLOC_CHECK_ABORT
    #define EMBANN_MALLOC_CHECK(x) if (!(x)) {*(embann_getErrno()) = ENOMEM; abort();}

#elif defined(CONFIG_MALLOC_CHECK_RETURN)
    #define EMBANN_MALLOC_CHECK(x) if (!(x)) {*(embann_getErrno()) = ENOMEM; return ENOMEM;}

#endif




/* Macro to stringify parameters in macros */
#define TOSTRING(x) #x
#define STRINGIFY(x) TOSTRING(x)




#if __GNUC__ >= 3
    #define MAX_ALIGNMENT __attribute__ ((aligned(__BIGGEST_ALIGNMENT__)))
    #define WEAK_FUNCTION __attribute__((weak))
#else
    #define MAX_ALIGNMENT
    #define WEAK_FUNCTION
#endif



#endif // Embann_macros_h
