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



/* Calculate number of elements in an array */
#define NUM_ARRAY_ELEMENTS(x) (sizeof(x) / sizeof((x)[0]))


#ifdef WEIGHT_IS_FLOAT
    #define WEIGHT_PRINT STRINGIFY(.3f)
    /* Random float between -1 and 1 */
    #define RAND_WEIGHT() (((float)random() / (RAND_MAX / 2)) - 1)
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
    #define WEIGHT_PRINT STRINGIFY(d)
    #define RAND_WEIGHT() ((random() % MAX_WEIGHT) - MIN_WEIGHT) // TODO this but for each type
#endif

#ifdef ACTIVATION_IS_FLOAT
    #define ACTIVATION_PRINT STRINGIFY(.3f)
    /* Random float between -1 and 1 */
    #define RAND_ACTIVATION() (((float)random() / (RAND_MAX / 2)) - 1)
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
    #define ACTIVATION_PRINT STRINGIFY(d)
    #define RAND_ACTIVATION() (random() % INT8_MAX) // TODO this but for each type
#endif

#ifdef BIAS_IS_FLOAT
    #define BIAS_PRINT STRINGIFY(.3f)
    /* Random float between -1 and 1 */
    #define RAND_BIAS() (((float)random() / (RAND_MAX / 2)) - 1)
#elif defined(BIAS_IS_SIGNED) || defined(BIAS_IS_UNSIGNED)
    #define BIAS_PRINT STRINGIFY(d)
    #define RAND_BIAS() (random() % INT8_MAX) // TODO this but for each type
#endif


#ifdef CONFIG_ERROR_CHECK_SET_ERRNO
    #define EMBANN_ERROR_CHECK(x) *(embann_getErrno()) = (x);     \
        if (*(embann_getErrno()) != EOK) {                        \
            printf("ERROR: %d", *(embann_getErrno()));            \
        }

#elif defined(CONFIG_ERROR_CHECK_ABORT)
    #define EMBANN_ERROR_CHECK(x) *(embann_getErrno()) = (x);     \
            if (*(embann_getErrno()) != EOK) {                    \
                abort();                                          \
            }

#elif defined(CONFIG_ERROR_CHECK_RETURN)
    #define EMBANN_ERROR_CHECK(x) *(embann_getErrno()) = (x); return *(embann_getErrno());

#elif defined(CONFIG_ERROR_CHECK_LOG)
    #define EMBANN_ERROR_CHECK(x) *(embann_getErrno()) = (x); EMBANN_LOGI(TAG, "Returned: %d", *(embann_getErrno())); 

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
