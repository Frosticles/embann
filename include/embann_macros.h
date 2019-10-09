#ifndef Embann_macros_h
#define Embann_macros_h

#ifndef EOK
#define EOK 0 /* Provides more clarity than just writing return 0; */
#endif// EOK

/* Random float between -1 and 1 */
#define RAND_WEIGHT() (((float)random() / (RAND_MAX / 2)) - 1)
/* Throw an error if malloc failed */
#define CHECK_MALLOC(a) if (!(a)) embann_errno = ENOMEM
/* Calculate number of elements in an array */
#define NUM_ARRAY_ELEMENTS(a) (sizeof(a) / sizeof((a)[0]))
/* Check error return value */
#ifdef CONFIG_ERROR_CHECK_SET_ERRNO
#define EMBANN_ERROR_CHECK(x) embann_errno = (x)
#elif defined(CONFIG_ERROR_CHECK_ABORT)
#define EMBANN_ERROR_CHECK(x) do {                                  \
        embann_errno = (x);                                         \
        if (embann_errno != EOK) {                                  \
            abort();                                                \
        }                                                           \
    } while(0)
#endif


#if __GNUC__ >= 3
    #define MAX_ALIGNMENT __attribute__ ((aligned(__BIGGEST_ALIGNMENT__)))
    #define WEAK_FUNCTION __attribute__((weak))
#else
    #define MAX_ALIGNMENT
    #define WEAK_FUNCTION
#endif

#endif // Embann_macros_h
