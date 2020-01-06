/*
 * Tweaked from a combination of the quirks.h file from libsodium 
 * and win32_support.h from psycopg2
 */
#ifndef Embann_quirks_h
#define Embann_quirks_h

#include <stdlib.h>


//C++Builder defines a "random" macro
#undef random



#ifdef __EMSCRIPTEN__
int strcmp(const char *s1, const char *s2);
#endif




#ifdef _WIN32

static inline void srandom(unsigned seed)
{
    srand(seed);
}

static inline long random(void)
{
    return (long) rand();
}

#ifndef __MINGW32__
int gettimeofday(struct timeval * tp, void * tzp);
void timeradd(struct timeval *a, struct timeval *b, struct timeval *c);
#endif // __MINGW32__

void timersub(struct timeval *a, struct timeval *b, struct timeval *c);

#ifndef timercmp
#define timercmp(a, b, cmp)          \
  (((a)->tv_sec == (b)->tv_sec) ?    \
   ((a)->tv_usec cmp (b)->tv_usec) : \
   ((a)->tv_sec  cmp (b)->tv_sec))
#endif // timercmp

#endif // _WIN32





#endif // Embann_quirks_h