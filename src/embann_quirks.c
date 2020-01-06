/*
 * Tweaked from a combination of the quirks.h file from libsodium 
 * and win32_support.c from psycopg2
 */
#include "embann.h"

#ifdef __EMSCRIPTEN__
int strcmp(const char *s1, const char *s2)
{
    while (*s1 == *s2++) {
        if (*s1++ == 0) {
            return 0;
        }
    }
    return *(unsigned char *) s1 - *(unsigned char *) --s2;
}
#endif


#ifdef _WIN32

#ifndef __MINGW32__
/* 
 * millisecond-precision port of gettimeofday for Win32, taken from
 * src/port/gettimeofday.c in PostgreSQL core 
 */

/* FILETIME of Jan 1 1970 00:00:00. */
static const unsigned __int64 epoch = ((unsigned __int64) 116444736000000000ULL);

/*
 * timezone information is stored outside the kernel so tzp isn't used anymore.
 *
 * Note: this function is not for Win32 high precision timing purpose. See
 * elapsed_time().
 */
int gettimeofday(struct timeval * tp, void * tzp)
{
    FILETIME	file_time;
    SYSTEMTIME	system_time;
    ULARGE_INTEGER ularge;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    ularge.LowPart = file_time.dwLowDateTime;
    ularge.HighPart = file_time.dwHighDateTime;

    tp->tv_sec = (long) ((ularge.QuadPart - epoch) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);

    return 0;
}

void timeradd(struct timeval *a, struct timeval *b, struct timeval *c)
{
  c->tv_sec = a->tv_sec + b->tv_sec;
  c->tv_usec = a->tv_usec + b->tv_usec;
  if(c->tv_usec >= 1000000L) {
    c->tv_usec -= 1000000L;
    c->tv_sec += 1;
  }
}
#endif // __MINGW32__

void timersub(struct timeval *a, struct timeval *b, struct timeval *c)
{
    c->tv_sec  = a->tv_sec  - b->tv_sec;
    c->tv_usec = a->tv_usec - b->tv_usec;
    if (c->tv_usec < 0) {
        c->tv_usec += 1000000;
        c->tv_sec  -= 1;
    }
}

#endif // _WIN32
