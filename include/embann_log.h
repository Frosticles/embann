// SPDX-License-Identifier: GPL-2.0-only
/*
    embann_log.h - EMbedded Backpropogating Artificial Neural Network.
    Copyright Peter Frost 2019
*/

#ifndef Embann_log_h
#define Embann_log_h

#ifdef CONFIG_LOG_COLORS
#define LOG_COLOR_BLACK   "30"
#define LOG_COLOR_RED     "31"
#define LOG_COLOR_GREEN   "32"
#define LOG_COLOR_BROWN   "33"
#define LOG_COLOR_BLUE    "34"
#define LOG_COLOR_PURPLE  "35"
#define LOG_COLOR_CYAN    "36"
// cppcheck-suppress misra-c2012-20.7 
#define LOG_COLOR(COLOR)  "\033[0;" COLOR "m"
// cppcheck-suppress misra-c2012-20.7
#define LOG_BOLD(COLOR)   "\033[1;" COLOR "m"
#define LOG_COLOR_E       LOG_COLOR(LOG_COLOR_RED)
#define LOG_COLOR_W       LOG_COLOR(LOG_COLOR_BROWN)
#define LOG_COLOR_I       LOG_COLOR(LOG_COLOR_GREEN)
#define LOG_COLOR_D       LOG_COLOR(LOG_COLOR_CYAN)
#define LOG_COLOR_V       LOG_COLOR(LOG_COLOR_PURPLE)
#define LOG_RESET_COLOR   "\033[0m"
#else //CONFIG_LOG_COLORS
#define LOG_COLOR_E
#define LOG_COLOR_W
#define LOG_COLOR_I
#define LOG_COLOR_D
#define LOG_COLOR_V
#define LOG_RESET_COLOR
#endif //CONFIG_LOG_COLORS

#define LOG_FORMAT(letter, tag, format)  LOG_COLOR_ ## letter #letter ": " tag " - " format LOG_RESET_COLOR "\n"

#define PRINT_CHECK(a) if ((a) < 0) *(embann_getErrno()) = EIO

#define EMBANN_LOG_LEVEL(level, tag, format, ...) do {                     \
        if      (level==EMBANN_LOG_ERROR)     { PRINT_CHECK(printf(LOG_FORMAT(E, tag, format), ##__VA_ARGS__)); } \
        else if (level==EMBANN_LOG_WARN)      { PRINT_CHECK(printf(LOG_FORMAT(W, tag, format), ##__VA_ARGS__)); } \
        else if (level==EMBANN_LOG_DEBUG)     { PRINT_CHECK(printf(LOG_FORMAT(D, tag, format), ##__VA_ARGS__)); } \
        else if (level==EMBANN_LOG_VERBOSE)   { PRINT_CHECK(printf(LOG_FORMAT(V, tag, format), ##__VA_ARGS__)); } \
        else                                  { PRINT_CHECK(printf(LOG_FORMAT(I, tag, format), ##__VA_ARGS__)); } \
    } while(0)

#define EMBANN_LOG_LEVEL_LOCAL(level, tag, format, ...) do {               \
        if (CONFIG_LOG_DEFAULT_LEVEL >= level) EMBANN_LOG_LEVEL(level, tag, format, ##__VA_ARGS__); \
    } while(0)


#define EMBANN_LOGE(tag, format, ...) EMBANN_LOG_LEVEL_LOCAL(EMBANN_LOG_ERROR,   tag, format, ##__VA_ARGS__)
#define EMBANN_LOGW(tag, format, ...) EMBANN_LOG_LEVEL_LOCAL(EMBANN_LOG_WARN,    tag, format, ##__VA_ARGS__)
#define EMBANN_LOGI(tag, format, ...) EMBANN_LOG_LEVEL_LOCAL(EMBANN_LOG_INFO,    tag, format, ##__VA_ARGS__)
#define EMBANN_LOGD(tag, format, ...) EMBANN_LOG_LEVEL_LOCAL(EMBANN_LOG_DEBUG,   tag, format, ##__VA_ARGS__)
#define EMBANN_LOGV(tag, format, ...) EMBANN_LOG_LEVEL_LOCAL(EMBANN_LOG_VERBOSE, tag, format, ##__VA_ARGS__)

enum {
    EMBANN_LOG_NONE,       /*!< No log output */
    EMBANN_LOG_ERROR,      /*!< Critical errors, software module can not recover on its own */
    EMBANN_LOG_WARN,       /*!< Error conditions from which recovery measures have been taken */
    EMBANN_LOG_INFO,       /*!< Information messages which describe normal flow of events */
    EMBANN_LOG_DEBUG,      /*!< Extra information which is not necessary for normal use (values, pointers, sizes, etc). */
    EMBANN_LOG_VERBOSE     /*!< Bigger chunks of debugging information, or frequent messages which can potentially flood the output. */
};

#endif //Embann_log_h
