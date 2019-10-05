/*
  Embann.h - ARDuino Backpropogating Artificial Neural Network.
  Created by Peter Frost, February 9, 2017.
  Released into the public domain.
*/
#ifndef Embann_h
#define Embann_h

#include "config.h"

#ifdef ARDUINO
#include "Arduino.h"
#else// ARDUINO
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
// Deviation from MISRA C2012 21.6 for printf() used in logging
// cppcheck-suppress misra-c2012-21.6
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <errno.h>
#include <string.h>
#define PI 3.14159
#endif// ARDUINO

#ifndef EOK
/* Provides more clarity than just writing return 0; */
#define EOK 0 
#endif// EOK

/* errno value specifically for internal embann errors */
static int embann_errno = EOK;

/* Random float between -1 and 1 */
#define RAND_WEIGHT() (((float)random() / (RAND_MAX / 2)) - 1)
/* Throw an error if malloc failed */
#define CHECK_MALLOC(a) if (!(a)) embann_errno = ENOMEM
/* Calculate number of elements in an array */
#define NUM_ARRAY_ELEMENTS(a) (sizeof(a) / sizeof((a)[0]))

#ifdef CONFIG_ERROR_CHECK_SET_ERRNO
#define EMBANN_ERROR_CHECK(x) embann_errno = (x)
#elif defined(CONFIG_ERROR_CHECK_ABORT)
#define EMBANN_ERROR_CHECK(x) do {                                     \
        embann_errno = (x);                                         \
        if (embann_errno != EOK) {                                  \
            abort();                                                \
        }                                                           \
    } while(0)
#endif

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

#define PRINT_CHECK(a) if ((a) < 0) embann_errno = EIO

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






#if __GNUC__ >= 3
    #define MAX_ALIGNMENT __attribute__ ((aligned(__BIGGEST_ALIGNMENT__)))
    #define WEAK_FUNCTION __attribute__((weak))
#else
    #define MAX_ALIGNMENT
    #define WEAK_FUNCTION
#endif


typedef enum {
    LINEAR_ACTIVATION,
    TAN_H_ACTIVATION,
    LOGISTIC_ACTIVATION,
    RELU_ACTIVATION
} activationFunction_t;

typedef struct  {
    float weight;
    float bias;
} neuronParams_t;

typedef struct
{
    float activation;
    neuronParams_t* params[];
} wNeuron_t;

typedef struct
{
    float activation;
} uNeuron_t;

typedef struct
{
    wNeuron_t forgetGate;
    wNeuron_t inputGate;
    wNeuron_t outputGate;
    wNeuron_t cell;
    float activation;
} lstmCell_t;

typedef struct trainingData
{
    uint16_t correctResponse;
    uint32_t length;
    struct trainingData* prev;
    struct trainingData* next;
    uint8_t data[];
} trainingData_t;

typedef struct 
{
    trainingData_t* head;
    trainingData_t* tail;
    uint32_t numEntries;
} trainingDataCollection_t;

typedef struct
{
    uint16_t numRawInputs;
    uint16_t maxInput;
    uint16_t* rawInputs;
    uint16_t* groupThresholds;
    uint16_t* groupTotal;
} downscaler_t;
typedef struct
{
    uint16_t numNeurons;
    uNeuron_t* neuron[];
} inputLayer_t;

typedef struct
{
    uint16_t numNeurons;
    wNeuron_t* neuron[];
} hiddenLayer_t;

typedef struct
{
    uint16_t numNeurons;
    wNeuron_t* neuron[];
} outputLayer_t;

typedef struct
{
    uint8_t numLayers;
    uint8_t numHiddenLayers;
    uint16_t networkResponse;
} networkProperties_t;


typedef struct
{
    networkProperties_t properties;
    inputLayer_t inputLayer;
    outputLayer_t outputLayer;
    hiddenLayer_t hiddenLayer[];
} network_t;


/*
  Combine these init statements with some pointer magic and null checking
*/
int embann_init(uint16_t numInputNeurons,
                 uint16_t numHiddenNeurons, 
                 uint8_t numHiddenLayers,
                 uint16_t numOutputNeurons);
int embann_sumAndSquash(wNeuron_t* Input[], wNeuron_t* Output[], uint16_t numInputs,
                           uint16_t numOutputs);
int embann_sumAndSquashInput(uNeuron_t* Input[], wNeuron_t* Output[], uint16_t numInputs,
                           uint16_t numOutputs);
uint8_t embann_outputLayer(void);
int embann_printNetwork(void);
int embann_trainDriverInTime(float learningRate, uint32_t numSeconds, bool verbose);
int embann_trainDriverInError(float learningRate, float desiredCost, bool verbose);
int embann_train(uint8_t correctOutput, float learningRate);
int embann_tanhDerivative(float inputValue, float* outputValue);
int embann_newInputRaw(uint16_t rawInputArray[], uint16_t numInputs);
int embann_errorReporting(uint8_t correctResponse);
int embann_printInputNeuronDetails(uint8_t neuronNum);
int embann_printOutputNeuronDetails(uint8_t neuronNum);
int embann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum);
int embann_benchmark(void);

#endif