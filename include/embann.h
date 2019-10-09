// SPDX-License-Identifier: GPL-2.0-only
/*
    embann.h - EMbedded Backpropogating Artificial Neural Network.
    Copyright Peter Frost 2019
*/

#ifndef Embann_h
#define Embann_h

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

#include "embann_config.h"
#include "embann_data_types.h"
#include "embann_macros.h"


/* errno value specifically for internal embann errors */
static int embann_errno = EOK;


// TODO, could put in a linear approximation for these
typedef enum {
    LINEAR_ACTIVATION,
    TAN_H_ACTIVATION,
    LOGISTIC_ACTIVATION,
    RELU_ACTIVATION
} activationFunction_t;

typedef struct  {
    weight_t weight;
    bias_t bias;
} neuronParams_t;

typedef struct
{
    activation_t activation;
    neuronParams_t* params[];
} wNeuron_t;

typedef struct
{
    float activation;
} uNeuron_t;

typedef struct
{
    wNeuron_t* forgetGate;
    wNeuron_t* inputGate;
    wNeuron_t* outputGate;
    wNeuron_t* cell;
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
    inputLayer_t* inputLayer;
    outputLayer_t* outputLayer;
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
int embann_outputLayer(uint16_t* networkResponse);
int embann_inputLayer(uint16_t* networkResponse);
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

#endif // Embann_h
