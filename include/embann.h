// SPDX-License-Identifier: GPL-2.0-only
/*
    embann.h - EMbedded Backpropogating Artificial Neural Network.
    Copyright Peter Frost 2019
*/

#ifndef Embann_h
#define Embann_h

#define TEST_BUILD // For now I'm always doing test builds but leave this for makefile later

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



int embann_init(numInputs_t numInputNeurons,
                numHiddenNeurons_t numHiddenNeurons, 
                numLayers_t numHiddenLayers,
                numOutputs_t numOutputNeurons);
int embann_initInputLayer(numInputs_t numInputNeurons);
int embann_initHiddenLayer(numHiddenNeurons_t numHiddenNeurons,
                            numLayers_t numHiddenLayers,
                            numInputs_t numInputNeurons);
int embann_initOutputLayer(numOutputs_t numOutputNeurons,
                            numHiddenNeurons_t numHiddenNeurons);
int embann_sumAndSquash(wNeuron_t* Input[], wNeuron_t* Output[], numInputs_t numInputs,
                           numOutputs_t numOutputs);
int embann_sumAndSquashInput(uNeuron_t* Input[], wNeuron_t* Output[], numInputs_t numInputs,
                           numOutputs_t numOutputs);
int embann_calculateNetworkResponse(void);
int embann_forwardPropagate(void);
int embann_printNetwork(void);
int embann_trainDriverInTime(activation_t learningRate, uint32_t numSeconds, bool verbose);
int embann_trainDriverInError(activation_t learningRate, activation_t desiredCost, bool verbose);
int embann_train(numOutputs_t correctOutput, activation_t learningRate);
int embann_tanhDerivative(activation_t inputValue, weight_t* outputValue);
int embann_errorReporting(numOutputs_t correctResponse);
int embann_printInputNeuronDetails(numInputs_t neuronNum);
int embann_printOutputNeuronDetails(numOutputs_t neuronNum);
int embann_printHiddenNeuronDetails(numLayers_t layerNum, numHiddenNeurons_t neuronNum);
int embann_benchmark(void);
int embann_inputRaw(activation_t data[]);
int embann_inputMinMaxScale(activation_t data[], activation_t min, activation_t max);
int embann_inputStandardizeScale(activation_t data[], float mean, float stdDev);
int embann_getTrainingDataMean(float* mean);
int embann_getTrainingDataStdDev(float* stdDev);
int embann_getTrainingDataMax(activation_t* max);
int embann_getTrainingDataMin(activation_t* min);
int embann_addTrainingData(activation_t data[], uint32_t length, numOutputs_t correctResponse);
int embann_copyTrainingData(activation_t data[], uint32_t length, numOutputs_t correctResponse);
int embann_shuffleTrainingData(void);
network_t* embann_getNetwork(void);
int embann_setNetwork(network_t* newNetwork);
int* embann_getErrno(void);
trainingDataCollection_t* embann_getDataCollection(void);


#ifndef ARDUINO
#ifdef _LARGE_TIME_API
static inline uint64_t millis(void)
{
    struct timeval64 time;
    gettimeofday(&time, NULL);
    return (uint64_t) round(time.tv_usec / 1000);
}
#else
static inline uint32_t millis(void)
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (uint32_t) roundf(time.tv_usec / 1000);
}
#endif // _LARGE_TIME_API
#endif // ARDUINO

#endif // Embann_h
