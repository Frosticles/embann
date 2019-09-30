/*
  Embann.h - ARDuino Backpropogating Artificial Neural Network.
  Created by Peter Frost, February 9, 2017.
  Released into the public domain.
*/
#ifndef Embann_h
#define Embann_h

#ifdef ARDUINO
#include "Arduino.h"
#else// ARDUINO
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#include <string.h>
#define PI 3.14159
#endif// ARDUINO

/* Random float between -1 and 1 */
#define RAND_WEIGHT() (((float)random() / (RAND_MAX / 2)) - 1)
/* Throw an error if malloc failed */
#define CHECK_MALLOC(a) if (!a) {return ENOMEM;}
/* Calculate number of elements in an array */
#define NUM_ARRAY_ELEMENTS(a) (sizeof(a) / sizeof(a[0]))
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
void embann_init(uint16_t numInputNeurons,
                 uint16_t numHiddenNeurons, 
                 uint8_t numHiddenLayers,
                 uint16_t numOutputNeurons);
uint8_t embann_inputLayer(void);
void embann_sumAndSquash(wNeuron_t* Input[], wNeuron_t* Output[], uint16_t numInputs,
                           uint16_t numOutputs);
uint8_t embann_outputLayer(void);
void embann_printNetwork(void);
void embann_trainDriverInTime(float learningRate, long numSeconds, bool verbose);
void embann_trainDriverInError(float learningRate, float desiredCost, bool verbose);
void embann_train(uint8_t correctOutput, float learningRate);
float embann_tanhDerivative(float inputValue);
void embann_newInputRaw(uint16_t rawInputArray[], uint16_t numInputs);
void embann_errorReporting(uint8_t correctResponse);
void embann_printInputNeuronDetails(uint8_t neuronNum);
void embann_printOutputNeuronDetails(uint8_t neuronNum);
void embann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum);
void embann_benchmark(void);

#endif