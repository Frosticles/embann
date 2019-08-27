/*
  Ardbann.h - ARDuino Backpropogating Artificial Neural Network.
  Created by Peter Frost, February 9, 2017.
  Released into the public domain.
*/
#ifndef Ardbann_h
#define Ardbann_h

#ifdef ARDUINO
#include "Arduino.h"
#else
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#define PI 3.14159
#endif

typedef struct
{
    uint16_t numNeurons;
    uint16_t numRawInputs;
    uint16_t maxInput;
    uint16_t* rawInputs;
    uint16_t* groupThresholds;
    uint16_t* groupTotal;
    float* neurons;
} inputLayer_t;

typedef struct
{
    uint16_t numNeurons;
    uint8_t numLayers;
    float** neuronTable;
    float*** weightLayerTable;
    float** neuronBiasTable;
} hiddenLayer_t;

typedef struct
{
    char* stringArray;
    uint16_t numNeurons;
    float* neurons;
    float** weightTable;
    float* neuronBiasTable;
} outputLayer_t;

typedef struct
{
    uint16_t numLayers;
    uint16_t networkResponse;
    inputLayer_t inputLayer;
    hiddenLayer_t hiddenLayer;
    outputLayer_t outputLayer;
} network_t;

typedef struct
{
    uint16_t* samples;
    uint32_t sampleRate;
} networkSampleBuffer_t;

/*
  Combine these init statements with some pointer magic and null checking
*/
void ardbann_initWithData(  uint16_t rawInputArray[], uint16_t maxInput, char *outputArray,
                            uint16_t numInputs, uint16_t numInputNeurons,
                            uint16_t numHiddenNeurons, uint8_t numHiddenLayers,
                            uint16_t numOutputNeurons);

void ardbann_initWithoutData(   uint16_t maxInput, char *outputArray,
                                uint16_t numInputNeurons, uint16_t numHiddenNeurons,
                                uint8_t numHiddenLayers, uint16_t numOutputNeurons);

uint8_t ardbann_inputLayer(void);
void ardbann_sumAndSquash(float *Input, float *Output, float *Bias, float **Weights,
                  uint16_t numInputs, uint16_t numOutputs);
uint8_t ardbann_outputLayer(void);
void ardbann_printNetwork(void);
void ardbann_trainDriverInTime(float learningRate, bool verbose, uint8_t numTrainingSets,
                 uint8_t inputPin, uint16_t bufferSize, long numSeconds);
void ardbann_trainDriverInError(float learningRate, bool verbose, uint8_t numTrainingSets,
                 uint8_t inputPin, uint16_t bufferSize, float desiredError);
void ardbann_train(uint8_t correctOutput, float learningRate);
float ardbann_tanhDerivative(float inputValue);
void ardbann_newInputRaw(uint16_t rawInputArray[], uint16_t numInputs);
void ardbann_newInputStruct(networkSampleBuffer_t sampleBuffer, uint16_t numInputs);
void ardbann_errorReporting(uint8_t correctResponse);
void ardbann_printInputNeuronDetails(uint8_t neuronNum);
void ardbann_printOutputNeuronDetails(uint8_t neuronNum);
void ardbann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum);

#endif