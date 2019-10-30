// SPDX-License-Identifier: GPL-2.0-only
/*
    Embann_data_types.h - EMbedded Backpropogating Artificial Neural Network.
    Copyright Peter Frost 2019
*/

#ifndef Embann_data_types_h
#define Embann_data_types_h


/*
    Types for use with Intel VPDPBUSD Instruction: 
    https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training

        typedef uint8_t activation_t;
        typedef int32_t bias_t;
        typedef int8_t weight_t;

        s32Accum += ((u8Act[0] * s8Wei[0]) + (u8Act[1] * s8Wei[1]) + (u8Act[2] * s8Wei[2]) + (u8Act[3] * s8Wei[3])) + u32Bias[0]
        ... Repeated 15 more times ...

    Types for use with Xtensa MAC16 Instruction:
    https://iis-people.ee.ethz.ch/~gmichi/asocd/exercises/ex_05.pdf

        typedef int16_t activation_t;
        typedef int32_t bias_t;
        typedef int16_t weight_t;

        s32Accum += ((s16Act * s16Wei) + u32Bias
*/

#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT8
typedef int8_t activation_t;
#define ACTIVATION_IS_SIGNED
#define MAX_ACTIVATION __INT8_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT16
typedef int16_t activation_t;
#define ACTIVATION_IS_SIGNED
#define MAX_ACTIVATION __INT16_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT32
typedef int32_t activation_t;
#define ACTIVATION_IS_SIGNED
#define MAX_ACTIVATION __INT32_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT64
typedef int64_t activation_t;
#define ACTIVATION_IS_SIGNED
#define MAX_ACTIVATION __INT64_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT8
typedef uint8_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION __UINT8_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT16
typedef uint16_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION __UINT16_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT32
typedef uint32_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION __UINT32_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT64
typedef uint64_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION __UINT64_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_FLOAT
typedef float activation_t;
#define ACTIVATION_IS_FLOAT
#define MAX_ACTIVATION __FLT_MAX__
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_DOUBLE
typedef double activation_t;
#define ACTIVATION_IS_FLOAT
#define MAX_ACTIVATION __DBL_MAX__
#endif

#ifdef CONFIG_BIAS_DATA_TYPE_INT8
typedef int8_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS __INT8_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT16
typedef int16_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS __INT16_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT32
typedef int32_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS __INT32_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT64
typedef int64_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS __INT64_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT8
typedef uint8_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS __UINT8_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT16
typedef uint16_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS __UINT16_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT32
typedef uint32_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS __UINT32_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT64
typedef uint64_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS __UINT64_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_FLOAT
typedef float bias_t;
#define BIAS_IS_FLOAT
#define MAX_BIAS __FLT_MAX__
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_DOUBLE
typedef double bias_t;
#define BIAS_IS_FLOAT
#define MAX_BIAS __DBL_MAX__
#endif

#ifdef CONFIG_WEIGHT_DATA_TYPE_INT8
typedef int8_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT __INT8_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT16
typedef int16_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT __INT16_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT32
typedef int32_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT __INT32_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT64
typedef int64_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT __INT64_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT8
typedef uint8_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT __UINT8_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT16
typedef uint16_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT __UINT16_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT32
typedef uint32_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT __UINT32_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT64
typedef uint64_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT __UINT64_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_FLOAT
typedef float weight_t;
#define WEIGHT_IS_FLOAT
#define MAX_WEIGHT __FLT_MAX__
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_DOUBLE
typedef double weight_t;
#define WEIGHT_IS_FLOAT
#define MAX_WEIGHT __DBL_MAX__
#endif



#ifdef CONFIG_NUM_OUTPUTS_DATA_TYPE_UINT8
typedef uint8_t numOutputs_t;
#endif
#ifdef CONFIG_NUM_OUTPUTS_DATA_TYPE_UINT16
typedef uint16_t numOutputs_t;
#endif
#ifdef CONFIG_NUM_OUTPUTS_DATA_TYPE_UINT32
typedef uint32_t numOutputs_t;
#endif
#ifdef CONFIG_NUM_OUTPUTS_DATA_TYPE_UINT64
typedef uint64_t numOutputs_t;
#endif


#ifdef CONFIG_NUM_INPUTS_DATA_TYPE_UINT8
typedef uint8_t numInputs_t;
#endif
#ifdef CONFIG_NUM_INPUTS_DATA_TYPE_UINT16
typedef uint16_t numInputs_t;
#endif
#ifdef CONFIG_NUM_INPUTS_DATA_TYPE_UINT32
typedef uint32_t numInputs_t;
#endif
#ifdef CONFIG_NUM_INPUTS_DATA_TYPE_UINT64
typedef uint64_t numInputs_t;
#endif


#ifdef CONFIG_NUM_HIDDEN_NEURONS_DATA_TYPE_UINT8
typedef uint8_t numHiddenNeurons_t;
#endif
#ifdef CONFIG_NUM_HIDDEN_NEURONS_DATA_TYPE_UINT16
typedef uint16_t numHiddenNeurons_t;
#endif
#ifdef CONFIG_NUM_HIDDEN_NEURONS_DATA_TYPE_UINT32
typedef uint32_t numHiddenNeurons_t;
#endif
#ifdef CONFIG_NUM_HIDDEN_NEURONS_DATA_TYPE_UINT64
typedef uint64_t numHiddenNeurons_t;
#endif


#ifdef CONFIG_NUM_LAYERS_DATA_TYPE_UINT8
typedef uint8_t numLayers_t;
#endif
#ifdef CONFIG_NUM_LAYERS_DATA_TYPE_UINT16
typedef uint16_t numLayers_t;
#endif
#ifdef CONFIG_NUM_LAYERS_DATA_TYPE_UINT32
typedef uint32_t numLayers_t;
#endif
#ifdef CONFIG_NUM_LAYERS_DATA_TYPE_UINT64
typedef uint64_t numLayers_t;
#endif

#ifdef CONFIG_NUM_TRAINING_DATA_ENTRIES_TYPE_UINT8
typedef uint8_t numTrainingDataEntries_t;
#endif
#ifdef CONFIG_NUM_TRAINING_DATA_ENTRIES_TYPE_UINT16
typedef uint16_t numTrainingDataEntries_t;
#endif
#ifdef CONFIG_NUM_TRAINING_DATA_ENTRIES_TYPE_UINT32
typedef uint32_t numTrainingDataEntries_t;
#endif
#ifdef CONFIG_NUM_TRAINING_DATA_ENTRIES_TYPE_UINT64
typedef uint64_t numTrainingDataEntries_t;
#endif



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
    activation_t activation;
} uNeuron_t;

typedef struct
{
    wNeuron_t* forgetGate;
    wNeuron_t* inputGate;
    wNeuron_t* outputGate;
    wNeuron_t* cell;
    activation_t activation;
} lstmCell_t;

typedef struct trainingData
{
    numOutputs_t correctResponse;
    numInputs_t length;
    struct trainingData* prev;
    struct trainingData* next;
    activation_t data[];
} trainingData_t;

typedef struct 
{
    trainingData_t* head;
    trainingData_t* tail;
    numTrainingDataEntries_t numEntries;
} trainingDataCollection_t;

typedef struct
{
    numInputs_t numRawInputs;
    activation_t maxInput;
    activation_t* rawInputs;
    activation_t* groupThresholds;
    activation_t* groupTotal;
} downscaler_t;

typedef struct
{
    numInputs_t numNeurons;
    uNeuron_t* neuron[];
} inputLayer_t;

typedef struct
{
    numHiddenNeurons_t numNeurons;
    wNeuron_t* neuron[];
} hiddenLayer_t;

typedef struct
{
    numOutputs_t numNeurons;
    wNeuron_t* neuron[];
} outputLayer_t;

typedef struct
{
    numLayers_t numLayers;
    numLayers_t numHiddenLayers;
    numOutputs_t networkResponse;
} networkProperties_t;

typedef struct
{
    networkProperties_t properties;
    inputLayer_t* inputLayer;
    outputLayer_t* outputLayer;
    hiddenLayer_t* hiddenLayer[];
} network_t;


#endif //Embann_data_types_h