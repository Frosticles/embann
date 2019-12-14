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
#define MAX_ACTIVATION INT8_MAX
#define MIN_ACTIVATION INT8_MIN
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT16
typedef int16_t activation_t;
#define ACTIVATION_IS_SIGNED
#define MAX_ACTIVATION INT16_MAX
#define MIN_ACTIVATION INT16_MIN
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT32
typedef int32_t activation_t;
#define ACTIVATION_IS_SIGNED
#define MAX_ACTIVATION INT32_MAX
#define MIN_ACTIVATION INT32_MIN
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT64
typedef int64_t activation_t;
#define ACTIVATION_IS_SIGNED
#define MAX_ACTIVATION INT64_MAX
#define MIN_ACTIVATION INT64_MIN
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT8
typedef uint8_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION UINT8_MAX
#define MIN_ACTIVATION 0
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT16
typedef uint16_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION UINT16_MAX
#define MIN_ACTIVATION 0
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT32
typedef uint32_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION UINT32_MAX
#define MIN_ACTIVATION 0
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT64
typedef uint64_t activation_t;
#define ACTIVATION_IS_UNSIGNED
#define MAX_ACTIVATION UINT64_MAX
#define MIN_ACTIVATION 0
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_FLOAT
typedef float activation_t;
#define ACTIVATION_IS_FLOAT
#define MAX_ACTIVATION FLT_MAX
#define MIN_ACTIVATION FLT_MIN
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_DOUBLE
typedef double activation_t;
#define ACTIVATION_IS_FLOAT
#define MAX_ACTIVATION DBL_MAX
#define MIN_ACTIVATION DBL_MIN
#endif

#ifdef CONFIG_BIAS_DATA_TYPE_INT8
typedef int8_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS INT8_MAX
#define MIN_BIAS INT8_MIN
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT16
typedef int16_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS INT16_MAX
#define MIN_BIAS INT16_MIN
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT32
typedef int32_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS INT32_MAX
#define MIN_BIAS INT32_MIN
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT64
typedef int64_t bias_t;
#define BIAS_IS_SIGNED
#define MAX_BIAS INT64_MAX
#define MIN_BIAS INT64_MIN
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT8
typedef uint8_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS UINT8_MAX
#define MIN_BIAS 0
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT16
typedef uint16_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS UINT16_MAX
#define MIN_BIAS 0
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT32
typedef uint32_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS UINT32_MAX
#define MIN_BIAS 0
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT64
typedef uint64_t bias_t;
#define BIAS_IS_UNSIGNED
#define MAX_BIAS UINT64_MAX
#define MIN_BIAS 0
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_FLOAT
typedef float bias_t;
#define BIAS_IS_FLOAT
#define MAX_BIAS FLT_MAX
#define MIN_BIAS FLT_MIN
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_DOUBLE
typedef double bias_t;
#define BIAS_IS_FLOAT
#define MAX_BIAS DBL_MAX
#define MIN_BIAS DBL_MIN
#endif

#ifdef CONFIG_WEIGHT_DATA_TYPE_INT8
typedef int8_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT INT8_MAX
#define MIN_WEIGHT INT8_MIN
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT16
typedef int16_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT INT16_MAX
#define MIN_WEIGHT INT16_MIN
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT32
typedef int32_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT INT32_MAX
#define MIN_WEIGHT INT32_MIN
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT64
typedef int64_t weight_t;
#define WEIGHT_IS_SIGNED
#define MAX_WEIGHT INT64_MAX
#define MIN_WEIGHT INT64_MIN
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT8
typedef uint8_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT UINT8_MAX
#define MIN_WEIGHT 0
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT16
typedef uint16_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT UINT16_MAX
#define MIN_WEIGHT 0
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT32
typedef uint32_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT UINT32_MAX
#define MIN_WEIGHT 0
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT64
typedef uint64_t weight_t;
#define WEIGHT_IS_UNSIGNED
#define MAX_WEIGHT UINT64_MAX
#define MIN_WEIGHT 0
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_FLOAT
typedef float weight_t;
#define WEIGHT_IS_FLOAT
#define MAX_WEIGHT FLT_MAX
#define MIN_WEIGHT FLT_MIN
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_DOUBLE
typedef double weight_t;
#define WEIGHT_IS_FLOAT
#define MAX_WEIGHT DBL_MAX
#define MIN_WEIGHT DBL_MIN
#endif




#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_INT8
typedef int8_t accumulator_t;
#define ACCUMULATOR_IS_SIGNED
#define MAX_ACCUMULATOR INT8_MAX
#define MIN_ACCUMULATOR INT8_MIN
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_INT16
typedef int16_t accumulator_t;
#define ACCUMULATOR_IS_SIGNED
#define MAX_ACCUMULATOR INT16_MAX
#define MIN_ACCUMULATOR INT16_MIN
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_INT32
typedef int32_t accumulator_t;
#define ACCUMULATOR_IS_SIGNED
#define MAX_ACCUMULATOR INT32_MAX
#define MIN_ACCUMULATOR INT32_MIN
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_INT64
typedef int64_t accumulator_t;
#define ACCUMULATOR_IS_SIGNED
#define MAX_ACCUMULATOR INT64_MAX
#define MIN_ACCUMULATOR INT64_MIN
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_UINT8
typedef uint8_t accumulator_t;
#define ACCUMULATOR_IS_UNSIGNED
#define MAX_ACCUMULATOR UINT8_MAX
#define MIN_ACCUMULATOR 0
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_UINT16
typedef uint16_t accumulator_t;
#define ACCUMULATOR_IS_UNSIGNED
#define MAX_ACCUMULATOR UINT16_MAX
#define MIN_ACCUMULATOR 0
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_UINT32
typedef uint32_t accumulator_t;
#define ACCUMULATOR_IS_UNSIGNED
#define MAX_ACCUMULATOR UINT32_MAX
#define MIN_ACCUMULATOR 0
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_UINT64
typedef uint64_t accumulator_t;
#define ACCUMULATOR_IS_UNSIGNED
#define MAX_ACCUMULATOR UINT64_MAX
#define MIN_ACCUMULATOR 0
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_FLOAT
typedef float accumulator_t;
#define ACCUMULATOR_IS_FLOAT
#define MAX_ACCUMULATOR FLT_MAX
#define MIN_ACCUMULATOR FLT_MIN
#endif
#ifdef CONFIG_ACCUMULATOR_DATA_TYPE_DOUBLE
typedef double accumulator_t;
#define ACCUMULATOR_IS_FLOAT
#define MAX_ACCUMULATOR DBL_MAX
#define MIN_ACCUMULATOR DBL_MIN
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



typedef enum 
{
    SOFTSIGN,
    RELU,
    LEAKY_RELU
} activationFunction_t;

typedef struct trainingData
{
    numOutputs_t correctResponse;
    numInputs_t length;
    struct trainingData* prev;
    struct trainingData* next;
    activation_t* data;
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
    activation_t* activation;
} inputLayer_t;

typedef struct
{
    numHiddenNeurons_t numNeurons;
    activation_t* activation;
    bias_t* bias;
    weight_t** weight;
} hiddenLayer_t;

typedef struct
{
    numOutputs_t numNeurons;
    activation_t* activation;
    bias_t* bias;
    weight_t** weight;
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
    hiddenLayer_t** hiddenLayer;
} network_t;


#endif //Embann_data_types_h