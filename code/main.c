#include <stdio.h>

typedef unsigned char u8;
typedef unsigned int  u32;
typedef float         f32;

#ifndef bool
#define bool u8
#define true (u8)1
#define false (u8)0
#endif

#define UINT32_MAX 0xffffffffui32

#define HEIGHT 28
#define WIDTH 28
#define NUMBER_OF_IMAGES 60000
#define NUMBER_OF_TEST_IMAGES 10000

#define RANDOM_NUMBER_SEED (42)

#define INPUT_LAYER_SIZE (HEIGHT*WIDTH)
#define SECOND_LAYER_SIZE (30)
#define OUTPUT_LAYER_SIZE (10)
#define LEARNING_RATE (0.1)

#define TRAINING_EXAMPLES 60000
#define TESTING_EXAMPLES 10000

#define BATCH_SIZE 10
#define EPOCHS 1

#define NORMALIZE_PIXEL(px) (1.0f*(px)/255)

static u32
u32BigEndianToLittleEndian(u32 X)
{
    u32 Byte1 = X & 0xFF000000;
    u32 Byte2 = X & 0x00FF0000;
    u32 Byte3 = X & 0x0000FF00;
    u32 Byte4 = X & 0x000000FF;
    
    return (Byte4 << 24) | (Byte3 << 8) | (Byte2 >> 8) | (Byte1 >> 24);
}

static bool
ReadIDX1(char* Filename, u8* Labels, u32 ExpectedNumberOfLabels)
{
    FILE* File = fopen(Filename, "rb");
    if(File)
    {
        u32 MagicConstant;
        fread(&MagicConstant, 4, 1, File);
        MagicConstant = u32BigEndianToLittleEndian(MagicConstant);
        if(MagicConstant == 2049) 
        {
            u32 NumberOfLabels;
            fread(&NumberOfLabels, 4, 1, File);
            NumberOfLabels = u32BigEndianToLittleEndian(NumberOfLabels);
            if(NumberOfLabels == ExpectedNumberOfLabels)
            {
                fread(Labels, 1, NumberOfLabels, File);
                fclose(File);
                return true;
            }
            else
            {
                fprintf(stderr, "Number Of Labels Does Not Match The Expected\n");
                fclose(File);
                return false;
            }
        }
        else
        {
            fprintf(stderr, "Magic Constant Not Found\n");
            fclose(File);
            return false;
        }
    }
    else
    {
        fprintf(stderr, "File not found\n");
        return false;
    }
}

static bool
ReadIDX3(char* Filename, u8* Images, 
         u32 ExpectedNumberOfImages,
         u32 ExpectedHeight,
         u32 ExpectedWidth)
{
    FILE* File = fopen(Filename, "rb");
    if(File)
    {
        u32 MagicConstant;
        fread(&MagicConstant, 4, 1, File);
        MagicConstant = u32BigEndianToLittleEndian(MagicConstant);
        if(MagicConstant == 2051)
        {
            u32 NumberOfImages;
            fread(&NumberOfImages, 4, 1, File);
            NumberOfImages = u32BigEndianToLittleEndian(NumberOfImages);
            u32 ImagesHeight;
            fread(&ImagesHeight, 4, 1, File);
            ImagesHeight = u32BigEndianToLittleEndian(ImagesHeight);
            u32 ImagesWidth;
            fread(&ImagesWidth, 4, 1, File);
            ImagesWidth = u32BigEndianToLittleEndian(ImagesWidth);
            
            if(NumberOfImages == ExpectedNumberOfImages && ImagesHeight == ExpectedHeight &&
               ImagesWidth && ExpectedWidth)
            {
                fread(Images, 1, NumberOfImages*ImagesWidth*ImagesHeight, File);
                fclose(File);
                return true;
            }
            else 
            {
                fprintf(stderr, "File constants do not match the expected\n");
                fclose(File);
                return false;
            }
        }
        else
        {
            fprintf(stderr, "Magic Constant Not Found\n");
            fclose(File);
            return false;
        }
    }
    else
    {
        fprintf(stderr, "File not found\n");
        return false;
    }
}

typedef struct random_state
{ 
    u32 State; 
} random_state;

static u32 
NextRandom(random_state* RandomState)
{
    // NOTE(luatil): https://en.wikipedia.org/wiki/Xorshift 32bit implementation
    u32 Result = RandomState->State;
    Result ^= Result << 13;
    Result ^= Result >> 17;
    Result ^= Result << 5;
    RandomState->State = Result;
    return Result;
}

static f32 
RandomZeroToOne(random_state* RandomState)
{
    f32 Divisor = 1.0f / UINT32_MAX;
    f32 Result = Divisor * NextRandom(RandomState);
    return Result;
}

static void
RandomInitialization(f32 W1[INPUT_LAYER_SIZE][SECOND_LAYER_SIZE],
                     f32 W2[SECOND_LAYER_SIZE][OUTPUT_LAYER_SIZE],
                     f32 Bias1[SECOND_LAYER_SIZE],
                     f32 Bias2[OUTPUT_LAYER_SIZE])
{
    random_state RandomState = {RANDOM_NUMBER_SEED};
    
    for(u32 I = 0; I < INPUT_LAYER_SIZE; I++)
    {
        for(u32 J = 0; J < SECOND_LAYER_SIZE; J++)
        {
            f32 Temp = 2*RandomZeroToOne(&RandomState)-1;
            W1[I][J] = Temp;
        }
    }
    
    for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
    {
        for(u32 J = 0; J < OUTPUT_LAYER_SIZE; J++)
        {
            f32 Temp = 2*RandomZeroToOne(&RandomState)-1;
            W2[I][J] = Temp;
        }
    }
    
    for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
    {
        Bias1[I] = 2*RandomZeroToOne(&RandomState)-1;
    }
    
    for(u32 I = 0; I < OUTPUT_LAYER_SIZE; I++)
    {
        Bias2[I] = 2*RandomZeroToOne(&RandomState)-1;
    }
}

static f32 
Exp(f32 X)
{
    if(X >= 0)
    {
        return 1 + X + X*X/2 + X*X*X/6 + X*X*X*X/24 + X*X*X*X*X/120;
    }
    else
    {
        X = -X;
        f32 Result = 1 + X + X*X/2 + X*X*X/6 + X*X*X*X/24 + X*X*X*X*X/120;
        return 1.0f / Result;
    }
    
}

static f32 
Sigmoid(f32 X)
{
    return 1.0f / (1.0f + Exp(-X));
}

static f32 
SigmoidPrime(f32 X)
{
    return Sigmoid(X)*(1 - Sigmoid(X));
}

static void
ForwardPass(f32 W1[INPUT_LAYER_SIZE][SECOND_LAYER_SIZE],
            f32 W2[SECOND_LAYER_SIZE][OUTPUT_LAYER_SIZE],
            f32 B1[SECOND_LAYER_SIZE],
            f32 B2[OUTPUT_LAYER_SIZE],
            f32 Z1[SECOND_LAYER_SIZE],
            f32 Z2[OUTPUT_LAYER_SIZE],
            u8* A0,
            f32 A1[SECOND_LAYER_SIZE],
            f32 A2[OUTPUT_LAYER_SIZE])
{
    for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
    {
        Z1[I] = B1[I];
        for(u32 J = 0; J < INPUT_LAYER_SIZE; J++)
        {
            Z1[I] += W1[J][I] * NORMALIZE_PIXEL(A0[J]);
        }
        A1[I] = Sigmoid(Z1[I]);
    }
    
    for(u32 I = 0; I < OUTPUT_LAYER_SIZE; I++)
    {
        Z2[I] = B2[I];
        for(u32 J = 0; J < SECOND_LAYER_SIZE; J++)
        {
            Z2[I] += W2[J][I] * A1[J];
        }
        A2[I] = Sigmoid(Z2[I]);
    }
}

static void
Backpropagate(f32 W1[INPUT_LAYER_SIZE][SECOND_LAYER_SIZE],
              f32 W2[SECOND_LAYER_SIZE][OUTPUT_LAYER_SIZE],
              f32 Z1[SECOND_LAYER_SIZE],
              f32 Z2[OUTPUT_LAYER_SIZE],
              f32 A2[OUTPUT_LAYER_SIZE],
              f32 Y[OUTPUT_LAYER_SIZE],
              f32 Nabla1[SECOND_LAYER_SIZE],
              f32 Nabla2[OUTPUT_LAYER_SIZE])
{
    for(u32 I = 0; I < OUTPUT_LAYER_SIZE; I++)
    {
        Nabla2[I] = (A2[I] - Y[I]) * SigmoidPrime(Z2[I]); 
    }
    
    for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
    {
        Nabla1[I] = 0.0f;
        for(u32 J = 0; J < OUTPUT_LAYER_SIZE; J++)
        {
            Nabla1[I] += W2[I][J] * Nabla2[J];
        }
        Nabla1[I] *= SigmoidPrime(Z1[I]);
    }
}


static void
StocasticUpdate(f32 W1[INPUT_LAYER_SIZE][SECOND_LAYER_SIZE],
                f32 W2[SECOND_LAYER_SIZE][OUTPUT_LAYER_SIZE],
                f32 B1[SECOND_LAYER_SIZE],
                f32 B2[OUTPUT_LAYER_SIZE],
                u8* RawImages,
                u8* Tags)
{
    f32 Nabla1[SECOND_LAYER_SIZE] = {0};
    f32 Nabla2[OUTPUT_LAYER_SIZE] = {0};
    
    f32 DeltaW1[INPUT_LAYER_SIZE][SECOND_LAYER_SIZE] = {0};
    f32 DeltaW2[SECOND_LAYER_SIZE][OUTPUT_LAYER_SIZE] = {0};
    f32 DeltaB1[SECOND_LAYER_SIZE] = {0};
    f32 DeltaB2[OUTPUT_LAYER_SIZE] = {0};
    
    for(u32 K = 0; K < BATCH_SIZE; K++)
    {
        u8* A0 = RawImages;
        f32 Y[OUTPUT_LAYER_SIZE] = {0};
        Y[Tags[K]] = 1.0f;
        
        RawImages += INPUT_LAYER_SIZE;
        
        f32 Z1[SECOND_LAYER_SIZE], Z2[OUTPUT_LAYER_SIZE];
        f32 A1[SECOND_LAYER_SIZE], A2[OUTPUT_LAYER_SIZE];
        ForwardPass(W1, W2, B1, B2, Z1, Z2, A0, A1, A2);
        Backpropagate(W1, W2, Z1, Z2, A2, Y, Nabla1, Nabla2);
        
        for(u32 I = 0; I < INPUT_LAYER_SIZE; I++)
        {
            for(u32 J = 0; J < SECOND_LAYER_SIZE; J++)
            {
                DeltaW1[I][J] += NORMALIZE_PIXEL(A0[I]) * Nabla1[J];
            }
        }
        
        for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
        {
            for(u32 J = 0; J < OUTPUT_LAYER_SIZE; J++)
            {
                DeltaW2[I][J] += A1[I] * Nabla2[J];
            }
        }
        
        for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
        {
            DeltaB1[I] += Nabla1[I];
        }
        
        for(u32 I = 0; I < OUTPUT_LAYER_SIZE; I++)
        {
            DeltaB2[I] += Nabla2[I];
        }
    }
    
    for(u32 I = 0; I < INPUT_LAYER_SIZE; I++)
    {
        for(u32 J = 0; J < SECOND_LAYER_SIZE; J++)
        {
            W1[I][J] -= LEARNING_RATE*DeltaW1[I][J];
        }
    }
    
    for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
    {
        for(u32 J = 0; J < OUTPUT_LAYER_SIZE; J++)
        {
            W2[I][J] -= LEARNING_RATE*DeltaW2[I][J];
        }
    }
    
    for(u32 I = 0; I < SECOND_LAYER_SIZE; I++)
    {
        B1[I] -= LEARNING_RATE*DeltaB1[I];
    }
    
    for(u32 I = 0; I < OUTPUT_LAYER_SIZE; I++)
    {
        B2[I] -= LEARNING_RATE*DeltaB2[I];
    }
}

static u8
FindMostProbableValue(f32 *Vector, u32 VectorLength)
{
    f32 *MaxValue = Vector;
    f32 *Base = Vector;
    for(u32 I = 0; I < VectorLength; I++)
    {
        MaxValue = (*MaxValue < *Vector) ? Vector : MaxValue;
        Vector++;
    }
    return (u8)(MaxValue - Base);
}

int
main()
{
    
    char* TrainingLabelsFilename = "..\\mnist\\train-labels.idx1-ubyte";
    static u8 TrainingLabels[NUMBER_OF_IMAGES];
    if(!ReadIDX1(TrainingLabelsFilename, TrainingLabels, NUMBER_OF_IMAGES)) return;
    
    char* TrainingImagesFilename = "..\\mnist\\train-images.idx3-ubyte";
    static u8 RawImages[NUMBER_OF_IMAGES*HEIGHT*WIDTH];
    if(!ReadIDX3(TrainingImagesFilename, RawImages, NUMBER_OF_IMAGES, HEIGHT, WIDTH)) return;
    
    char* TestingLabelsFilename  = "..\\mnist\\t10k-labels.idx1-ubyte";
    static u8 TestingLabels[NUMBER_OF_TEST_IMAGES];
    if(!ReadIDX1(TestingLabelsFilename, TestingLabels, NUMBER_OF_TEST_IMAGES)) return;
    
    char* TestingImagesFilename  = "..\\mnist\\t10k-images.idx3-ubyte";
    static u8 RawTestImages[NUMBER_OF_TEST_IMAGES*HEIGHT*WIDTH];
    if(!ReadIDX3(TestingImagesFilename, RawTestImages, NUMBER_OF_TEST_IMAGES, HEIGHT, WIDTH)) return;
    
    static f32 W1[INPUT_LAYER_SIZE][SECOND_LAYER_SIZE];
    static f32 W2[SECOND_LAYER_SIZE][OUTPUT_LAYER_SIZE];
    
    static f32 B1[SECOND_LAYER_SIZE];
    static f32 B2[OUTPUT_LAYER_SIZE];
    
    static f32 BatchDelta1[BATCH_SIZE][SECOND_LAYER_SIZE];
    static f32 BatchDelta2[BATCH_SIZE][OUTPUT_LAYER_SIZE];
    
    RandomInitialization(W1, W2, B1, B2);
    
    u32 Epochs = EPOCHS;
    while(Epochs--)
    {
        u8* ImageBatch = RawImages;
        u8* LabelBatch = TrainingLabels;
        u32 NumberOfBatches = TRAINING_EXAMPLES / BATCH_SIZE;
        while(NumberOfBatches--)
        {
            StocasticUpdate(W1, W2, B1, B2, ImageBatch, LabelBatch);
            ImageBatch += HEIGHT*WIDTH*BATCH_SIZE;
            LabelBatch += BATCH_SIZE;
        }
        
        u32 Correct = 0;
        u8* PtrRawTestImages = RawTestImages;
        for(u32 K = 0; K < TESTING_EXAMPLES; K++)
        {
            u8* A0 = PtrRawTestImages;
            u8 Tag = TestingLabels[K];
            
            PtrRawTestImages += INPUT_LAYER_SIZE;
            
            f32 Y[OUTPUT_LAYER_SIZE] = {0};
            Y[Tag] = 1.0f;
            
            f32 Z1[SECOND_LAYER_SIZE], Z2[OUTPUT_LAYER_SIZE];
            f32 A1[SECOND_LAYER_SIZE], A2[OUTPUT_LAYER_SIZE];
            
            ForwardPass(W1, W2, B1, B2, Z1, Z2, A0, A1, A2); 
            u8 MostProbableValue = FindMostProbableValue(A2, OUTPUT_LAYER_SIZE);
            
            if(MostProbableValue == Tag)
            {
                Correct++;
            }
        }
        printf("Epoch: %2d - Correct: --%d/%d\n", EPOCHS - Epochs, Correct, TESTING_EXAMPLES);
    }
    return 0;
}