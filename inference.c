#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "onnxruntime_c_api.h"

#define INPUT_H       28
#define INPUT_W       28
#define INPUT_SIZE    (INPUT_H * INPUT_W)
#define NUM_CLASSES   10

static const OrtApi *ort = NULL;

static void check(OrtStatusPtr s) {
    if (s) {
        fprintf(stderr, "ONNX Runtime error: %s\n", ort->GetErrorMessage(s));
        ort->ReleaseStatus(s);
        exit(1);
    }
}

int predict(OrtSession *session, OrtAllocator *allocator,
            const float *input, float *out_probs) {
    int64_t input_shape[] = {1, 1, INPUT_H, INPUT_W};

    /* --- create input tensor (wraps user buffer, no copy) --- */
    OrtMemoryInfo *mem = NULL;
    check(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));

    OrtValue *in_tensor = NULL;
    check(ort->CreateTensorWithDataAsOrtValue(
        mem, (void *)input, INPUT_SIZE * sizeof(float),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor));

    /* --- get I/O names from the model --- */
    char *in_name = NULL, *out_name = NULL;
    check(ort->SessionGetInputName(session, 0, allocator, &in_name));
    check(ort->SessionGetOutputName(session, 0, allocator, &out_name));

    const char *in_names[] = {in_name};
    const char *out_names[] = {out_name};
    OrtValue *out_tensor = NULL;

    /* --- run inference --- */
    check(ort->Run(session, NULL,
                   in_names, (const OrtValue *const *)&in_tensor, 1,
                   out_names, 1, &out_tensor));

    /* --- read output logits --- */
    float *logits = NULL;
    check(ort->GetTensorMutableData(out_tensor, (void **)&logits));

    /* softmax to probabilities */
    float maxv = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++)
        if (logits[i] > maxv) maxv = logits[i];

    float sum = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        out_probs[i] = expf(logits[i] - maxv);
        sum += out_probs[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) out_probs[i] /= sum;

    /* argmax */
    int pred = 0;
    float best = out_probs[0];
    for (int i = 1; i < NUM_CLASSES; i++)
        if (out_probs[i] > best) { best = out_probs[i]; pred = i; }

    ort->ReleaseValue(in_tensor);
    ort->ReleaseValue(out_tensor);
    ort->ReleaseMemoryInfo(mem);

    return pred;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx> [input.raw]\n", argv[0]);
        return 1;
    }

    /* --- init ONNX Runtime --- */
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) { fprintf(stderr, "Failed to get ONNX Runtime API\n"); return 1; }

    OrtEnv *env = NULL;
    check(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mnist", &env));

    OrtSessionOptions *opts = NULL;
    check(ort->CreateSessionOptions(&opts));
    check(ort->SetIntraOpNumThreads(opts, 1));

    OrtSession *session = NULL;
    check(ort->CreateSession(env, argv[1], opts, &session));
    ort->ReleaseSessionOptions(opts);
    printf("Loaded model from %s\n", argv[1]);

    OrtAllocator *allocator = NULL;
    check(ort->GetAllocatorWithDefaultOptions(&allocator));

    /* --- run inference --- */
    if (argc >= 3) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) { perror("fopen input"); return 1; }

        float input[INPUT_SIZE];
        if (fread(input, sizeof(float), INPUT_SIZE, f) != INPUT_SIZE) {
            fprintf(stderr, "Failed to read %d floats from %s\n", INPUT_SIZE, argv[2]);
            fclose(f);
            return 1;
        }
        fclose(f);

        float probs[NUM_CLASSES];
        int digit = predict(session, allocator, input, probs);

        printf("Predicted digit: %d\n", digit);
        printf("Confidence:\n");
        for (int i = 0; i < NUM_CLASSES; i++)
            printf("  %d: %.2f%%\n", i, probs[i] * 100.0f);
    } else {
        float input[INPUT_SIZE] = {0};
        float probs[NUM_CLASSES];
        int digit = predict(session, allocator, input, probs);
        printf("Model loaded. Demo prediction on zero input: %d\n", digit);
    }

    ort->ReleaseSession(session);
    ort->ReleaseEnv(env);
    return 0;
}
