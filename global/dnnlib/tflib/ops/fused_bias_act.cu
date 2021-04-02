// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#define EIGEN_USE_GPU
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

#define OP_CHECK_CUDA_ERROR(CTX, CUDA_CALL) do { cudaError_t err = CUDA_CALL; OP_REQUIRES(CTX, err == cudaSuccess, errors::Internal(cudaGetErrorName(err))); } while (false)

//------------------------------------------------------------------------
// CUDA kernel.

template <class T>
struct FusedBiasActKernelParams
{
    const T*    x;      // [sizeX]
    const T*    b;      // [sizeB] or NULL
    const T*    xref;   // [sizeX] or NULL
    const T*    yref;   // [sizeX] or NULL
    T*          y;      // [sizeX]

    int         grad;
    int         axis;
    int         act;
    float       alpha;
    float       gain;
    float       clamp;

    int         sizeX;
    int         sizeB;
    int         stepB;
    int         loopX;
};

template <class T>
static __global__ void FusedBiasActKernel(const FusedBiasActKernelParams<T> p)
{
    const float expRange        = 80.0f;
    const float halfExpRange    = 40.0f;
    const float seluScale       = 1.0507009873554804934193349852946f;
    const float seluAlpha       = 1.6732632423543772848170429916717f;

    // Loop over elements.
    int xi = blockIdx.x * p.loopX * blockDim.x + threadIdx.x;
    for (int loopIdx = 0; loopIdx < p.loopX && xi < p.sizeX; loopIdx++, xi += blockDim.x)
    {
        // Load and apply bias.
        float x = (float)p.x[xi];
        if (p.b)
            x += (float)p.b[(xi / p.stepB) % p.sizeB];
        float xref = (p.xref) ? (float)p.xref[xi] : 0.0f;
        float yref = (p.yref) ? (float)p.yref[xi] : 0.0f;
        float yy = (p.gain != 0.0f) ? yref / p.gain : 0.0f;

        // Evaluate activation func.
        float y;
        switch (p.act * 10 + p.grad)
        {
            // linear
            default:
            case 10: y = x; break;
            case 11: y = x; break;
            case 12: y = 0.0f; break;

            // relu
            case 20: y = (x > 0.0f) ? x : 0.0f; break;
            case 21: y = (yy > 0.0f) ? x : 0.0f; break;
            case 22: y = 0.0f; break;

            // lrelu
            case 30: y = (x > 0.0f) ? x : x * p.alpha; break;
            case 31: y = (yy > 0.0f) ? x : x * p.alpha; break;
            case 32: y = 0.0f; break;

            // tanh
            case 40: { float c = expf(x); float d = 1.0f / c; y = (x < -expRange) ? -1.0f : (x > expRange) ? 1.0f : (c - d) / (c + d); } break;
            case 41: y = x * (1.0f - yy * yy); break;
            case 42: y = x * (1.0f - yy * yy) * (-2.0f * yy); break;

            // sigmoid
            case 50: y = (x < -expRange) ? 0.0f : 1.0f / (expf(-x) + 1.0f); break;
            case 51: y = x * yy * (1.0f - yy); break;
            case 52: y = x * yy * (1.0f - yy) * (1.0f - 2.0f * yy); break;

            // elu
            case 60: y = (x >= 0.0f) ? x : expf(x) - 1.0f; break;
            case 61: y = (yy >= 0.0f) ? x : x * (yy + 1.0f); break;
            case 62: y = (yy >= 0.0f) ? 0.0f : x * (yy + 1.0f); break;

            // selu
            case 70: y = (x >= 0.0f) ? seluScale * x : (seluScale * seluAlpha) * (expf(x) - 1.0f); break;
            case 71: y = (yy >= 0.0f) ? x * seluScale : x * (yy + seluScale * seluAlpha); break;
            case 72: y = (yy >= 0.0f) ? 0.0f : x * (yy + seluScale * seluAlpha); break;

            // softplus
            case 80: y = (x > expRange) ? x : logf(expf(x) + 1.0f); break;
            case 81: y = x * (1.0f - expf(-yy)); break;
            case 82: { float c = expf(-yy); y = x * c * (1.0f - c); } break;

            // swish
            case 90: y = (x < -expRange) ? 0.0f : x / (expf(-x) + 1.0f); break;
            case 91:
            case 92:
                {
                    float c = expf(xref);
                    float d = c + 1.0f;
                    if (p.grad == 1)
                        y = (xref > halfExpRange) ? x : x * c * (xref + d) / (d * d);
                    else
                        y = (xref > halfExpRange) ? 0.0f : x * c * (xref * (2.0f - d) + 2.0f * d) / (d * d * d);
                    yref = (xref < -expRange) ? 0.0f : xref / (expf(-xref) + 1.0f) * p.gain;
                }
                break;
        }

        // Apply gain.
        y *= p.gain;

        // Clamp.
        if (p.clamp >= 0.0f)
        {
            if (p.grad == 0)
                y = (fabsf(y) < p.clamp) ? y : (y >= 0.0f) ? p.clamp : -p.clamp;
            else
                y = (fabsf(yref) < p.clamp) ? y : 0.0f;
        }

        // Store.
        p.y[xi] = (T)y;
    }
}

//------------------------------------------------------------------------
// TensorFlow op.

template <class T>
struct FusedBiasActOp : public OpKernel
{
    FusedBiasActKernelParams<T> m_attribs;

    FusedBiasActOp(OpKernelConstruction* ctx) : OpKernel(ctx)
    {
        memset(&m_attribs, 0, sizeof(m_attribs));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("grad",    &m_attribs.grad));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",    &m_attribs.axis));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("act",     &m_attribs.act));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha",   &m_attribs.alpha));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gain",    &m_attribs.gain));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("clamp",   &m_attribs.clamp));
        OP_REQUIRES(ctx, m_attribs.grad >= 0, errors::InvalidArgument("grad must be non-negative"));
        OP_REQUIRES(ctx, m_attribs.axis >= 0, errors::InvalidArgument("axis must be non-negative"));
        OP_REQUIRES(ctx, m_attribs.act >= 0, errors::InvalidArgument("act must be non-negative"));
    }

    void Compute(OpKernelContext* ctx)
    {
        FusedBiasActKernelParams<T> p = m_attribs;
        cudaStream_t stream = ctx->eigen_device<Eigen::GpuDevice>().stream();

        const Tensor& x     = ctx->input(0); // [...]
        const Tensor& b     = ctx->input(1); // [sizeB] or [0]
        const Tensor& xref  = ctx->input(2); // x.shape or [0]
        const Tensor& yref  = ctx->input(3); // x.shape or [0]
        p.x = x.flat<T>().data();
        p.b = (b.NumElements()) ? b.flat<T>().data() : NULL;
        p.xref = (xref.NumElements()) ? xref.flat<T>().data() : NULL;
        p.yref = (yref.NumElements()) ? yref.flat<T>().data() : NULL;
        OP_REQUIRES(ctx, b.NumElements() == 0 || m_attribs.axis < x.dims(), errors::InvalidArgument("axis out of bounds"));
        OP_REQUIRES(ctx, b.dims() == 1, errors::InvalidArgument("b must have rank 1"));
        OP_REQUIRES(ctx, b.NumElements() == 0 || b.NumElements() == x.dim_size(m_attribs.axis), errors::InvalidArgument("b has wrong number of elements"));
        OP_REQUIRES(ctx, xref.NumElements() == 0 || xref.NumElements() == x.NumElements(), errors::InvalidArgument("xref has wrong number of elements"));
        OP_REQUIRES(ctx, yref.NumElements() == 0 || yref.NumElements() == x.NumElements(), errors::InvalidArgument("yref has wrong number of elements"));
        OP_REQUIRES(ctx, x.NumElements() <= kint32max, errors::InvalidArgument("x is too large"));

        p.sizeX = (int)x.NumElements();
        p.sizeB = (int)b.NumElements();
        p.stepB = 1;
        for (int i = m_attribs.axis + 1; i < x.dims(); i++)
            p.stepB *= (int)x.dim_size(i);

        Tensor* y = NULL; // x.shape
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
        p.y = y->flat<T>().data();

        p.loopX = 4;
        int blockSize = 4 * 32;
        int gridSize = (p.sizeX - 1) / (p.loopX * blockSize) + 1;
        void* args[] = {&p};
        OP_CHECK_CUDA_ERROR(ctx, cudaLaunchKernel((void*)FusedBiasActKernel<T>, gridSize, blockSize, args, 0, stream));
    }
};

REGISTER_OP("FusedBiasAct")
    .Input      ("x: T")
    .Input      ("b: T")
    .Input      ("xref: T")
    .Input      ("yref: T")
    .Output     ("y: T")
    .Attr       ("T: {float, half}")
    .Attr       ("grad: int = 0")
    .Attr       ("axis: int = 1")
    .Attr       ("act: int = 0")
    .Attr       ("alpha: float = 0.0")
    .Attr       ("gain: float = 1.0")
    .Attr       ("clamp: float = -1.0");
REGISTER_KERNEL_BUILDER(Name("FusedBiasAct").Device(DEVICE_GPU).TypeConstraint<float>("T"), FusedBiasActOp<float>);
REGISTER_KERNEL_BUILDER(Name("FusedBiasAct").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), FusedBiasActOp<Eigen::half>);

//------------------------------------------------------------------------
