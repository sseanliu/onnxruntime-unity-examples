#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Unity
{
    public class ImageInference<T> : IDisposable
        where T : unmanaged
    {
        public readonly ImageInferenceOptions options;

        protected readonly InferenceSession session;
        protected readonly string[] inputNames;
        protected readonly OrtValue[] inputs;
        protected readonly string[] outputNames;
        protected readonly OrtValue[] outputs;

        protected readonly TextureToTensor<T> textureToTensor;

        protected readonly string inputImageKey;
        protected readonly int channels;
        protected readonly int height;
        protected readonly int width;

        public Texture InputTexture => textureToTensor.Texture;

        /// <summary>
        /// Create an inference that has Image input
        /// </summary>
        /// <param name="model">byte array of the Ort model</param>
        public ImageInference(byte[] model, ImageInferenceOptions options)
        {
            this.options = options;

            try
            {
                // TODO: support GPU options
                session = new InferenceSession(model);
            }
            catch (Exception e)
            {
                session?.Dispose();
                throw e;
            }
            session.LogIOInfo();

            // Allocate inputs/outputs
            (inputNames, inputs) = AllocateTensors(session.InputMetadata);
            (outputNames, outputs) = AllocateTensors(session.OutputMetadata);

            // Find image input info
            foreach (var kv in session.InputMetadata)
            {
                NodeMetadata meta = kv.Value;
                if (meta.IsTensor)
                {
                    int[] shape = meta.Dimensions;
                    if (IsSupportedImage(meta.Dimensions))
                    {
                        inputImageKey = kv.Key;
                        channels = shape[1];
                        height = shape[2];
                        width = shape[3];
                        break;
                    }
                }
            }
            if (inputImageKey == null)
            {
                throw new ArgumentException("Image input not found");
            }

            textureToTensor = new TextureToTensor<T>(width, height);
        }

        public void Dispose()
        {
            foreach (var ortValue in inputs)
            {
                ortValue.Dispose();
            }
            foreach (var ortValue in outputs)
            {
                ortValue.Dispose();
            }
            textureToTensor?.Dispose();
            session?.Dispose();
        }

        public virtual void Run(Texture texture)
        {
            // Prepare input tensor
            textureToTensor.Transform(texture, options.aspectMode);
            var inputSpan = inputs[0].GetTensorMutableDataAsSpan<T>();
            textureToTensor.TensorData.CopyTo(inputSpan);

            // Run inference

            // var timer = Stopwatch.StartNew();
            session.Run(null, inputNames, inputs, outputNames, outputs);
            // timer.Stop();
            // UnityEngine.Debug.Log($"Inference time: {timer.ElapsedMilliseconds} ms");
        }

        private static (string[], OrtValue[]) AllocateTensors(IReadOnlyDictionary<string, NodeMetadata> metadata)
        {
            var names = new List<string>();
            var values = new List<OrtValue>();

            foreach (var kv in metadata)
            {
                NodeMetadata meta = kv.Value;
                if (meta.IsTensor)
                {
                    names.Add(kv.Key);
                    values.Add(TensorFromMetadata(meta));
                }
                else
                {
                    throw new ArgumentException("Only tensor input is supported");
                }
            }
            return (names.ToArray(), values.ToArray());
        }

        private static OrtValue TensorFromMetadata(NodeMetadata metadata)
        {
            long[] shape = metadata.Dimensions.Select(x => (long)x).ToArray();
            var ortValue = OrtValue.CreateAllocatedTensorValue(
                OrtAllocator.DefaultInstance, metadata.ElementDataType, shape);
            return ortValue;
        }

        private static bool IsSupportedImage(int[] shape)
        {
            int channels = shape.Length switch
            {
                4 => shape[0] == 1 ? shape[1] : 0,
                3 => shape[0],
                _ => 0
            };
            // Only RGB is supported for now
            return channels == 3;
            // return channels == 1 || channels == 3 || channels == 4;
        }

    }
}
