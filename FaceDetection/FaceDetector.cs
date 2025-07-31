using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using System.Xml;
using FaceAnalysisApp;



namespace FaceAnalysisApp.FaceDetection
{
    public class FaceDetector
    {
        private readonly InferenceSession session;
        private const int InputWidth = 640;
        private const int InputHeight = 640;

        public FaceDetector(string modelPath)
        {
            session = new InferenceSession(modelPath);
            
            Console.WriteLine("[INFO] Model loaded successfully");

        }

        public void DetectFaces(string imagePath)
        {
            Console.WriteLine($"[INFO] Ready to detect faces in image: {imagePath}");
        }


        public static float ComputeIoU(RectangleF a, RectangleF b)
        {
            float ix1 = Math.Max(a.Left, b.Left);
            float iy1 = Math.Max(a.Top, b.Top);
            float ix2 = Math.Min(a.Right, b.Right);
            float iy2 = Math.Min(a.Bottom, b.Bottom);
            float iW = Math.Max(0, ix2 - ix1);
            float iH = Math.Max(0, iy2 - iy1);
            float interArea = iW * iH;
            float areaA = a.Width * a.Height;
            float areaB = b.Width * b.Height;
            float unionArea = areaA + areaB - interArea;
            return interArea / unionArea;
        }


        public float[] Preprocess(Bitmap image)
        {
            Bitmap resized = new Bitmap(image, InputWidth, InputHeight);
            float[] data = new float[InputWidth * InputHeight * 3];

            for (int y = 0; y < InputHeight; y++)
            {
                for (int x =0;x < InputWidth; x++)
                {
                    int baseIndex = y * InputWidth + x;
                    Color pixel = resized.GetPixel(x, y);
                    data[baseIndex + 0 * InputWidth * InputHeight] = (pixel.B - 127.5f) / 128.0f;
                    data[baseIndex + 1 * InputWidth * InputHeight] = (pixel.G - 127.5f) / 128.0f;
                    data[baseIndex + 2 * InputWidth * InputHeight] = (pixel.R - 127.5f) / 128.0f;
                }
            }
            return data;
        }

        public Dictionary<string, Tensor<float>> RunInference(float[] data)
        {
            int[] dimensions = new int[] { 1, 3, InputHeight, InputWidth};
            var tensor = new DenseTensor<float>(data, dimensions);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input.1", tensor)
            };
            var results = session.Run(inputs);
            var outputDict = new Dictionary<string, Tensor<float>>();
            foreach (var r in results)
            {
                outputDict[r.Name] = r.AsTensor<float>();
            }
            return outputDict;
        }

        public struct Detection {
            public RectangleF Box;
            public float Confidence;
        }

        public List<Detection> Postprocess(Dictionary<string, Tensor<float>> outputs, int origWidth, int origHeight,
                                            float confThreshold = 0.25f, float iouThreshold = 0.3f)
        {
            var scoreTensors = outputs
                .Where(kv => kv.Value.Dimensions.Length == 2 && kv.Value.Dimensions[1] == 1)
                .Select(kv => kv.Value)
                .ToList();
            var bboxTensors = outputs
                .Where(kv => kv.Value.Dimensions.Length == 2 && kv.Value.Dimensions[1] == 4)
                .Select(kv => kv.Value)
                .ToList();
            
            int[] strides = new int[] { 8, 16, 32 };
            List<float[]> scoresList = new List<float[]>();
            var candidates = new List<Detection>();

            // Sigmoid for scores.
            foreach (var tensor in scoreTensors)
            {
                var data = tensor.ToArray();
                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = 1f / (1f + (float)Math.Exp(-data[i]));
                }
                scoresList.Add(data);
            }

            float scaleX = (float)origWidth / InputWidth;
            float scaleY = (float)origHeight / InputHeight;

            for (int i = 0; i < bboxTensors.Count; i++)
            {
                var bboxTensor = bboxTensors[i];
                var scoreArray = scoresList[i];
                int stride = strides[i];

                int featureW = InputWidth / stride;
                int featureH = InputHeight / stride;
                int numAnchors = featureH * featureW * 2;

                var bboxData = bboxTensor.ToArray();

                for (int j = 0; j < numAnchors; j++) {
                    int baseIndex = j * 4;

                    float dx1 = bboxData[baseIndex + 0];
                    float dy1 = bboxData[baseIndex + 1];
                    float dx2 = bboxData[baseIndex + 2];
                    float dy2 = bboxData[baseIndex + 3];

                    float conf = scoreArray[j];

                    int cellIndex = j / 2;
                    int col = cellIndex % featureW;
                    int row = cellIndex % featureH;

                    float cx = col * stride;
                    float cy = row * stride;


                    float x1 = cx - dx1;
                    float y1 = cy - dy1;
                    float x2 = cx + dx2;
                    float y2 = cy + dy2;

                    if (conf >= confThreshold)
                    {
                        var det = new Detection
                        {
                            Box = new RectangleF(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY),
                            Confidence = conf
                        };
                        candidates.Add(det);
                    }

                }
            }

            //Sort and NMS
            candidates.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
            var results = new List<Detection>();

            while (candidates.Count > 0)
            {
                var current = candidates[0];
                results.Add(current);
                candidates.RemoveAt(0);

                for (int i = candidates.Count - 1; i >= 0; i--)
                {
                    if (ComputeIoU(current.Box, candidates[i].Box) > iouThreshold)
                    {
                        candidates.RemoveAt(i);
                    }
                }

            }

            return results;
        }

    }
}

 