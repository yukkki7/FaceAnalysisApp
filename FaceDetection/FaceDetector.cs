using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;



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

        private float[] Preprocess(Bitmap image)
        {
            Bitmap resized = new Bitmap(image, InputWidth, InputHeight);
            float[] data = new float[InputWidth * InputHeight * 3];

            for (int y = 0; y < InputHeight; y++)
            {
                for (int x =0;x < InputWidth; x++)
                {
                    int baseIndex = y * InputWidth + y;
                    Color pixel = resized.GetPixel(x, y);
                    data[baseIndex + 0 * InputWidth * InputHeight] = pixel.B;
                    data[baseIndex + 1 * InputWidth * InputHeight] = pixel.G;
                    data[baseIndex + 2 * InputWidth * InputHeight] = pixel.R;
                }
            }
            return data;
        }

        //private Dictionary<string, Tensor<float>> RunInference(float[] data)
        //{
        //    int dimensions = new int[] { 1, 3, 640, 640};
        //    var tensor = new DenseTensor<float>(data, dimensions);
        //    return outputs
        //}

    }
}

 