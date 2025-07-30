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
            """
            Console.WriteLine("[INFO] Model loaded successfully");
            """
        }

        public void DetectFaces(string imagePath)
        {
            Console.WriteLine($"[INFO] Ready to detect faces in image: {imagePath}")
        }

    }
}

 