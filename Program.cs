// See https://aka.ms/new-console-template for more information
using System.Drawing;
using FaceAnalysisApp.FaceDetection;
using Microsoft.ML.OnnxRuntime;


class Program
{
    static void Main()
    {
        string modelPath = Path.Combine(AppContext.BaseDirectory, "models", "scrfd_500m.onnx");
        Console.WriteLine($"[INFO] Loading model from: {modelPath}");

        var detector = new FaceDetector(modelPath);

        string imagePath = Path.Combine(AppContext.BaseDirectory,"test.jpg");
        Console.WriteLine($"[INFO] Loading image from: {imagePath}");

        if (!File.Exists(imagePath))
        {
            Console.WriteLine("[ERROR] test.jpg not found in output folder.");
            return;
        }

        using var image = new Bitmap(imagePath);

        float[] inputData = detector.Preprocess(image);
        var outputs = detector.RunInference(inputData);
        var detections = detector.Postprocess(outputs, image.Width, image.Height, confThreshold: 0.6f);

        Console.WriteLine($"Detected {detections.Count} faces:");
        foreach (var det in detections)
        {
            var box = det.Box;
            Console.WriteLine(
                $"  Box: ({box.X:F1},{box.Y:F1}) - ({box.X + box.Width:F1},{box.Y + box.Height:F1}), " +
                $"Conf: {det.Confidence:F2}"
            );
        }
        Console.WriteLine("[INFO] Detection completed.");
    }
}