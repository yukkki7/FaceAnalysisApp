using System;
using System.Drawing;
using System.IO;
using FaceAnalysisApp.FaceDetection;

class Program
{
    static void Main(string[] args)
    {
        string modelPath = Path.Combine(AppContext.BaseDirectory, "models", "scrfd_500m_bnkps.onnx");
        Console.WriteLine($"[INFO] Loading model from: {modelPath}");
        if (!File.Exists(modelPath))
        {
            Console.WriteLine("[ERROR] Model file not found.");
            return;
        }

        using var detector = new FaceDetector(modelPath, inputWidth: 640, inputHeight: 640);

        string imagePath = Path.Combine(AppContext.BaseDirectory, "test.jpg");
        Console.WriteLine($"[INFO] Loading image from: {imagePath}");
        if (!File.Exists(imagePath))
        {
            Console.WriteLine("[ERROR] test.jpg not found.");
            return;
        }

        using var image = (Bitmap)Image.FromFile(imagePath);

        var dets = detector.Detect(image, confThreshold: 0.02f, nmsThreshold: 0.4f);

        string outputPath = Path.Combine(AppContext.BaseDirectory, "output.jpg");
        using var imageCopy = new Bitmap(image);
        using var g = Graphics.FromImage(imageCopy);
        using var pen = new Pen(Color.Red, 2);

        if (dets.Count > 0)
        {
            var best = dets[0];
            var temp = dets[0];
            var box = temp.Item1;
            var score = temp.Item2;
            g.DrawRectangle(pen, box.X, box.Y, box.Width, box.Height);
            Console.WriteLine($"Best score = {score:F2}");
        }
        else
        {
            Console.WriteLine("[INFO] No face above threshold.");
        }

        imageCopy.Save(outputPath);
        Console.WriteLine($"[INFO] Result saved to: {outputPath}");
    }
}
