using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using FaceAnalysisApp.FaceDetection;

namespace FaceAnalysisApp.Utils
{
    public static class ImageUtils
    {
        public static void DrawDetections(Bitmap image, List<FaceDetector.Detection> detections, string outputPath)
        {
            if (detections == null || detections.Count == 0)
            {
                Console.WriteLine("[INFO] No detections to draw.");
                return;
            }

            var bestDetection = detections.OrderByDescending(d => d.Confidence).First();
            var box = bestDetection.Box;

            using (var graphics = Graphics.FromImage(image))
            using (var font = new Font("Arial", 12))
            {
                Color color = Color.Green;
                using var pen = new Pen(color, 2);
                using var brush = new SolidBrush(color);

                graphics.DrawRectangle(pen, box.X, box.Y, box.Width, box.Height);

                graphics.DrawString($"{bestDetection.Confidence:F2}", font, brush, box.X, Math.Max(0, box.Y - 15));
            }

            image.Save(outputPath);
            Console.WriteLine($"[INFO] Detection image saved to: {outputPath}");
        }
    }
}
