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
    public class FaceDetector: IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _inputW;
        private readonly int _inputH;

        public FaceDetector(string modelPath, int inputWidth = 640, int inputHeight = 640)
        {
            _session = new InferenceSession(modelPath);
            _inputW = inputWidth;
            _inputH = inputHeight;

            Console.WriteLine("[INFO] Model loaded successfully");

        }

        public List<(RectangleF, float score)> Detect(Bitmap image, float confThreshold = 0.6f, float nmsThreshold = 0.4f)
        {
            var inputTensor = Preprocess(image, _inputW, _inputH);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input.1", inputTensor) };

            using var results = _session.Run(inputs);

            var score8 = results.First(r => r.Name.EndsWith("score_8")).AsTensor<float>();
            var score16 = results.First(r => r.Name.EndsWith("score_16")).AsTensor<float>();
            var score32 = results.First(r => r.Name.EndsWith("score_32")).AsTensor<float>();
            var bbox8 = results.First(r => r.Name.EndsWith("bbox_8")).AsTensor<float>();
            var bbox16 = results.First(r => r.Name.EndsWith("bbox_16")).AsTensor<float>();
            var bbox32 = results.First(r => r.Name.EndsWith("bbox_32")).AsTensor<float>();

            var dets = new List<(RectangleF box, float score)>();

            float scale = Math.Min((float)_inputW / image.Width, (float)_inputH / image.Height);
            int padW = (_inputW - (int)(image.Width * scale)) / 2;
            int padH = (_inputH - (int)(image.Height * scale)) / 2;

            void Collect(Tensor<float> scores, Tensor<float> boxes)
            {
                int N = scores.Dimensions[1];
                for (int i = 0; i < N; i++)
                {
                    float s = scores[0, i, 0];
                    if (s < confThreshold) continue;

                    float l = boxes[0, i, 0], t = boxes[0, i, 1];
                    float r = boxes[0, i, 2], b = boxes[0, i, 3];

                    float x1 = (l * _inputW - padW) / scale;
                    float y1 = (t * _inputH - padH) / scale;
                    float x2 = (r * _inputW - padW) / scale;
                    float y2 = (b * _inputH - padH) / scale;

                    dets.Add((new RectangleF(x1, y1, x2 - x1, y2 - y1), s));
                }
            }

            Collect(score8, bbox8);
            Collect(score16, bbox16);
            Collect(score32, bbox32);

            var kept = new List<(RectangleF box, float score)>();
            foreach (var det in dets.OrderByDescending(d => d.score))
            {
                if (kept.All(k => IoU(k.box, det.box) <= nmsThreshold))
                    kept.Add(det);
            }
            return kept;
        }



        public static float IoU(RectangleF a, RectangleF b)
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


        public static Tensor<float> Preprocess(Bitmap src, int inputW, int inputH)
        {
            float scale = Math.Min((float)inputW / src.Width, (float)inputH / src.Height);
            int resizedW = (int)Math.Round(src.Width * scale);
            int resizedH = (int)Math.Round(src.Height * scale);

            var resized = new Bitmap(src, new Size(resizedW, resizedH));

            var canvas = new Bitmap(inputW, inputH);
            using (var g = Graphics.FromImage(canvas))
            {
                g.Clear(Color.FromArgb(114, 114, 114));
                int padW = (inputW - resizedW) / 2;
                int padH = (inputH - resizedH) / 2;
                g.DrawImage(resized, padW, padH, resizedW, resizedH);
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, inputH, inputW });
            for (int y = 0; y < inputH; y++)
                for (int x = 0; x < inputW; x++)
                {
                    var pixel = canvas.GetPixel(x, y);
                    tensor[0, 0, y, x] = (pixel.B / 255f - 0.485f) / 0.229f;
                    tensor[0, 1, y, x] = (pixel.G / 255f - 0.456f) / 0.224f;
                    tensor[0, 2, y, x] = (pixel.R / 255f - 0.406f) / 0.225f;
                }


            return tensor;

        }

  

        public List<RectangleF> Postprocess(Tensor<float> scores, Tensor<float> boxes, 
                                            int origW, int origH, int inputW, int inputH,
                                            float confThresh, float nmsThresh)
        {
            var detections = new List<(RectangleF box, float score)>();
            float scale = Math.Min((float)inputW / origW, (float)inputH / origH);
            int padW = (inputW - (int)Math.Round(origW * scale)) / 2;
            int padH = (inputH - (int)Math.Round(origH * scale)) / 2;

            int count = scores.Dimensions[1];
            for (int i = 0; i < count; i++)
            {
                float score = scores[0, i, 0];
                if (score < confThresh) continue;
                float l = boxes[0, i, 0];
                float t = boxes[0, i, 1];
                float r = boxes[0, i, 2];
                float b = boxes[0, i, 3];

                float x1 = (l * inputW - padW) / scale;
                float y1 = (t * inputH - padH) / scale;
                float x2 = (r * inputW - padW) / scale;
                float y2 = (b * inputH - padH) / scale;

                detections.Add((new RectangleF(x1, y1, x2 - x1, y2 - y1), score));
            }

            var resultsList = new List<RectangleF>();
            foreach (var det in detections.OrderByDescending(d => d.score))
            {
                bool keep = true;
                foreach (var prev in resultsList)
                {
                    if (IoU(det.box, prev) > nmsThresh)
                    {
                        keep = false;
                        break;
                    }
                }
                if (keep) resultsList.Add(det.box);
            }
            return resultsList;
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}