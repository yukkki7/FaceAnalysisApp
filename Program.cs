// See https://aka.ms/new-console-template for more information
using System.Drawing;
using FaceAnalysisApp.FaceDetection;
using Microsoft.ML.OnnxRuntime;

class Program
{
    static void Main()
    {
        var session = new InferenceSession("models/scrfd_500m.onnx");
        Console.WriteLine("Model inputs:");
        foreach (var name in session.InputMetadata.Keys)
        {
            Console.WriteLine($"  {name}");
        }


        //var baseDir = AppContext.BaseDirectory;
        //Console.WriteLine($"BaseDirectory: {baseDir}");
        //Console.WriteLine("Files under BaseDirectory:");
        //foreach (var file in Directory.GetFiles(baseDir))
        //    Console.WriteLine("  " + Path.GetFileName(file));
        //Console.WriteLine("Subdirs under BaseDirectory:");
        //foreach (var dir in Directory.GetDirectories(baseDir))
        //{
        //    Console.WriteLine("  " + Path.GetFileName(dir));
        //    foreach (var f in Directory.GetFiles(dir))
        //        Console.WriteLine("    " + Path.GetFileName(f));
        //}

        //// 停在这里，看看输出后再继续
        //return;
    }
}
