using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Shape;
using Emgu.CV.Structure;

namespace ShaprCVTest
{
  public class CV_Main
  {
    public static void Main()
    {
      //CV_Program.DetectCups_Image ("..\\..\\Images\\Cups1.jpg");
      //CV_Program.DetectCups_Video ("..\\..\\Videos\\Cups.mp4", true, true);
//      CV_Program.DetectCups_Video ("..\\..\\Videos\\Sample5.mp4", true, true);

      //CV_Program.DetectCups_Video ();

      testDirections();

    }

    private static void testDirections()
    {
      string imageFolder = "..\\..\\Images\\directionTestFiles";
      DirectoryInfo d = new DirectoryInfo(imageFolder);
      int count = 0;
      int passes = 0;
      foreach (FileInfo file in d.GetFiles("*"))
      {
        count++;
        string fileName = file.Name;
        string expectedResult = fileName.Split('_')[1].Split('.')[0].Trim();
        Mat testFile = CvInvoke.Imread(file.FullName, LoadImageType.Grayscale);
        Rectangle roi = new Rectangle( 0, 0, testFile.Width, testFile.Height );
        int result = PreProcessing.MovementDirection(testFile.ToImage<Gray, byte>(), roi);

        if (result == 1 && expectedResult == "right")
        {
          passes++;
          Console.WriteLine("PASS | " + fileName);
        }
        else if (result == -1 && expectedResult == "left")
        {
          passes++;
          Console.WriteLine("PASS | " + fileName);
        }
        else Console.WriteLine("FAIL ---- Expected | " + expectedResult + " Recieved | " + result);
      }
        
      Console.Write("____________________________________________________________________\n" +
                    "" + passes + "/" + count + " of tests passed.");
    }
  }
}
