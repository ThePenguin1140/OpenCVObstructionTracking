using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
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
      
      Mat test_image = CvInvoke.Imread( "..\\..\\Images\\CornerTest1.jpg", LoadImageType.AnyColor );
      Rectangle roi = new Rectangle( 0, 0, test_image.Width, test_image.Height );
      int result = PreProcessing.MovementDirection(test_image.ToImage<Gray, byte>(), roi);
      
      Console.WriteLine( result );
      
    }
  }
}
