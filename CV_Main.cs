using System;

namespace ShaprCVTest
{
  public class CV_Main
  {
    public static void Main()
    {
      //CV_Program.DetectCups_Image ("..\\..\\Images\\Cups1.jpg");
      //CV_Program.DetectCups_Video ("..\\..\\Videos\\Cups.mp4", true, true);
      CV_Program.DetectCups_Video ("..\\..\\Videos\\Sample2.mp4", true, true);

      //CV_Program.DetectCups_Video ();
    }
  }
}
