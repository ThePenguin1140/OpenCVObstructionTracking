using System;
using System.Drawing;

namespace ShaprCVTest
{
    public class CV_Cup
    {
        public int CupID = -1;
        public Rectangle BoundingBox;

        public void Init(int ID, Rectangle Box)
        {
            CupID = ID;
            BoundingBox = Box;
        }
    }
}