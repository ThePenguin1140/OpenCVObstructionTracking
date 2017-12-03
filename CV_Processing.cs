using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Collections;
using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Windows.Forms;

namespace ShaprCVTest {
  class PreProcessing {
    //To keep track of which 'contours' need to be split
    //1 = no split, 2 = in half, etc.
    private static ArrayList BoxSplits;
    private static int[] CupNumsFound = new int[99];
    private static Rectangle[] NewBoxes = new Rectangle[99];
    private static int CNF_i = 0;


    public static bool HasParent( VectorOfVectorOfPoint currentContours, Rectangle checkMe ) {
      for ( int n = 0; n < currentContours.Size; n++ ) {
        Rectangle Box = CvInvoke.BoundingRectangle( currentContours[n] );

        if ( Box.Location.X < checkMe.Location.X &&
              Box.Location.Y < checkMe.Location.Y &&
              Box.Location.X + Box.Width > checkMe.Location.X + checkMe.Width &&
              Box.Location.Y + Box.Height > checkMe.Location.Y + checkMe.Width )

          return true;
      }

      return false;
    }

    private static int[,] ExtractContours( Image<Gray, byte> input, VectorOfVectorOfPoint output )
    {
      Image<Gray, byte> laplace_image = input.Canny(150, 50);
      Image<Gray, byte> erode_image = laplace_image.Dilate( 3 );
      Image<Gray, byte> thresholded_image = erode_image.ThresholdToZero( new Gray( 240 ) );
      thresholded_image = thresholded_image.SmoothGaussian(7);
      Image<Gray, byte> erode2_image = thresholded_image.Erode( 5 );
      erode2_image._ThresholdBinary( new Gray( 0 ), new Gray( 255 ));
      
      
//      CvInvoke.Imshow("laplace", laplace_image);
//      CvInvoke.Imshow("thresh", thresholded_image);
//      CvInvoke.Imshow("erode2", erode2_image);
      

      return CvInvoke.FindContourTree( erode2_image, output, ChainApproxMethod.ChainApproxSimple );
    }

    public static VectorOfVectorOfPoint GetContours( Image<Gray, byte> input ) {
      VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
      VectorOfVectorOfPoint contour2 = new VectorOfVectorOfPoint();

      BoxSplits = new ArrayList();

      int[,] tree = ExtractContours(input, contours);
      
      int[,] tre2 = new int[tree.Length / 4, 4];

      int t2id = 0;

      //Filter Countors: remove oddly sized ones
      for ( int i = 0; i < contours.Size; i++ ) {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

		    if ( ( box.Width < 400 && box.Height < 400 ) &&
			     ( box.Width > 50 && box.Height > 50 ) &&
			     ( box.Height > box.Width && box.Location.Y > CV_Program.MinY ) 
		         && !HasParent( contour2, CvInvoke.BoundingRectangle( contours[i] ) ) 
		         )
        {
              tre2[t2id, 0] = tree[i, 0];
        			tre2[t2id, 1] = tree[i, 1];
        			tre2[t2id, 2] = tree[i, 2];
        			tre2[t2id, 3] = tree[i, 3];

              if ( (float)box.Width / (float)box.Height > 0.95f )
              {
                BoxSplits.Add( 3 );
              } 
              else if ( (float)box.Width / (float)box.Height > 0.55f || (float)box.Width >= CV_Program.MinWidth)
              {
                BoxSplits.Add( 2 );
              } 
              else
              {
                BoxSplits.Add( 1 );
              }

          contour2.Push( contours[i] );
          t2id++;
        }
      }

      //Uncomment the following Console.WriteLines to display Bounding Box details
      //Console.WriteLine( "\nCV_Program: GetControus():\n" );

      for ( int n = 0; n < t2id; n++ ) {
        for ( int m = 0; m < 4; m++ ) {
          //Console.Write( tre2[n, m].ToString().PadLeft( 5 ) );
        }
        //Console.WriteLine();
      }

      return contour2;
    }

    public static void DrawContours( Image<Bgr, byte> output, VectorOfVectorOfPoint contours, Image<Gray, Byte> filtered_img = null ) {
      Bgr bgrRed = new Bgr( Color.Red );
      Bgr bgrBlu = new Bgr( Color.Blue );

      ResetCupsFound();

      ArrayList boxesFound = new ArrayList();

      for ( int i = 0; i < contours.Size; i++ )
      {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

        if ( true || GetMinDistance( box ) < 150 )
        {
          if ( (int)BoxSplits[i] == 1 || contours.Size >= 3 )
          {
            if (CV_Program.TrackCups && box.Height > CV_Program.MinHeight)
            {
              //box2 is assumed to be the one behind box1
              Rectangle box2 = new Rectangle( box.X, box.Y, box.Width, (int) (CV_Program.AvgHeight * 0.85f) ); 
              Rectangle box1 = new Rectangle( box.X, box.Y, box.Width, box.Height ); 
              
              int diff = box.Height - (int) CV_Program.MinHeight;
              box1.Height = (int)CV_Program.MinHeight;
              box1.Y += diff;

              
              int moveDirection = MovementDirection(filtered_img, box);
              
              Console.WriteLine( moveDirection );
              
              //The box to the back is movin right
              if (moveDirection == 1)
              {
                box1.Width = (int) CV_Program.AvgWidth;
                box2.Width = (int) CV_Program.AvgWidth;
                
                box2.X += box.Width - box2.Width;
              }
              //Moving left
              else if (moveDirection == -1)
              {
                box1.Width = (int) CV_Program.AvgWidth;
                box2.Width = (int) CV_Program.AvgWidth;
                
                box1.X += box.Width - box1.Width;
              }

              if (moveDirection == 0)
              {
                output.Draw( box1, bgrRed, 2 );
            
                output.Draw(new Rectangle(Center(box1).X, Center(box1).Y, 2, 2), bgrBlu, 2);

                boxesFound.Add( box1 );
              }
              else
              {
                output.Draw( box1, bgrRed, 2 );
                output.Draw( box2, bgrRed, 2 );
            
                output.Draw(new Rectangle(Center(box1).X, Center(box1).Y, 2, 2), bgrBlu, 2);
                output.Draw(new Rectangle(Center(box2).X, Center(box2).Y, 2, 2), bgrBlu, 2);

                boxesFound.Add( box1 );
                boxesFound.Add( box2 );
              }
            }
            else
            {
              //Draw the Box
              output.Draw( box, bgrRed, 2 );
          
              //Draw Center
              output.Draw(new Rectangle(Center(box).X, Center(box).Y, 2, 2), bgrBlu, 2);

              boxesFound.Add( box );
            }
          }
          else if ( (int)BoxSplits[i] == 2 )
          {
            //Code for splitting in half only
            Rectangle box1 = new Rectangle( box.X - 9, box.Y, box.Width / 2, box.Height );
            Rectangle box2 = new Rectangle( box.X + ( box.Width / 2 ) + 9, box.Y, box.Width / 2, box.Height );
            
            //We need to split this box
            //ClosestOldBox is the box from the previous frame that is closet to this one being split now
            Rectangle ClosestOldBox = GetClosestOldCupBoundingBox(box1);

            //So, if the old box and the new one start at a similar place
            //We will generate the first half to be as wide as the previous frame
            //otherwise, we generate the first half to be the difference between the old box and the new bigger one
            //this way, we don't simply split in half
            
            if (ClosestOldBox.X < (box.X + 10) && ClosestOldBox.X > (box.X - 10))
            {
              //Console.WriteLine("FIRST 1 1 1 1 1");
              box1 = new Rectangle(box.X - 1, ClosestOldBox.Y, ClosestOldBox.Width, ClosestOldBox.Height);
              //box2 = new Rectangle(box.X + box1.Width + 1, box.Y, box.Width - ClosestOldBox.Width, box.Height );

              int box2Y = GetClosestOldCupBoundingBox(box2).Y;
              if (box2Y == box1.Y) box2Y = box.Y;
              
              //box2 = new Rectangle((box.X + box.Width + 1)-(int)CV_Program.AvgWidth, GetClosestOldCupBoundingBox(box2).Y, (int)CV_Program.AvgWidth, (int)CV_Program.AvgHeight );
              box2 = new Rectangle((box.X + box.Width + 1)-(int)CV_Program.AvgWidth, box2Y, (int)CV_Program.AvgWidth, (int)CV_Program.AvgHeight );
            }
            else
            {
              //Console.WriteLine("SECOND 2 2 2 2 2");
              
              int box1Y = GetClosestOldCupBoundingBox(box1).Y;
              if (box1Y == ClosestOldBox.Y) box1Y = box.Y;
              //box1 = new Rectangle(box.X - 1, box.Y, box.Width - ClosestOldBox.Width, box.Height );
              //box1 = new Rectangle(box.X - 1, GetClosestOldCupBoundingBox(box1).Y, (int)CV_Program.AvgWidth, (int)CV_Program.AvgHeight );
              box1 = new Rectangle(box.X - 1, box.Y, (int)CV_Program.AvgWidth, (int)CV_Program.AvgHeight );
              box2 = new Rectangle(box.X + box1.Width + 1, ClosestOldBox.Y, ClosestOldBox.Width, ClosestOldBox.Height);
            }
            

            output.Draw( box1, bgrRed, 2 );
            output.Draw( box2, bgrRed, 2 );
            
            output.Draw(new Rectangle(Center(box1).X, Center(box1).Y, 2, 2), bgrBlu, 2);
            output.Draw(new Rectangle(Center(box2).X, Center(box2).Y, 2, 2), bgrBlu, 2);

            boxesFound.Add( box1 );
            boxesFound.Add( box2 );
          }
          else if ( (int)BoxSplits[i] == 3 )
          {
            Rectangle box1 = new Rectangle( box.X - 1, box.Y, box.Width / 3, box.Height );
            Rectangle box2 = new Rectangle( box.X + ( (box.Width / 3)*1 ) + 1, box.Y, box.Width / 3, box.Height );
            Rectangle box3 = new Rectangle( box.X + ( (box.Width / 3)*2 ) + 1, box.Y, box.Width / 3, box.Height );

            output.Draw( box1, bgrRed, 2 );
            output.Draw( box2, bgrRed, 2 );
            output.Draw( box3, bgrRed, 2 );
            
            output.Draw(new Rectangle(Center(box1).X, Center(box1).Y, 2, 2), bgrBlu, 2);
            output.Draw(new Rectangle(Center(box2).X, Center(box2).Y, 2, 2), bgrBlu, 2);
            output.Draw(new Rectangle(Center(box3).X, Center(box3).Y, 2, 2), bgrBlu, 2);

            boxesFound.Add( box1 );
            boxesFound.Add( box2 );
            boxesFound.Add( box3 );
          }
        }
      }

      if ( contours.Size != 0 ) ReLabelCups( boxesFound, output.Mat );
      if ( contours.Size != 0 ) UpdateCupTracking();
    }

    public static Rectangle GetClosestOldCupBoundingBox(Rectangle BoxBeingChecked)
    {
      float dist = GetDistance(CV_Program.Cups[0].BoundingBox, BoxBeingChecked);
      Rectangle ClosestOldBox = CV_Program.Cups[0].BoundingBox;
      
      for (int i = 1; i < 3; i++)
      {
        float newDist = GetDistance(CV_Program.Cups[i].BoundingBox, BoxBeingChecked);
        
        if (newDist < dist)
        {
          dist = newDist;
          ClosestOldBox = CV_Program.Cups[i].BoundingBox;
        }
      }
      
      return ClosestOldBox;
    }

    public static void ReLabelCups( ArrayList boxes, Mat frame)
    {
      CV_Cup[] prevCups = CV_Program.Cups;

      Rectangle AnyMissingbox = new Rectangle(888, 888, 8, 8);

      if ( CV_Program.TrackCups )
      {
        for ( int x = 0; x < 3; x++ )
        {
          float d = float.PositiveInfinity;
          Rectangle correctBB = new Rectangle();
          int id = -2;

          /*
           * Compares previous frame to current frame.
           * A bounding box is assigned to the cup with the smallest 
           * euclidian distance between it's bounding boxes.
           */
          foreach ( CV_Cup oldCup in prevCups ) //previous frame
          {
            if ( !IsIn( oldCup.CupID ) )
            {
              foreach ( Rectangle box in boxes ) //current frame
              {
                float newDist = GetDistance( box, oldCup.BoundingBox );
  
//                if ( newDist < d && newDist < 100)
                if( newDist < d )
                {
                  d = newDist;
                  correctBB = box;
                  id = oldCup.CupID;
                }
              }
            }
          }

          if ( id >= 0 )
          {
            boxes.Remove( correctBB );
            NewBoxes[CNF_i] = correctBB;
            CupNumsFound[CNF_i] = id;
            CNF_i++;
          }
        }

        /*
         * If we have fewer than 3 cups check which box is missing. 
         */
        if ( CNF_i < 3 && boxes.Count > 0)
        {
          int missing = WhichCupIsMissing();
          Console.WriteLine( "MISSING: " + missing );
          if ( missing != -1 ) 
          {
            NewBoxes[CNF_i] = (Rectangle)boxes[0];
            CupNumsFound[CNF_i] = missing;
            CNF_i++;
          }
        }

        for ( int i = 0; i < CNF_i; i++ )
        {
          if ( frame != null && CV_Program.TrackCups )
            CvInvoke.PutText( frame, "[" + ( CupNumsFound[i] + 1 ) + "]", new System.Drawing.Point( NewBoxes[i].Location.X + 5, NewBoxes[i].Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );
        }
        
        if ( CV_Program.ShowMinYLine ) CvInvoke.PutText( frame, "__MinY_______________________________________________________________", new System.Drawing.Point( 0, CV_Program.MinY ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );

        /*
        foreach ( Rectangle box in boxes )
        {
          float d = float.PositiveInfinity;
          Rectangle correctBB = new Rectangle();
          int id = -2;

          foreach ( CV_Cup oldCup in prevCups )
          {
            float newDist = GetDistance( box, oldCup.BoundingBox );

            if ( newDist < d && newDist < 2000 && !IsIn(oldCup.CupID))
            {
              d = newDist;
              correctBB = box;
              id = oldCup.CupID;
            }
          }

          if ( frame != null && CV_Program.TrackCups )
                CvInvoke.PutText( frame, "[" + ( id + 1 ) + "]", new System.Drawing.Point( correctBB.Location.X + 5, correctBB.Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );
          if ( id >= 0 )
          {
            NewBoxes[CNF_i] = correctBB;
            CupNumsFound[CNF_i] = id;
            CNF_i++;
          }
          else 
          {
            AnyMissingbox = box;
          }
        }

        if ( CNF_i < 3 )
        {
          int missing = WhichCupIsMissing();
          if ( missing != -1 ) 
          {
            NewBoxes[CNF_i] = AnyMissingbox;
            CupNumsFound[CNF_i] = missing;
            CNF_i++;
          }
        }
      */
      }  
    }

    public static int GetCupNum( Rectangle Box )
    {
      int n = -1;
      float d = -1.0f;

      for ( int i = 0; i < 3; i++ ) {
        float newDist = GetDistance( Box, CV_Program.Cups[i].BoundingBox );
        if ( !IsIn( i ) && ( newDist < d || d == -1.0f ) ) {
          n = i;
          d = newDist;
        }
      }

      if ( n >= 0 ) {
        NewBoxes[CNF_i] = Box;
        CupNumsFound[CNF_i] = n;
        CNF_i++;
      }

      return n;
    }

    public static float GetDistance( Rectangle rect1, Rectangle rect2 ) {
      Point center1 = Center( rect1 );
      Point center2 = Center( rect2 );

      float horizontalDistance = Math.Abs( ( center2.X - center1.X ) );
      float verticalDistance = Math.Abs( ( center2.Y - center1.Y ) );

      float distance = (float)Math.Sqrt( ( horizontalDistance * horizontalDistance ) + ( verticalDistance * verticalDistance ) );

      return distance;
    }

    public static float GetMinDistance( Rectangle box ) {
      float d = float.PositiveInfinity;
      if ( CV_Program.TrackCups ) {
        for ( int i = 0; i < 3; i++ ) {
          float newDist = GetDistance( box, CV_Program.Cups[i].BoundingBox );

          if ( newDist < d ) {
            d = newDist;
          }
        }
      }

      return d;
    }

    public static Point Center( Rectangle rect ) {
      return new Point( rect.Left + rect.Width / 2, rect.Top + rect.Height / 2 );
    }

    private static void ResetCupsFound() {
      CupNumsFound = new int[99];
      NewBoxes = new Rectangle[99];
      CNF_i = 0;

      for ( int i = 0; i < 99; i++ )
      {
        CupNumsFound[i] = -1;
      }
    }

    private static int WhichCupIsMissing() 
    {
      Console.Write( "WhichMissing: [" + CupNumsFound[0] + "] [" + CupNumsFound[1] + "] [" + CupNumsFound[2] + "]" );
      for ( int i = 0; i < 3; i++ )
      {
        bool found = false;

        for ( int j = 0; j < 3; j++ )
        {
          if ( CupNumsFound[j] == i ) found = true;
        }

        if ( !found )
          return i;
      }

      return -1;
    }

    private static bool IsIn( int x ) {
      for ( int i = 0; i < CNF_i; i++ ) {
        if ( x == CupNumsFound[i] ) return true;
      }
      return false;
    }

    public static void UpdateCupTracking() {
      bool[] found = new bool[3];
      found[0] = false;
      found[1] = false;
      found[2] = false;

      if ( CV_Program.TrackCups ) {
        for ( int i = NewBoxes.Length - 1; i >= 0; i-- ) {
          for ( int j = 0; j < 3; j++ ) {
            if ( CupNumsFound[i] == j ) { 
              CV_Program.Cups[j].BoundingBox = NewBoxes[i];
              found[j] = true;
              CV_Program.Cups[j].Missing = false;
            }
          }
        }
        
        if ( !found[0] ) CV_Program.Cups[0].BoundingBox = new Rectangle( CV_Program.Cups[0].BoundingBox.X, GetNewYForMissingBox(0), 
                                                                         CV_Program.Cups[0].BoundingBox.Width, 
                                                                         CV_Program.Cups[0].BoundingBox.Height);
        if ( !found[1] ) CV_Program.Cups[1].BoundingBox = new Rectangle( CV_Program.Cups[1].BoundingBox.X, GetNewYForMissingBox(1), 
                                                                         CV_Program.Cups[1].BoundingBox.Width,
                                                                         CV_Program.Cups[1].BoundingBox.Height);
        if ( !found[2] ) CV_Program.Cups[2].BoundingBox = new Rectangle( CV_Program.Cups[2].BoundingBox.X, GetNewYForMissingBox(2),  
                                                                         CV_Program.Cups[2].BoundingBox.Width,
                                                                         CV_Program.Cups[2].BoundingBox.Height);
        
      }
    }

    //Checks the Y location value of a bounding box
    //If it's divisible by 800, that means it was already moved up and we add another 800 to it
    //Else, it only just disappeared, so we set it to 800
    private static int GetNewYForMissingBox(int id)
    {
      if (!CV_Program.Cups[id].Missing)
      {
        CV_Program.Cups[id].Missing = true;
        return CV_Program.Cups[id].BoundingBox.Y;
      }
      if (CV_Program.Cups[id].BoundingBox.Y % 800 == 0)
      {
        return CV_Program.Cups[id].BoundingBox.Y + 800;
      }
      else
      {
        return 800;
      }
    }

    public static Image<Hsv, byte> Preprocess( Mat input, Size size, bool ShowHSV ) {

      //Resize image
      Image<Bgr, byte> resized_image = new Image<Bgr, byte>( size );
      CvInvoke.Resize( input, resized_image, size );

      //Causes a lot of lag between frames
      //CvInvoke.FastNlMeansDenoisingColored( resized_image, resized_image, 3, 3, 7, 21 );

      //Causes a bit of lag between frames
      resized_image = resized_image.SmoothGaussian( 15 );
      resized_image._GammaCorrect( 2.5 );
//      resized_image._EqualizeHist();

      resized_image = resized_image.Erode( 10 );
      resized_image = resized_image.Dilate( 10 );

      Image<Hsv, byte> output = new Image<Hsv, byte>( size );
      CvInvoke.CvtColor( resized_image, output, ColorConversion.Bgr2Hsv );

      if ( ShowHSV )
        CvInvoke.Imshow( "hsv", output );

      return output;
    }

    public static Image<Gray, byte> FilterCups( Image<Hsv, byte> input, bool ShowFiltered ) {
//      ScalarArray lower = new ScalarArray( new Hsv( 0, (255*0.5), 255 ).MCvScalar );
//      ScalarArray upper = new ScalarArray( new Hsv( (180* ( 360 / 328) ), 255, (255*0.75) ).MCvScalar );
//      
//      ScalarArray lower = new ScalarArray( new Hsv( 130, 0, 0 ).MCvScalar );
//      ScalarArray upper = new ScalarArray( new Hsv( 180, 200, 255 ).MCvScalar );

      ScalarArray lower = new ScalarArray( new Bgr( Color.Red ).MCvScalar );
      ScalarArray upper = new ScalarArray( new Bgr( Color.Purple).MCvScalar);

      Image<Gray, byte>[] channels = input.Split();

      CvInvoke.InRange( channels[0], new ScalarArray(20), new ScalarArray(160), channels[0] );

      channels[0]._Not();
//      channels[0]._ThresholdBinary( new Gray(200), new Gray(255.0));
      CvInvoke.BitwiseAnd( channels[0], channels[1], channels[0], null);

      channels[0]._ThresholdToZero( new Gray( 150 ) );
      
      if ( ShowFiltered )
        CvInvoke.Imshow( "Cup Filter", channels[0] );

//      CvInvoke.InRange( input, lower, upper, output );

      return channels[0];
    }

    public static Image<Gray, byte> FilterGlare( Image<Hsv, byte> input, bool ShowFiltered ) {
      Image<Gray, byte> output = new Image<Gray, byte>( input.Size );

      ScalarArray lower = new ScalarArray( new Hsv( 75, 0, 0 ).MCvScalar );
      ScalarArray upper = new ScalarArray( new Hsv( 180, 200, 255 ).MCvScalar );

      CvInvoke.InRange( input, lower, upper, output );

      if ( ShowFiltered )
        CvInvoke.Imshow( "Glare Filter", output );

      return output;
    }

    /*
     * Takes an input matrix and a region of interest in the matrix.
     * Calculates the side on which the top cup is in this ROI.
     * -1 is left
     * 0 is none
     * 1 is right
     */
    public static int MovementDirection( Image<Gray, byte> inputImage, Rectangle ROI )
    {
      int offset = 20;
      int direction = 0;
      bool debug = false;
      bool[] corners = new bool[4]; //top left, top right, bottom left, bottom right
      for (int i = 0; i < 4; i++)
      {
        corners[i] = false;
      }

      Image<Bgr, byte> debuggingImage = null;
      
      //shrink roi to cut off edges
      ROI.X += offset/2;
      ROI.Y += offset/2;
      ROI.Width -= offset;
      ROI.Height -= offset;
      inputImage.ROI = ROI;
      inputImage._Not();
      
      inputImage._Erode(5);
      
      if (debug)
      {  
        debuggingImage = new Image<Bgr, byte>( ROI.Size );
        CvInvoke.CvtColor(inputImage, debuggingImage, ColorConversion.Gray2Bgr);
      }
      
      
      VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
      ExtractContours(inputImage, contours);

      if (contours.Size < 2 || contours.Size > 3 ) return direction;
      
      for (int i = 0; i < contours.Size; i++)
      {
        Rectangle boundingBox = CvInvoke.BoundingRectangle(contours[i]);

        //make sure it's not the whole image
        if( boundingBox.Width + 1 == ROI.Width || boundingBox.Height + 1 == ROI.Height ) continue;

        if (boundingBox.X == 1 && boundingBox.Y == 1) corners[0] = true;
        else if (boundingBox.Y == 1 && boundingBox.Right + 1 == ROI.Width) corners[1] = true;
        else if (boundingBox.Bottom + 1 == ROI.Height && boundingBox.X == 1) corners[2] = true;
        else if (boundingBox.Bottom + 1 == ROI.Height && boundingBox.Right + 1 == ROI.Width) corners[3] = true;
        
        if (debug)
        {
          debuggingImage.Draw( boundingBox, new Bgr(Color.Red), 2);
        }
      }

      if (debug)
      {
        CvInvoke.Imshow("ROI", debuggingImage);
        CvInvoke.WaitKey(0);
        CvInvoke.DestroyWindow("ROI");
      }

      if (corners[0] || corners[3]) direction = 1;
      else if (corners[1] || corners[2]) direction = -1;

      if (    corners[0] && corners[1] 
           || corners[2] && corners[3]
           || corners[0] && corners[2]
           || corners[1] && corners[3] ) direction = 0;
      
      return direction;
    }
  }
}
