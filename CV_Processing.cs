﻿using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Collections;

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

    public static VectorOfVectorOfPoint GetContours( Image<Gray, byte> input ) {
      VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
      VectorOfVectorOfPoint contour2 = new VectorOfVectorOfPoint();

      BoxSplits = new ArrayList();


      Image<Gray, float> laplace_image = input.Laplace( 3 );
      Image<Gray, float> erode_image = laplace_image.Erode( 2 );
      Image<Gray, byte> byteErode_image = erode_image.Convert<Gray, byte>();
      Image<Gray, byte> thresholded_image = byteErode_image.ThresholdToZero( new Gray( 240 ) );
      Image<Gray, byte> erode2_image = thresholded_image.Erode( 3 );

      int[,] tree = CvInvoke.FindContourTree( erode2_image, contours, ChainApproxMethod.ChainApproxSimple );
      int[,] tre2 = new int[tree.Length / 4, 4];

      int t2id = 0;

      //Filter Countors: remove oddly sized ones
      for ( int i = 0; i < contours.Size; i++ ) {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

		    if ( ( box.Width < 400 && box.Height < 400 ) &&
			     ( box.Width > 50 && box.Height > 175 ) &&
			     ( box.Height > box.Width && box.Location.Y > 100 ) && !HasParent( contour2, CvInvoke.BoundingRectangle( contours[i] ) ) )
           {
              tre2[t2id, 0] = tree[i, 0];
        			tre2[t2id, 1] = tree[i, 1];
        			tre2[t2id, 2] = tree[i, 2];
        			tre2[t2id, 3] = tree[i, 3];

              if ( (float)box.Width / (float)box.Height > 0.95f )
              {
                BoxSplits.Add( 3 );
              } 
              else if ( (float)box.Width / (float)box.Height > 0.55f )
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

    public static void DrawContours( Image<Bgr, byte> output, VectorOfVectorOfPoint contours, Mat frame = null ) {
      Bgr bgrRed = new Bgr( Color.Red );

      ResetCupsFound();

      ArrayList boxesFound = new ArrayList();

      for ( int i = 0; i < contours.Size; i++ )
      {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

        if ( true || GetMinDistance( box ) < 150 )
        {
          if ( (int)BoxSplits[i] == 1 )
          {
            output.Draw( box, bgrRed, 2 );

            boxesFound.Add( box );
          }
          else if ( (int)BoxSplits[i] == 2 )
          {
            Rectangle box1 = new Rectangle( box.X - 1, box.Y, box.Width / 2, box.Height );
            Rectangle box2 = new Rectangle( box.X + ( box.Width / 2 ) + 1, box.Y, box.Width / 2, box.Height );

            output.Draw( box1, bgrRed, 2 );
            output.Draw( box2, bgrRed, 2 );

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

            boxesFound.Add( box1 );
            boxesFound.Add( box2 );
            boxesFound.Add( box3 );
          }
        }
      }

      if ( contours.Size != 0 ) ReLabelCups( boxesFound, frame );
      if ( contours.Size != 0 ) UpdateCupTracking();
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

          foreach ( CV_Cup oldCup in prevCups )
          {
            foreach ( Rectangle box in boxes )
            {
              float newDist = GetDistance( box, oldCup.BoundingBox );

              if ( newDist<d && newDist< 100 && !IsIn(oldCup.CupID))
              {
                d = newDist;
                correctBB = box;
                id = oldCup.CupID;
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
            }
          }
        }

        if ( !found[0] ) CV_Program.Cups[0].BoundingBox = new Rectangle( CV_Program.Cups[0].BoundingBox.X, 9000, 
                                                                         CV_Program.Cups[0].BoundingBox.Width, 
                                                                         CV_Program.Cups[0].BoundingBox.Height);
        if ( !found[1] ) CV_Program.Cups[1].BoundingBox = new Rectangle( CV_Program.Cups[1].BoundingBox.X, 9000, 
                                                                         CV_Program.Cups[1].BoundingBox.Width,
                                                                         CV_Program.Cups[1].BoundingBox.Height);
        if ( !found[2] ) CV_Program.Cups[2].BoundingBox = new Rectangle( CV_Program.Cups[2].BoundingBox.X, 9000,  
                                                                         CV_Program.Cups[2].BoundingBox.Width,
                                                                         CV_Program.Cups[2].BoundingBox.Height);
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
      resized_image._EqualizeHist();

      resized_image = resized_image.Erode( 10 );
      resized_image = resized_image.Dilate( 10 );

      Image<Hsv, byte> output = new Image<Hsv, byte>( size );
      CvInvoke.CvtColor( resized_image, output, ColorConversion.Bgr2Hsv );

      if ( ShowHSV )
        CvInvoke.Imshow( "hsv", output );

      return output;
    }

    public static Image<Gray, byte> FilterCups( Image<Hsv, byte> input, bool ShowFiltered ) {
      ScalarArray lower = new ScalarArray( new Hsv( 0, 0, 0 ).MCvScalar );
      ScalarArray upper = new ScalarArray( new Hsv( 35, 255, 255 ).MCvScalar );

      Image<Gray, byte> output = new Image<Gray, byte>( input.Size );

      CvInvoke.InRange( input, lower, upper, output );

      if ( ShowFiltered )
        CvInvoke.Imshow( "Cup Filter", output );

      return output;
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

  }
}
