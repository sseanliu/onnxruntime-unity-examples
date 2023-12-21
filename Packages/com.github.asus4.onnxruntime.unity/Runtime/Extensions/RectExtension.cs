using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Unity
{
    public static class RectExtension
    {
        public static float IntersectionOverUnion(this in Rect rect0, in Rect rect1)
        {
            float sx0 = rect0.xMin;
            float sy0 = rect0.yMin;
            float ex0 = rect0.xMax;
            float ey0 = rect0.yMax;
            float sx1 = rect1.xMin;
            float sy1 = rect1.yMin;
            float ex1 = rect1.xMax;
            float ey1 = rect1.yMax;

            float xmin0 = Mathf.Min(sx0, ex0);
            float ymin0 = Mathf.Min(sy0, ey0);
            float xmax0 = Mathf.Max(sx0, ex0);
            float ymax0 = Mathf.Max(sy0, ey0);
            float xmin1 = Mathf.Min(sx1, ex1);
            float ymin1 = Mathf.Min(sy1, ey1);
            float xmax1 = Mathf.Max(sx1, ex1);
            float ymax1 = Mathf.Max(sy1, ey1);

            float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
            float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
            if (area0 <= 0 || area1 <= 0)
            {
                return 0.0f;
            }

            float intersect_xmin = Mathf.Max(xmin0, xmin1);
            float intersect_ymin = Mathf.Max(ymin0, ymin1);
            float intersect_xmax = Mathf.Min(xmax0, xmax1);
            float intersect_ymax = Mathf.Min(ymax0, ymax1);

            float intersect_area = Mathf.Max(intersect_ymax - intersect_ymin, 0.0f) *
                                   Mathf.Max(intersect_xmax - intersect_xmin, 0.0f);

            return intersect_area / (area0 + area1 - intersect_area);
        }
    }
}