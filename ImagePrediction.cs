﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TransferLearning
{
    public class ImagePrediction : ImageData
    {
        public float[] Score { get; set; }

        public string PredictedLabelValue { get; set; }
    }
}