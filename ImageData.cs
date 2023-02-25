﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace TransferLearning
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }
}