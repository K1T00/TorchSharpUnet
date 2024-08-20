using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using static Utils.HelperFunctions;

namespace Utils
{


    public class Db
    {
        private string modelPath;
        private string trainGrabsPath;
        private string trainGrabsPrePath;
        private string trainMasksPath;
		private string trainMasksPrePath;
        private string testGrabsPath;
		private string heatmapsTestPath;
        private List<Tuple<string, int>>datasetImages;
        private List<Tuple<string, int, int>>datasetMasks;
        private int downSampling;
        private int roi;
        private int amtDataset;
        private string projectPath;
        private bool withBoarderPadding;
        private int thresholdHeatmaps;

        public Db()
        {
            this.modelPath = "";
            this.trainGrabsPath = "";
            this.trainGrabsPrePath = "";
            this.trainMasksPath = "";
			this.trainMasksPrePath = "";
            this.testGrabsPath = "";
			this.heatmapsTestPath = "";
			this.datasetImages = new List<Tuple<string, int>>();
            this.datasetMasks = new List<Tuple<string, int, int>>();
            this.downSampling = 0;
            this.roi = 192;
            this.amtDataset = 0;
            this.projectPath = "";
			this.withBoarderPadding = false;
			this.thresholdHeatmaps = 0;
        }

        public Db(Db other)
        {
	        this.modelPath = other.modelPath;
	        this.trainGrabsPath = other.trainGrabsPath;
	        this.trainGrabsPrePath = other.trainGrabsPrePath;
	        this.trainMasksPath = other.trainMasksPath;
	        this.trainMasksPrePath = other.trainMasksPrePath;
	        this.testGrabsPath = other.testGrabsPath;
	        this.heatmapsTestPath = other.heatmapsTestPath;
	        this.datasetImages = other.datasetImages;
	        this.datasetMasks = other.datasetMasks;
	        this.amtDataset = other.amtDataset;
	        this.downSampling = other.downSampling;
	        this.roi = other.roi;
	        this.projectPath = other.projectPath;
			this.withBoarderPadding = other.withBoarderPadding;
			this.thresholdHeatmaps = other.thresholdHeatmaps;
        }
		public Db(string projDir, Phase phase, bool moreThanOneFeature)
        {
            this.modelPath = projDir + @"models\"; ;
            this.trainGrabsPath = projDir + @"train\grabs\";
            this.trainGrabsPrePath = projDir + @"train\grabsPre\";
            this.trainMasksPath = projDir + @"train\masks\";
            this.trainMasksPrePath = projDir + @"train\masksPre\";
            this.testGrabsPath = projDir + @"test\grabs\";
            this.heatmapsTestPath = projDir + @"test\heatmaps\";
            this.downSampling = 0;
            this.roi = 192;
            this.projectPath = projDir;
			this.withBoarderPadding = false;
			this.thresholdHeatmaps = 0;

			switch (phase)
            {
                case Phase.Train:

					this.datasetImages = SortImageFiles(Directory.GetFiles(this.trainGrabsPrePath, "*", SearchOption.TopDirectoryOnly));
					this.datasetMasks = SortMaskFiles(Directory.GetFiles(this.trainMasksPrePath, "*", SearchOption.TopDirectoryOnly), moreThanOneFeature);
					break;

                case Phase.ResizeTrain:

					this.datasetImages = SortImageFiles(Directory.GetFiles(this.trainGrabsPath, "*", SearchOption.TopDirectoryOnly));
					this.datasetMasks = SortMaskFiles(Directory.GetFiles(this.trainMasksPath, "*", SearchOption.TopDirectoryOnly), moreThanOneFeature);
					break;

                case Phase.Test:

					this.datasetImages = SortImageFiles(Directory.GetFiles(this.testGrabsPath, "*", SearchOption.TopDirectoryOnly));
					this.datasetMasks = new List<Tuple<string, int, int>>();
					break;

                default:

                    this.datasetImages = new List<Tuple<string, int>>();
                    this.datasetMasks = new List<Tuple<string, int, int>>();
                    break;
            }

            this.amtDataset = datasetImages.Count;
            }



		/// <summary>
		/// Model path
		/// </summary>
		[XmlIgnore]
		public string ModelPath
        {
            get => this.modelPath;
            set => this.modelPath = value;
        }

		/// <summary>
		/// Train images path
		/// </summary>
		[XmlIgnore]
		public string TrainGrabsPath
        {
            get => this.trainGrabsPath;
            set => this.trainGrabsPath = value;
        }

		/// <summary>
		/// Prepared train images path
		/// </summary>
		[XmlIgnore]
		public string TrainGrabsPrePath
        {
	        get => this.trainGrabsPrePath;
	        set => this.trainGrabsPrePath = value;
        }

		/// <summary>
		/// Validation images path
		/// </summary>
		[XmlIgnore]
		public string TrainMasksPath
        {
            get => this.trainMasksPath;
            set => this.trainMasksPath = value;
        }

		/// <summary>
		/// Prepared validation images path
		/// </summary>
		[XmlIgnore]
		public string TrainMasksPrePath
		{
            get => this.trainMasksPrePath;
            set => this.trainMasksPrePath = value;
        }

		/// <summary>
		/// Test images path
		/// </summary>
		[XmlIgnore]
		public string TestGrabsPath
        {
            get => this.testGrabsPath;
            set => this.testGrabsPath = value;
        }

		/// <summary>
		/// Test results heatmaps path
		/// </summary>
		[XmlIgnore]
		public string HeatmapsTestPath
        {
	        get => this.heatmapsTestPath;
	        set => this.heatmapsTestPath = value;
        }

        /// <summary>
        /// Threshold for heatmaps
        /// </summary>
        public int ThresholdHeatmaps
		{
	        get => this.thresholdHeatmaps;
	        set => this.thresholdHeatmaps = value;
        }

		/// <summary>
		/// Tuple of images filepath, image number
		/// </summary>
		[XmlIgnore]
		public List<Tuple<string, int>> DatasetImages
		{
	        get => this.datasetImages;
	        set => this.datasetImages = value;
        }

		/// <summary>
		/// Tuple of masks filepath, image number, mask/feature number
		/// </summary>
		[XmlIgnore]
		public List<Tuple<string, int, int>> DatasetMasks
		{
	        get => this.datasetMasks;
	        set => this.datasetMasks = value;
        }


		/// <summary>
		/// Amount of images used for training
		/// </summary>
		public int AmtDataset
		{
			get => this.amtDataset;
			set => this.amtDataset = value;
		}

		/// <summary>
		/// ROI used for batch creation
		/// </summary>
		public int Roi
		{
			get => this.roi;
			set => this.roi = value;
		}

		/// <summary>
		/// Down-sampling used for batch creation
		/// </summary>
		public int DownSampling
		{
			get => this.downSampling;
			set => this.downSampling = value;
		}

		/// <summary>
		/// Project path
		/// </summary>
		public string ProjectPath
		{
			get => this.projectPath;
			set => this.projectPath = value;
		}

		/// <summary>
		/// Use boarder padding
		/// </summary>
		public bool WithBoarderPadding
		{
			get => this.withBoarderPadding;
			set => this.withBoarderPadding = value;
		}

	}
}
