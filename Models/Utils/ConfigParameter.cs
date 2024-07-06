using Models.UNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.Utils
{
    public class ConfigParameter
    {

        private string modelPath;
        private string newModelPath;
        private string trainGrabsPath;
        private string trainMasksPath;
        private string validateGrabsPath;
        private string validateMasksPath;
        private string heatmapsPath;


        public ConfigParameter()
        {
            this.modelPath = "";
            this.newModelPath = "";
            this.trainGrabsPath = "";
            this.trainMasksPath = "";
            this.validateGrabsPath = "";
            this.validateMasksPath = "";
            this.heatmapsPath = "";
        }

        public ConfigParameter(ConfigParameter other)
        {
            this.modelPath = other.modelPath;
            this.newModelPath = other.newModelPath;
            this.trainGrabsPath = other.trainGrabsPath;
            this.trainMasksPath = other.trainMasksPath;
            this.validateGrabsPath = other.validateGrabsPath;
            this.validateMasksPath = other.validateMasksPath;
            this.heatmapsPath = other.heatmapsPath;
        }

        /// <summary>
        /// Last Model path
        /// </summary>
        public string ModelPath
        {
            get => this.modelPath;
            set => this.modelPath = value;
        }

        /// <summary>
        /// Actual Model path
        /// </summary>
        public string NewModelPath
        {
            get => this.newModelPath;
            set => this.newModelPath = value;
        }

        /// <summary>
        /// Train images path
        /// </summary>
        public string TrainGrabsPath
        {
            get => this.trainGrabsPath;
            set => this.trainGrabsPath = value;
        }

        /// <summary>
        /// Train masks path
        /// </summary>
        public string TrainMasksPath
        {
            get => this.trainMasksPath;
            set => this.trainMasksPath = value;
        }

        /// <summary>
        /// Validation images path
        /// </summary>
        public string ValidateGrabsPath
        {
            get => this.validateGrabsPath;
            set => this.validateGrabsPath = value;
        }

        /// <summary>
        /// Validation masks path
        /// </summary>
        public string ValidateMasksPath
        {
            get => this.validateMasksPath;
            set => this.validateMasksPath = value;
        }

        /// <summary>
        /// Resulting heatmaps path
        /// </summary>
        public string HeatmapsPath
        {
            get => this.heatmapsPath;
            set => this.heatmapsPath = value;
        }




    }
}
