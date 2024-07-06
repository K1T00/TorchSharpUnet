using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.UNet
{
	public class UNetParameter
	{

		private bool useSavedModel;
		private int maxEpochs;
		private double learningRate;
		private int batchSize;
		private int amtClasses;
		private float stopAtLoss;

		public UNetParameter()
		{
			this.useSavedModel = false;
			this.maxEpochs = 0;
			this.learningRate = 0.003;
			this.batchSize = 20;
			this.amtClasses = 6;
			this.stopAtLoss = 0.45f;
		}

		public UNetParameter(UNetParameter other)
		{
			this.useSavedModel = other.useSavedModel;
			this.maxEpochs = other.maxEpochs;
			this.learningRate = other.learningRate;
			this.batchSize = other.batchSize;
			this.amtClasses = other.amtClasses;
			this.stopAtLoss = other.stopAtLoss;
		}

		/// <summary>
		/// Should be a new model trained or should be the weights being loaded/used from the model file
		/// </summary>
		public bool UseSavedModel
		{
			get => this.useSavedModel;
			set => this.useSavedModel = value;
		}

		/// <summary>
		/// If the model reaches this epoch number it should stop the training
		/// </summary>
		public int MaxEpochs
		{
			get => this.maxEpochs;
			set => this.maxEpochs = value;
		}

		/// <summary>
		/// Learning rate of the optimizer
		/// </summary>
		public double LearningRate
		{
			get => this.learningRate;
			set => this.learningRate = value;
		}

		/// <summary>
		/// Size of one batch (Tensor)
		/// </summary>
		public int BatchSize
		{
			get => this.batchSize;
			set => this.batchSize = value;
		}

		/// <summary>
		/// Amount of classes the model has
		/// </summary>
		public int AmtClasses
		{
			get => this.amtClasses;
			set => this.amtClasses = value;
		}

		/// <summary>
		/// At which loss the model should stop training
		/// </summary>
		public float StopAtLoss
		{
			get => this.stopAtLoss;
			set => this.stopAtLoss = value;
		}



	}
}
