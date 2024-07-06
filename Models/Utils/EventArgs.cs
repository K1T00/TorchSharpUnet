using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.Utils
{
	public class LossDataAvailableEventArgs(int epoch, float trainLoss, float valLoss) : EventArgs
	{
		public int Epoch { get; } = epoch;
		public float TrainLoss { get; } = trainLoss;
		public float ValLoss { get; } = valLoss;

	}

	public class LogDataAvailableEventArgs(string logText) : EventArgs
	{
		public string LogText { get; } = logText;

	}
}
