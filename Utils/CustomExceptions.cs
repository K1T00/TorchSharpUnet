using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils
{

	[Serializable]
	public class TrainImagesException : Exception
	{
		public TrainImagesException() { }

		public TrainImagesException(string message)
			: base(message)
		{

		}
	}
}
