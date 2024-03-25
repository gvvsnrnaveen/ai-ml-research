#ifndef __NAV_METRICS_H
#define __NAV_METRICS_H
#include <mlpack/core.hpp>
namespace naveen {
	class NavMetrics {
		public:
			NavMetrics();
			~NavMetrics();
			void calculateMetrics(const arma::Row<size_t> &trueLabels, const arma::Row<size_t> &predictions, double &precision, double &recall, double &f1score);
		private:
	};
}

#endif
