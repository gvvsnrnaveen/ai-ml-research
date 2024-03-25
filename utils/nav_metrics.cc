#include "nav_metrics.h"

naveen::NavMetrics::NavMetrics(){

}

naveen::NavMetrics::~NavMetrics(){

}

/**
 *  Confusion Matrix
 */

void naveen::NavMetrics::calculateMetrics(const arma::Row<size_t> &trueLabels, const arma::Row<size_t> &predictions, double &precision, double &recall, double &f1score){
	const size_t positiveLabel = 1;
	size_t truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;

	for(size_t i = 0; i < trueLabels.n_elem; i++){
		if(predictions(i) == positiveLabel && trueLabels(i) == positiveLabel){
			truePositive++;
		} 
		else if(predictions(i) != positiveLabel && trueLabels(i) == positiveLabel){
			falseNegative++;
		}
		else if(predictions(i) == positiveLabel && trueLabels(i) != positiveLabel){
			falsePositive++;
		}
		else if(predictions(i) != positiveLabel && trueLabels(i) != positiveLabel){
			trueNegative++;
		}
	}

	precision = truePositive / double(truePositive + falsePositive);
	recall = truePositive / double(truePositive + falseNegative);
	f1score = 2 * (precision * recall) / (precision + recall);
}
