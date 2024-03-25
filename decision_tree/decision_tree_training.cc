#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/core/data/split_data.hpp>

#include "nav_metrics.h"

#define DECISION_TREE_MODEL_FILE "model/hotel_reservation_dt.bin"
#define DECISION_TREE_TRAINING_DATASET "../datasets/hotel_booking_arma.csv"

int main()
{
	arma::mat dataset;
	arma::field<std::string> header(19);

	header(0) = "Booking_ID";
	header(1) = "no_of_adults";
	header(2) = "no_of_children";
	header(3) = "no_of_weekend_nights";
	header(4) = "no_of_week_nights";
	header(5) = "type_of_meal_plan";
	header(6) = "required_car_parking_space";
	header(7) = "room_type_reserved";
	header(8) = "lead_time";
	header(9) = "arrival_year";
	header(10) = "arrival_month";
	header(11) = "arrival_date";
	header(12) = "market_segment_type";
	header(13) = "repeated_guest";
	header(14) = "no_of_previous_cancellations";
	header(15) = "no_of_previous_bookings_1";
	header(16) = "avg_price_per_room";
	header(17) = "no_of_special_requests";
	header(18) = "booking_status";

	dataset.load(arma::csv_name(DECISION_TREE_TRAINING_DATASET, header));

	// Removing the Booking ID Column
	dataset.shed_col(0);
	dataset.brief_print();

	arma::mat features;
	features = dataset.submat(0, 0, dataset.n_rows - 1, dataset.n_cols - 2);

	// perform matrix transpose as per arma
	features = features.t();
	features.brief_print();

	// extract the labels from the dataset
	// last column is set as labels, tune accordingly
	arma::Row<size_t> labels = arma::conv_to<arma::Row<size_t>>::from(dataset.col(dataset.n_cols - 1));
	labels.brief_print();

	// data split ratio is 0.8 training and 0.2 testing
	const double splitRatio = 0.2;

	// perform data splitting
	arma::mat trainFeatures, testFeatures;
	arma::Row<size_t> trainLabels, testLabels;
	mlpack::data::Split(features, labels, trainFeatures, testFeatures, trainLabels, testLabels, splitRatio);

	std::cout << "Training Features Matrix: " << std::endl;
	trainFeatures.brief_print();

	std::cout << "Test Features Matrix: " << std::endl;
	testFeatures.brief_print();

	std::cout << "Training Labels Matrix: " << std::endl;
	trainLabels.brief_print();

	std::cout << "Test Labels Matrix: " << std::endl;
	testLabels.brief_print();
	
	// Train the Decision tree
	mlpack::tree::DecisionTree<> dt(trainFeatures, trainLabels, 2, 10);

	// Save the model to .bin file for future loading into inference
	mlpack::data::Save(DECISION_TREE_MODEL_FILE, "model", dt, false);


	// Generate the metrics with test data
	arma::Row<size_t> predictions;
	dt.Classify(testFeatures, predictions);

	const double accuracy = arma::accu(predictions == testLabels) / double(testLabels.n_elem);
	std::cout << "accuracy: " << accuracy << std::endl;

	double precision = 0, recall = 0, f1score = 0;

	naveen::NavMetrics *metrics = new naveen::NavMetrics();
	metrics->calculateMetrics(testLabels, predictions, precision, recall, f1score);
	std::cout << "Precision: " << precision << ", Recall: " << recall << ", F1score: " << f1score << std::endl;

	delete metrics;
	
	return 0;
}

